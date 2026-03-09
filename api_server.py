import base64
import copy
import io
import json
import os
import re
import sys
import time
import threading
from datetime import datetime
from typing import Any, Dict, Tuple, List, Optional
from queue import Queue

import numpy as np
import soundfile as sf
import subprocess
import tempfile
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from qwen_tts import Qwen3TTSModel

# 本服务支持所有模型（CustomVoice / VoiceDesign / Base 语音克隆），对应前端页面为 web/index.html
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
WEB_DIR = os.path.join(APP_DIR, "web")
INDEX_FILE = os.path.join(WEB_DIR, "index.html")  # 全功能页面
TASKS_DIR = os.path.join(APP_DIR, "tasks")
AUDIO_OUTPUT_DIR = os.path.join(TASKS_DIR, "audio")
SCRIPTS_DIR = os.path.join(APP_DIR, "scripts")

# 参考音频时长限制（秒）
REF_AUDIO_DURATION_MIN = 5.0
REF_AUDIO_DURATION_MAX = 15.0

# 确保任务目录存在
os.makedirs(TASKS_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


def _resolve_model_path(model_type: str, model_size: str) -> str:
    model_type_map = {
        "custom_voice": "CustomVoice",
        "voice_design": "VoiceDesign",
        "base": "Base",
    }
    if model_type not in model_type_map:
        raise ValueError(f"Unsupported model_type: {model_type}")
    if model_size not in ("0.6B", "1.7B"):
        raise ValueError(f"Unsupported model_size: {model_size}")
    model_name = f"Qwen3-TTS-12Hz-{model_size}-{model_type_map[model_type]}"
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def _default_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    return "cpu", torch.float32


_MODEL_CACHE: Dict[Tuple[str, str], Qwen3TTSModel] = {}
_WHISPERX_AVAILABLE: Optional[bool] = None
_TASKS_STORE: Dict[str, Dict] = {}  # 内存中的任务存储
_TASKS_LOCK = threading.Lock()  # 任务存储的线程锁
_TASK_QUEUE = Queue()  # 任务队列
_QUEUE_WORKER_STARTED = False  # 标记队列工作线程是否已启动


def _generate_task_id() -> str:
    """生成短任务ID：时间戳后6位 + 3位随机数"""
    timestamp = str(int(time.time() * 1000))[-6:]  # 取毫秒时间戳后6位
    import random
    random_suffix = str(random.randint(100, 999))
    return f"{timestamp}{random_suffix}"


def _get_task_key(user_id: str, task_id: str) -> str:
    """生成任务的唯一键"""
    return f"{user_id}:{task_id}"


def _save_task(user_id: str, task_id: str, task_data: Dict) -> None:
    """保存任务信息"""
    with _TASKS_LOCK:
        key = _get_task_key(user_id, task_id)
        _TASKS_STORE[key] = task_data
        # 同时保存到文件，以便重启后恢复
        task_file = os.path.join(TASKS_DIR, f"{user_id}_{task_id}.json")
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task_data, f, ensure_ascii=False, indent=2)


def _get_task(user_id: str, task_id: str) -> Optional[Dict]:
    """获取任务信息"""
    with _TASKS_LOCK:
        key = _get_task_key(user_id, task_id)
        if key in _TASKS_STORE:
            return _TASKS_STORE[key]
        # 尝试从文件加载
        task_file = os.path.join(TASKS_DIR, f"{user_id}_{task_id}.json")
        if os.path.exists(task_file):
            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)
                _TASKS_STORE[key] = task_data
                return task_data
    return None


def _delete_task(user_id: str, task_id: str) -> bool:
    """删除任务：从内存和文件移除，并删除音频文件。返回是否成功。"""
    task_data = _get_task(user_id, task_id)
    if task_data and task_data.get("audio_file"):
        audio_path = os.path.join(AUDIO_OUTPUT_DIR, task_data["audio_file"])
        srt_path, json_path = _subtitle_paths_for_audio(audio_path)
        for path in (audio_path, srt_path, json_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    with _TASKS_LOCK:
        key = _get_task_key(user_id, task_id)
        if key in _TASKS_STORE:
            del _TASKS_STORE[key]
    task_file = os.path.join(TASKS_DIR, f"{user_id}_{task_id}.json")
    if os.path.exists(task_file):
        try:
            os.remove(task_file)
        except OSError:
            pass
    return True


def _get_user_tasks(user_id: str) -> List[Dict]:
    """获取用户的所有任务"""
    with _TASKS_LOCK:
        user_tasks = []
        # 从内存中查找
        for key, task in _TASKS_STORE.items():
            if key.startswith(f"{user_id}:"):
                user_tasks.append(task)
        # 从文件中查找
        for filename in os.listdir(TASKS_DIR):
            if filename.startswith(f"{user_id}_") and filename.endswith(".json"):
                task_file = os.path.join(TASKS_DIR, filename)
                with open(task_file, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                    key = _get_task_key(task_data["user_id"], task_data["task_id"])
                    if key not in _TASKS_STORE:
                        _TASKS_STORE[key] = task_data
                        user_tasks.append(task_data)
        # 按创建时间倒序排列
        user_tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return user_tasks


def _queue_worker():
    """队列工作线程，依次处理任务"""
    print("[Queue Worker] 任务队列工作线程已启动")
    while True:
        try:
            # 从队列获取任务（阻塞等待）
            task_item = _TASK_QUEUE.get()
            
            if task_item is None:  # None 作为停止信号
                print("[Queue Worker] 收到停止信号，退出")
                break
            
            user_id = task_item["user_id"]
            task_id = task_item["task_id"]
            params = task_item["params"]
            
            print(f"[Queue Worker] 开始处理任务 {task_id} (用户: {user_id})")
            print(f"[Queue Worker] 队列中剩余任务数: {_TASK_QUEUE.qsize()}")
            
            # 处理任务
            _process_voice_clone_task(user_id, task_id, params)
            
            # 标记任务完成
            _TASK_QUEUE.task_done()
            
            print(f"[Queue Worker] 任务 {task_id} 处理完成")
            
        except Exception as e:
            print(f"[Queue Worker] 处理任务时出错: {str(e)}")
            import traceback
            traceback.print_exc()


def _start_queue_worker():
    """启动队列工作线程"""
    global _QUEUE_WORKER_STARTED
    if not _QUEUE_WORKER_STARTED:
        worker_thread = threading.Thread(target=_queue_worker, daemon=True)
        worker_thread.start()
        _QUEUE_WORKER_STARTED = True
        print("[Queue Worker] 队列工作线程初始化完成")


def _get_model(model_type: str, model_size: str) -> Qwen3TTSModel:
    key = (model_type, model_size)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model_path = _resolve_model_path(model_type, model_size)
    device, dtype = _default_device_and_dtype()
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation="eager",
    )
    _MODEL_CACHE[key] = model
    return model


def _build_atempo_filter(speed: float) -> str:
    # ffmpeg atempo supports 0.5 - 2.0. Chain to reach wider range.
    speed = float(speed)
    if speed <= 0:
        return "atempo=1.0"
    filters = []
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5
    filters.append(f"atempo={speed:.6f}")
    return ",".join(filters)


def _apply_speed_ffmpeg(wav: np.ndarray, sr: int, speed: float) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.wav")
        out_path = os.path.join(td, "out.wav")
        sf.write(in_path, wav, sr, format="WAV")
        flt = _build_atempo_filter(speed)
        # Pad after time-stretch to avoid clipping the last phoneme.
        flt = f"{flt},apad=pad_dur=0.4"
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            in_path,
            "-filter:a",
            flt,
            out_path,
        ]
        subprocess.run(cmd, check=True)
        out_wav, _ = sf.read(out_path, dtype="float32", always_2d=False)
        if out_wav.ndim > 1:
            out_wav = np.mean(out_wav, axis=-1)
        return out_wav.astype(np.float32)


def _apply_speed(wav, speed: float, sr: int):
    wav = np.asarray(wav, dtype=np.float32)
    tail_pad = np.zeros(int(sr * 0.3), dtype=np.float32)
    if speed is None or speed == 1.0:
        return np.concatenate([wav, tail_pad])
    speed = float(speed)
    if speed < 0.1:
        speed = 0.1
    wav = np.concatenate([wav, np.zeros(int(sr * 0.4), dtype=np.float32)])
    stretched = _apply_speed_ffmpeg(wav, sr, speed)
    return np.concatenate([stretched, tail_pad])


def _get_audio_duration_seconds(path_or_b64: str) -> float:
    """
    获取音频时长（秒）。
    path_or_b64: 文件路径，或 base64 字符串（支持 data:audio/...;base64,XXX 或纯 base64）
    """
    if path_or_b64.strip().startswith("data:"):
        raw = path_or_b64.split(",", 1)[1] if "," in path_or_b64 else path_or_b64
        data_bytes = base64.b64decode(raw)
        data, sr = sf.read(io.BytesIO(data_bytes), dtype="float32", always_2d=False)
    elif os.path.isfile(path_or_b64):
        data, sr = sf.read(path_or_b64, dtype="float32", always_2d=False)
    else:
        data_bytes = base64.b64decode(path_or_b64)
        data, sr = sf.read(io.BytesIO(data_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    return len(data) / float(sr)


def _validate_ref_audio_duration(duration: float) -> None:
    """校验参考音频时长在 5-15 秒内，否则抛出 ValueError"""
    if duration < REF_AUDIO_DURATION_MIN:
        raise ValueError(
            f"参考音频时长为 {duration:.1f} 秒，须不少于 {REF_AUDIO_DURATION_MIN:.0f} 秒"
        )
    if duration > REF_AUDIO_DURATION_MAX:
        raise ValueError(
            f"参考音频时长为 {duration:.1f} 秒，须不超过 {REF_AUDIO_DURATION_MAX:.0f} 秒"
        )


def _encode_wav_base64(wav, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _check_whisperx_available() -> bool:
    """检测 WhisperX 是否可用（语音识别 + 字幕生成统一使用）"""
    global _WHISPERX_AVAILABLE
    if _WHISPERX_AVAILABLE is not None:
        return _WHISPERX_AVAILABLE
    try:
        import whisperx  # noqa: F401
        _WHISPERX_AVAILABLE = True
        return True
    except ImportError:
        _WHISPERX_AVAILABLE = False
        return False


def _subtitle_paths_for_audio(audio_path: str) -> Tuple[str, str]:
    """根据音频路径推导字幕文件路径。"""
    return (
        audio_path.replace(".wav", ".srt"),
        audio_path.replace(".wav", "_subtitle.json"),
    )


def _generate_subtitles_for_audio(audio_path: str, language: str, original_text: str = "") -> Optional[Dict]:
    """调用 WhisperX 为生成的音频生成字幕数据。"""
    if not _check_whisperx_available():
        return None
    try:
        orig_path = list(sys.path)
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            from subtitle import generate_subtitle_inline  # pyright: ignore[reportMissingImports]
            return generate_subtitle_inline(audio_path, language, original_text=original_text)
        finally:
            sys.path[:] = orig_path
    except Exception as e:
        print(f"[字幕] 生成失败: {e}")
        return None


def _write_subtitle_files(subtitle_data: Dict, audio_path: str) -> Tuple[Optional[str], Optional[str]]:
    """将字幕数据写入 SRT 和 JSON 文件。"""
    srt_path = None
    json_path = None
    try:
        orig_path = list(sys.path)
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            from subtitle import write_srt, write_subtitle_json  # pyright: ignore[reportMissingImports]
            srt_out, json_out = _subtitle_paths_for_audio(audio_path)
            write_srt(subtitle_data["segments"], srt_out, granularity="sentence")
            write_subtitle_json(subtitle_data, json_out)
            srt_path = srt_out
            json_path = json_out
        finally:
            sys.path[:] = orig_path
    except Exception as e:
        print(f"[字幕] 写入文件失败: {e}")
    return srt_path, json_path


def _build_inline_subtitle_payload(audio_path: str, language: str, original_text: str = "") -> Dict[str, Any]:
    """为同步接口构造内联字幕内容。"""
    payload: Dict[str, Any] = {"subtitle_generated": False}
    if not _check_whisperx_available():
        payload["subtitle_error"] = "WhisperX 未安装，无法为合成结果生成字幕"
        return payload

    try:
        orig_path = list(sys.path)
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            from subtitle import (  # pyright: ignore[reportMissingImports]
                build_srt_content,
                build_subtitle_json_payload,
                generate_subtitle_inline,
            )
            subtitle_data = generate_subtitle_inline(audio_path, language, original_text=original_text)
        finally:
            sys.path[:] = orig_path

        payload["subtitle_generated"] = True
        payload["subtitle_language"] = subtitle_data.get("language", "")
        payload["subtitle_srt"] = build_srt_content(
            subtitle_data.get("segments", []),
            granularity="sentence",
        )
        payload["subtitle_json"] = build_subtitle_json_payload(subtitle_data)
    except Exception as e:
        payload["subtitle_error"] = str(e)

    return payload


def _check_local_whisperx_cache() -> Dict[str, Any]:
    """检查本地 WhisperX 模型缓存状态。"""
    orig_path = list(sys.path)
    sys.path.insert(0, SCRIPTS_DIR)
    try:
        from subtitle import get_whisperx_cache_status  # pyright: ignore[reportMissingImports]
        return get_whisperx_cache_status()
    finally:
        sys.path[:] = orig_path


def _preload_subtitle_models() -> Dict[str, Any]:
    """预加载 WhisperX 字幕模型。"""
    orig_path = list(sys.path)
    sys.path.insert(0, SCRIPTS_DIR)
    try:
        from subtitle import preload_whisperx_assets  # pyright: ignore[reportMissingImports]
        return preload_whisperx_assets()
    finally:
        sys.path[:] = orig_path


def _strip_inline_pinyin(text: str) -> str:
    """Strip inline pinyin annotations like 汉[pin1] / 汉(pin1) / 汉{pin1} to avoid reading pinyin."""
    # Remove inline pinyin markers to prevent model from reading the pinyin letters
    text = re.sub(r'[\(\[\{]\s*[a-zA-Z0-9,，\s]+\s*[\)\]\}]', '', text)
    return text


def _normalize_text_for_tts(text: str) -> str:
    """Collapse newlines and excess whitespace so the model sees a single paragraph."""
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _apply_polyphonic(text: str) -> str:
    """Clean up text before synthesis: normalise whitespace + strip pinyin annotations."""
    if not text:
        return text
    text = _normalize_text_for_tts(text)
    text = _strip_inline_pinyin(text)
    return text


class CustomVoiceRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "Auto"
    speaker: str = Field(..., min_length=1)
    instruct: str | None = None
    speed: float = Field(1.0, ge=0.1)
    speed_enabled: bool = True
    model_size: str = "1.7B"


class VoiceDesignRequest(BaseModel):
    text: str = Field(..., min_length=1)
    instruct: str = Field(..., min_length=1)
    language: str = "Auto"
    speed: float = Field(1.0, ge=0.1)
    speed_enabled: bool = True
    model_size: str = "1.7B"


class VoiceCloneRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "Auto"
    ref_audio_b64: str = Field(..., min_length=32)
    ref_text: str | None = None
    x_vector_only_mode: bool = False
    speed: float = Field(1.0, ge=0.1)
    speed_enabled: bool = True
    model_size: str = "1.7B"


class VoiceCloneAPIRequest(BaseModel):
    """专门的语音克隆API接口（异步任务）"""
    user_id: str = Field(..., min_length=1, description="用户标识")
    task_id: Optional[str] = Field(None, description="任务ID，不填则自动生成")
    text: str = Field(..., min_length=1, description="要合成的文本")
    ref_audio_b64: str = Field(..., min_length=32, description="参考音频的base64编码")
    ref_text: str = Field(..., min_length=1, description="参考音频对应的文本")
    speed: float = Field(1.0, ge=0.1, le=5.0, description="语速，范围0.1-5.0")
    lang: str = Field("English", description="语言，如English, Chinese等")
    model: str = Field("1.7B", description="模型大小：0.6B 或 1.7B")


app = FastAPI(title="Qwen3-TTS API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_preload_models():
    """启动时检查并预加载 WhisperX 字幕模型。"""
    print("\n" + "=" * 60)
    print("正在启动 Qwen3-TTS 全功能服务...")
    print("=" * 60)

    if _check_whisperx_available():
        try:
            cache_status = _check_local_whisperx_cache()
            model_dir = cache_status.get("model_dir", "")
            if cache_status.get("has_cache"):
                print("[1/1] 检测到本地 WhisperX 缓存，开始预加载字幕模型...")
            else:
                print("[1/1] 未检测到本地 WhisperX 模型缓存，开始自动下载并预加载...")
                print(f"      下载目录: {model_dir}")

            preload_info = _preload_subtitle_models()
            align_lang = preload_info.get("align_language")
            align_note = f", 对齐语言: {align_lang}" if align_lang else ""
            print(
                f"[OK] WhisperX 字幕模型加载成功 "
                f"(模型: {preload_info.get('model_name')}, 目录: {model_dir}{align_note})"
            )
        except Exception as e:
            print(f"[FAIL] WhisperX 字幕模型预加载失败: {e}")
    else:
        print("[1/1] WhisperX 未安装，跳过字幕模型预加载")
        print("      安装后模型缓存目录: models/WhisperX")


@app.get("/")
def index():
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/custom-voice")
def custom_voice(req: CustomVoiceRequest):
    try:
        model = _get_model("custom_voice", req.model_size)
        text = _apply_polyphonic(req.text)
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=req.language,
            speaker=req.speaker,
            instruct=req.instruct,
        )
        wav_out = _apply_speed(
            wavs[0],
            req.speed if req.speed_enabled else 1.0,
            sr,
        )
        audio_b64 = _encode_wav_base64(wav_out, sr)
        response: Dict[str, Any] = {"sample_rate": sr, "audio_b64": audio_b64}

        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                temp_audio_path = tmp_audio.name
            sf.write(temp_audio_path, wav_out, sr, format="WAV")
            response.update(_build_inline_subtitle_payload(temp_audio_path, req.language, original_text=req.text))
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except OSError:
                    pass

        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/voice-design")
def voice_design(req: VoiceDesignRequest):
    try:
        model = _get_model("voice_design", req.model_size)
        text = _apply_polyphonic(req.text)
        wavs, sr = model.generate_voice_design(
            text=text,
            language=req.language,
            instruct=req.instruct,
        )
        wav_out = _apply_speed(
            wavs[0],
            req.speed if req.speed_enabled else 1.0,
            sr,
        )
        audio_b64 = _encode_wav_base64(wav_out, sr)
        response: Dict[str, Any] = {"sample_rate": sr, "audio_b64": audio_b64}

        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                temp_audio_path = tmp_audio.name
            sf.write(temp_audio_path, wav_out, sr, format="WAV")
            response.update(_build_inline_subtitle_payload(temp_audio_path, req.language, original_text=req.text))
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except OSError:
                    pass

        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/voice-clone")
def voice_clone(req: VoiceCloneRequest):
    try:
        # 验证参考音频时长（5-15 秒）
        try:
            duration = _get_audio_duration_seconds(req.ref_audio_b64)
            _validate_ref_audio_duration(duration)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        model = _get_model("base", req.model_size)
        text = _apply_polyphonic(req.text)
        ref_text = _apply_polyphonic(req.ref_text)
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=req.language,
            ref_audio=req.ref_audio_b64,
            ref_text=ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
        )
        wav_out = _apply_speed(
            wavs[0],
            req.speed if req.speed_enabled else 1.0,
            sr,
        )
        audio_b64 = _encode_wav_base64(wav_out, sr)
        response: Dict[str, Any] = {"sample_rate": sr, "audio_b64": audio_b64}

        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                temp_audio_path = tmp_audio.name
            sf.write(temp_audio_path, wav_out, sr, format="WAV")
            response.update(_build_inline_subtitle_payload(temp_audio_path, req.language, original_text=req.text))
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except OSError:
                    pass

        return response
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _process_voice_clone_task(user_id: str, task_id: str, params: Dict):
    """后台处理语音克隆任务"""
    try:
        # 更新任务状态为处理中
        task_data = _get_task(user_id, task_id)
        if not task_data:
            return
        
        task_data["status"] = "processing"
        task_data["started_at"] = datetime.now().isoformat()
        _save_task(user_id, task_id, task_data)
        
        # 加载模型
        model = _get_model("base", params["model"])
        
        # 处理文本（移除拼音标注）
        text = _apply_polyphonic(params["text"])
        ref_text = _apply_polyphonic(params["ref_text"])
        
        # 生成语音
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=params["lang"],
            ref_audio=params["ref_audio_b64"],
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        
        # 应用语速调整
        wav_out = _apply_speed(wavs[0], params["speed"], sr)
        
        # 保存音频文件
        audio_filename = f"{user_id}_{task_id}.wav"
        audio_path = os.path.join(AUDIO_OUTPUT_DIR, audio_filename)
        sf.write(audio_path, wav_out, sr, format="WAV")

        # 生成字幕（可选，失败不影响主任务）
        subtitle_srt_url = None
        subtitle_json_url = None
        subtitle_error = None
        try:
            if not _check_whisperx_available():
                subtitle_error = "WhisperX 未安装，无法为合成结果生成字幕"
            subtitle_data = _generate_subtitles_for_audio(audio_path, params["lang"], original_text=params.get("text", ""))
            if subtitle_data:
                srt_path, json_path = _write_subtitle_files(subtitle_data, audio_path)
                if srt_path:
                    subtitle_srt_url = f"/api/download/{user_id}/{task_id}?type=srt"
                if json_path:
                    subtitle_json_url = f"/api/download/{user_id}/{task_id}?type=json"
        except Exception as e:
            subtitle_error = str(e)
            print(f"[字幕] 生成跳过: {e}")
        
        # 更新任务状态为完成
        task_data["status"] = "completed"
        task_data["completed_at"] = datetime.now().isoformat()
        task_data["audio_url"] = f"/api/download/{user_id}/{task_id}"
        task_data["audio_file"] = audio_filename
        task_data["sample_rate"] = int(sr)
        if subtitle_srt_url:
            task_data["subtitle_srt"] = subtitle_srt_url
        if subtitle_json_url:
            task_data["subtitle_json"] = subtitle_json_url
        if subtitle_error:
            task_data["subtitle_error"] = subtitle_error
        _save_task(user_id, task_id, task_data)
        
        print(f"[Task {task_id}] 任务完成，音频已保存到: {audio_path}")
        
    except Exception as e:
        # 更新任务状态为失败
        task_data = _get_task(user_id, task_id)
        if task_data:
            task_data["status"] = "failed"
            task_data["error"] = str(e)
            task_data["failed_at"] = datetime.now().isoformat()
            _save_task(user_id, task_id, task_data)
        print(f"[Task {task_id}] 任务失败: {str(e)}")


@app.post("/api/clone")
def submit_voice_clone_task(req: VoiceCloneAPIRequest):
    """
    提交语音克隆任务（异步）
    
    参数说明：
    - user_id: 用户标识（必填）
    - task_id: 任务ID（可选，不填则自动生成）
    - text: 要合成的文本内容
    - ref_audio_b64: 参考音频文件的base64编码
    - ref_text: 参考音频对应的文本
    - speed: 语速（默认1.0，范围0.1-5.0）
    - lang: 语言（默认English）
    - model: 模型大小（默认1.7B，可选0.6B）
    
    返回：
    - task_id: 任务ID
    - user_id: 用户ID
    - status: 任务状态（pending/processing/completed/failed）
    """
    try:
        # 验证模型参数
        if req.model not in ("0.6B", "1.7B"):
            raise ValueError(f"不支持的模型: {req.model}，请使用 0.6B 或 1.7B")
        
        # 验证参考音频时长（5-15 秒）
        try:
            duration = _get_audio_duration_seconds(req.ref_audio_b64)
            _validate_ref_audio_duration(duration)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # 生成或使用提供的任务ID
        task_id = req.task_id if req.task_id else _generate_task_id()
        user_id = req.user_id
        
        # 检查任务是否已存在
        existing_task = _get_task(user_id, task_id)
        if existing_task:
            raise HTTPException(status_code=400, detail=f"任务ID {task_id} 已存在")
        
        # 创建任务记录
        task_data = {
            "task_id": task_id,
            "user_id": user_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "params": {
                "text": req.text,
                "ref_text": req.ref_text,
                "ref_audio_b64": req.ref_audio_b64,
                "speed": req.speed,
                "lang": req.lang,
                "model": req.model,
            }
        }
        _save_task(user_id, task_id, task_data)
        
        # 确保队列工作线程已启动
        _start_queue_worker()
        
        # 将任务加入队列
        task_item = {
            "user_id": user_id,
            "task_id": task_id,
            "params": task_data["params"]
        }
        _TASK_QUEUE.put(task_item)
        
        # 获取队列位置
        queue_position = _TASK_QUEUE.qsize()
        
        return {
            "task_id": task_id,
            "user_id": user_id,
            "status": "pending",
            "queue_position": queue_position,
            "message": f"任务已加入队列，当前排队位置: {queue_position}"
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _base_url(request: Request) -> str:
    """获取当前请求的根 URL（http://ip:端口）"""
    return str(request.base_url).rstrip("/")


@app.get("/api/tasks/{user_id}")
def get_user_tasks(user_id: str, request: Request):
    """
    查询用户的所有任务
    
    参数：
    - user_id: 用户标识
    
    返回：
    - tasks: 任务列表，audio_url 为完整 HTTP 地址
    """
    try:
        base = _base_url(request)
        tasks = _get_user_tasks(user_id)
        out_tasks = []
        for task in tasks:
            t = copy.deepcopy(task)
            if "params" in t and "ref_audio_b64" in t["params"]:
                t["params"]["ref_audio_b64"] = "[已隐藏]"
            if t.get("audio_url") and not t["audio_url"].startswith("http"):
                t["audio_url"] = base + t["audio_url"]
            if t.get("subtitle_srt") and not t["subtitle_srt"].startswith("http"):
                t["subtitle_srt"] = base + t["subtitle_srt"]
            if t.get("subtitle_json") and not t["subtitle_json"].startswith("http"):
                t["subtitle_json"] = base + t["subtitle_json"]
            out_tasks.append(t)
        
        return {
            "user_id": user_id,
            "total": len(out_tasks),
            "tasks": out_tasks
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/task/{user_id}/{task_id}")
def get_task_status(user_id: str, task_id: str, request: Request):
    """
    查询指定任务的状态
    
    返回的 audio_url 为完整 HTTP 地址（http://ip:端口/api/download/...）
    """
    try:
        task = _get_task(user_id, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        out = copy.deepcopy(task)
        if "params" in out and "ref_audio_b64" in out["params"]:
            out["params"]["ref_audio_b64"] = "[已隐藏]"
        base_url = _base_url(request)
        if out.get("audio_url") and not out["audio_url"].startswith("http"):
            out["audio_url"] = base_url + out["audio_url"]
        if out.get("subtitle_srt") and not out["subtitle_srt"].startswith("http"):
            out["subtitle_srt"] = base_url + out["subtitle_srt"]
        if out.get("subtitle_json") and not out["subtitle_json"].startswith("http"):
            out["subtitle_json"] = base_url + out["subtitle_json"]
        
        return out
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/task/{user_id}/{task_id}")
def delete_task(user_id: str, task_id: str):
    """
    删除任务（含音频文件）。
    仅支持删除已完成、失败或等待中的任务；处理中的任务不可删除。
    """
    try:
        task = _get_task(user_id, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task.get("status") == "processing":
            raise HTTPException(status_code=400, detail="任务处理中，无法删除")
        _delete_task(user_id, task_id)
        return {"ok": True, "message": "任务已删除"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/task/{user_id}/{task_id}/retry")
def retry_task(user_id: str, task_id: str):
    """
    使用原任务参数重新提交（生成新任务ID，加入队列）。
    返回新任务的 task_id 与 status。
    """
    try:
        task = _get_task(user_id, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        params = task.get("params")
        if not params or "ref_audio_b64" not in params or params.get("ref_audio_b64") == "[已隐藏]":
            raise HTTPException(status_code=400, detail="无法重试：缺少参考音频数据")
        new_task_id = _generate_task_id()
        new_task_data = {
            "task_id": new_task_id,
            "user_id": user_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "params": params,
        }
        _save_task(user_id, new_task_id, new_task_data)
        _TASK_QUEUE.put({"user_id": user_id, "task_id": new_task_id, "params": params})
        return {
            "ok": True,
            "task_id": new_task_id,
            "user_id": user_id,
            "status": "pending",
            "message": "已重新加入队列",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/download/{user_id}/{task_id}")
def download_audio(user_id: str, task_id: str, type: str = Query("audio", regex="^(audio|srt|json)$")):
    """
    下载任务生成的音频文件
    
    参数：
    - user_id: 用户标识
    - task_id: 任务ID
    
    返回：音频文件或字幕文件
    """
    try:
        task = _get_task(user_id, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        
        if task.get("status") != "completed":
            raise HTTPException(status_code=400, detail=f"任务未完成，当前状态: {task.get('status')}")
        
        audio_filename = task.get("audio_file")
        if not audio_filename:
            raise HTTPException(status_code=404, detail="音频文件不存在")
        
        audio_path = os.path.join(AUDIO_OUTPUT_DIR, audio_filename)
        srt_path, json_path = _subtitle_paths_for_audio(audio_path)

        if type == "srt":
            if not os.path.exists(srt_path):
                raise HTTPException(status_code=404, detail="SRT 字幕文件不存在（WhisperX 可能未安装或生成失败）")
            return FileResponse(
                path=srt_path,
                media_type="text/plain; charset=utf-8",
                filename=f"{task_id}.srt"
            )

        if type == "json":
            if not os.path.exists(json_path):
                raise HTTPException(status_code=404, detail="JSON 字幕文件不存在（WhisperX 可能未安装或生成失败）")
            return FileResponse(
                path=json_path,
                media_type="application/json; charset=utf-8",
                filename=f"{task_id}_subtitle.json"
            )

        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail="音频文件未找到")
        
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename=f"clone_{task_id}.wav"
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/queue/status")
def get_queue_status():
    """
    获取队列状态
    
    返回：
    - queue_size: 队列中等待的任务数
    - worker_running: 工作线程是否运行中
    """
    try:
        return {
            "queue_size": _TASK_QUEUE.qsize(),
            "worker_running": _QUEUE_WORKER_STARTED
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
