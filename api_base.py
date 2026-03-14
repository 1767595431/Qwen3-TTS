"""
语音克隆专用服务（仅提供 Base 模型异步任务接口）
支持 WhisperX 语音识别与字幕生成
启动: python api_base.py
"""
import base64
import copy
import io
import json
import os
import re
import sys
import time
import threading
import warnings
from datetime import datetime, timezone
from typing import Dict, Tuple, List, Optional, Any
from queue import Queue

import requests as http_requests

warnings.filterwarnings("ignore", message="torchcodec is not installed")
warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing` to a config")

import logging
logging.getLogger("whisperx").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.migration.utils").setLevel(logging.ERROR)

import numpy as np
import soundfile as sf
import subprocess
import tempfile
import torch
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from qwen_tts import Qwen3TTSModel

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
WEB_DIR = os.path.join(APP_DIR, "web")
INDEX_FILE = os.path.join(WEB_DIR, "index_base.html")
TASKS_DIR = os.path.join(APP_DIR, "tasks")
AUDIO_OUTPUT_DIR = os.path.join(TASKS_DIR, "audio")
SCRIPTS_DIR = os.path.join(APP_DIR, "scripts")
GTCRN_MODEL_PATH = os.path.join(SCRIPTS_DIR, "model_trained_on_dns3.tar")

# 参考音频时长限制（秒）
REF_AUDIO_DURATION_MIN = 3.0  # 官方支持3秒快速克隆
REF_AUDIO_DURATION_MAX = 15.0

# 确保任务目录存在
os.makedirs(TASKS_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


def _resolve_model_path(model_size: str) -> str:
    """仅支持语音克隆 Base 模型"""
    if model_size not in ("0.6B", "1.7B"):
        raise ValueError(f"Unsupported model_size: {model_size}")
    model_name = f"Qwen3-TTS-12Hz-{model_size}-Base"
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def _default_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    return "cpu", torch.float32


_MODEL_CACHE: Dict[str, Qwen3TTSModel] = {}
_ENHANCER_AVAILABLE: Optional[bool] = None  # GTCRN 降噪是否可用
_WHISPERX_AVAILABLE: Optional[bool] = None  # WhisperX 是否可用（语音识别 + 字幕生成）
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
            f.flush()  # 确保立即写入磁盘
            os.fsync(f.fileno())  # 强制同步到磁盘


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
    while True:
        try:
            # 从队列获取任务（阻塞等待）
            task_item = _TASK_QUEUE.get()
            
            if task_item is None:  # None 作为停止信号
                break
            
            user_id = task_item["user_id"]
            task_id = task_item["task_id"]
            params = task_item["params"]
            
            print(f"[队列] 开始处理任务 {task_id} (剩余: {_TASK_QUEUE.qsize()})")
            
            # 处理任务
            _process_voice_clone_task(user_id, task_id, params)
            
            # 标记任务完成
            _TASK_QUEUE.task_done()
            
        except Exception as e:
            print(f"[队列] 处理任务出错: {str(e)}")
            import traceback
            traceback.print_exc()


def _start_queue_worker():
    """启动队列工作线程"""
    global _QUEUE_WORKER_STARTED
    if not _QUEUE_WORKER_STARTED:
        worker_thread = threading.Thread(target=_queue_worker, daemon=True)
        worker_thread.start()
        _QUEUE_WORKER_STARTED = True


def _get_model(model_size: str) -> Qwen3TTSModel:
    """仅加载语音克隆 Base 模型"""
    if model_size not in _MODEL_CACHE:
        model_path = _resolve_model_path(model_size)
        device, dtype = _default_device_and_dtype()
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation="eager",
        )
        _MODEL_CACHE[model_size] = model
    return _MODEL_CACHE[model_size]


def _check_enhancer_available() -> bool:
    """检测 GTCRN 降噪是否可用"""
    global _ENHANCER_AVAILABLE
    if _ENHANCER_AVAILABLE is not None:
        return _ENHANCER_AVAILABLE
    try:
        if not os.path.isfile(GTCRN_MODEL_PATH):
            _ENHANCER_AVAILABLE = False
            return False
        orig_path = list(sys.path)
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            from gtcrn import enhance_audio  # noqa: F401
            _ENHANCER_AVAILABLE = True
            return True
        finally:
            sys.path[:] = orig_path
    except Exception:
        _ENHANCER_AVAILABLE = False
        return False


def _enhance_audio_for_asr(input_path: str) -> str:
    """
    使用 GTCRN 对参考音频降噪，输出 16kHz wav。
    返回增强后的临时文件路径，调用方需负责删除。
    """
    orig_path = list(sys.path)
    sys.path.insert(0, SCRIPTS_DIR)
    try:
        from gtcrn import enhance_audio
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            enhance_audio(input_path, out_path, GTCRN_MODEL_PATH)
            return out_path
        except Exception:
            try:
                os.remove(out_path)
            except OSError:
                pass
            raise
    finally:
        sys.path[:] = orig_path


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


def _recognize_audio(audio_path: str, language: str = "Auto") -> str:
    """使用 WhisperX 识别音频文本，返回纯文本"""
    orig_path = list(sys.path)
    sys.path.insert(0, SCRIPTS_DIR)
    try:
        from subtitle import recognize_text  # pyright: ignore[reportMissingImports]
        return recognize_text(audio_path, language)
    finally:
        sys.path[:] = orig_path


def _generate_subtitles_for_audio(audio_path: str, language: str, original_text: str = "") -> Optional[Dict]:
    """
    调用 WhisperX 为生成的音频生成字幕数据。
    如果提供了 original_text，字幕文本使用原文（WhisperX 仅提供时间戳）。
    失败时返回 None（不影响 TTS 主流程）。
    """
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
    """
    将字幕数据写入 SRT 和 JSON 文件。
    返回 (srt_path, json_path)，失败则对应项为 None。
    """
    srt_path = None
    json_path = None
    try:
        orig_path = list(sys.path)
        sys.path.insert(0, SCRIPTS_DIR)
        try:
            from subtitle import write_srt, write_subtitle_json  # pyright: ignore[reportMissingImports]
            srt_out = audio_path.replace(".wav", ".srt")
            write_srt(subtitle_data["segments"], srt_out, granularity="sentence")
            srt_path = srt_out

            json_out = audio_path.replace(".wav", "_subtitle.json")
            write_subtitle_json(subtitle_data, json_out)
            json_path = json_out
        finally:
            sys.path[:] = orig_path
    except Exception as e:
        print(f"[字幕] 写入文件失败: {e}")
    return srt_path, json_path


def _subtitle_paths_for_audio(audio_path: str) -> Tuple[str, str]:
    """根据音频路径推导字幕文件路径。"""
    return (
        audio_path.replace(".wav", ".srt"),
        audio_path.replace(".wav", "_subtitle.json"),
    )


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
    path_or_b64: 文件路径 或 base64 字符串（支持 data:audio/...;base64,XXX 或纯 base64）
    """
    path_or_b64 = path_or_b64.strip()
    
    # 1. 处理 data:audio 格式
    if path_or_b64.startswith("data:"):
        raw = path_or_b64.split(",", 1)[1] if "," in path_or_b64 else path_or_b64
        data_bytes = base64.b64decode(raw)
        data, sr = sf.read(io.BytesIO(data_bytes), dtype="float32", always_2d=False)
    
    # 2. 检查是否为文件路径（必须实际存在）
    elif os.path.isfile(path_or_b64):
        data, sr = sf.read(path_or_b64, dtype="float32", always_2d=False)
    
    # 3. 作为纯base64解码
    else:
        data_bytes = base64.b64decode(path_or_b64)
        data, sr = sf.read(io.BytesIO(data_bytes), dtype="float32", always_2d=False)
    
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    duration = len(data) / sr
    return float(duration)


def _validate_ref_audio_duration(duration: float) -> None:
    """校验参考音频时长在 3-15 秒内，否则抛出 ValueError"""
    if duration < REF_AUDIO_DURATION_MIN:
        raise ValueError(f"参考音频时长过短（{duration:.1f}秒），需要 {REF_AUDIO_DURATION_MIN}-{REF_AUDIO_DURATION_MAX} 秒")
    if duration > REF_AUDIO_DURATION_MAX:
        raise ValueError(f"参考音频时长过长（{duration:.1f}秒），需要 {REF_AUDIO_DURATION_MIN}-{REF_AUDIO_DURATION_MAX} 秒")


def _encode_wav_base64(wav, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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


class VoiceCloneRequest(BaseModel):
    """同步语音克隆接口（立即返回音频）"""
    text: str = Field(..., min_length=1, description="要合成的文本")
    ref_audio_b64: str = Field(..., min_length=32, description="参考音频的base64编码")
    ref_text: str = Field(..., min_length=1, description="参考音频对应的文本")
    speed: float = Field(1.0, ge=0.1, le=5.0, description="语速，范围0.1-5.0")
    speed_enabled: bool = Field(True, description="是否启用语速调整")
    language: str = Field("English", description="语言，如English, Chinese等")
    model_size: str = Field("1.7B", description="模型大小：0.6B 或 1.7B")
    x_vector_only_mode: bool = Field(False, description="是否只使用 x-vector 模式")


from xfyun_service import translate_router, XFYUN_TAGS

tags_metadata = [
    {"name": "系统", "description": "健康检查、页面等基础接口"},
    {"name": "语音识别", "description": "WhisperX 语音识别（ASR）"},
    {"name": "语音克隆", "description": "Qwen3-TTS 语音合成与克隆"},
    {"name": "任务管理", "description": "异步语音克隆任务的查询、下载、删除"},
] + XFYUN_TAGS

app = FastAPI(
    title="Qwen3-TTS 语音克隆 API",
    description="语音合成、语音克隆、语音识别、讯飞机器翻译",
    openapi_tags=tags_metadata,
)
app.include_router(translate_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_preload_models():
    """启动时预加载所有模型"""
    print("\n" + "="*60)
    print("正在启动 Qwen3-TTS 语音克隆服务...")
    print("="*60)
    
    # 1. 预加载 WhisperX 模型（语音识别 + 字幕生成统一使用）
    if _check_whisperx_available():
        try:
            cache_status = _check_local_whisperx_cache()
            model_dir = cache_status.get("model_dir", "")
            if cache_status.get("has_cache"):
                print("[1/2] 检测到本地 WhisperX 缓存，开始预加载模型...")
            else:
                print("[1/2] 未检测到本地 WhisperX 模型缓存，开始自动下载并预加载...")
                print(f"      下载目录: {model_dir}")

            preload_info = _preload_subtitle_models()
            align_lang = preload_info.get("align_language")
            align_note = f", 对齐语言: {align_lang}" if align_lang else ""
            print(
                f"[OK] WhisperX 模型加载成功 "
                f"(模型: {preload_info.get('model_name')}, 目录: {model_dir}{align_note})"
            )
        except Exception as e:
            print(f"[FAIL] WhisperX 模型预加载失败: {e}")
    else:
        print("[1/2] WhisperX 未安装，语音识别和字幕生成均不可用")
        print("      安装命令: pip install whisperx")

    # 2. 预加载 GTCRN 降噪模型
    if _check_enhancer_available():
        try:
            print("[2/2] 加载 GTCRN 降噪模型...")
            orig_path = list(sys.path)
            sys.path.insert(0, SCRIPTS_DIR)
            try:
                from gtcrn import load_gtcrn_model
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                model = load_gtcrn_model(GTCRN_MODEL_PATH, device)
                print(f"[OK] GTCRN 降噪模型加载成功 (设备: {device})")
            finally:
                sys.path[:] = orig_path
        except Exception as e:
            print(f"[FAIL] GTCRN 降噪模型加载失败: {e}")
    else:
        print("[2/2] GTCRN 降噪模型不可用，跳过预加载")

    # 3. 启动任务队列工作线程
    _start_queue_worker()
    
    # 注意：实际端口在 main 启动时才确定，这里先不打印


@app.get("/", tags=["系统"], summary="主页")
def index():
    return FileResponse(INDEX_FILE)


@app.get("/xfyun", tags=["系统"], summary="讯飞 AI 服务页面")
def xfyun_page():
    return FileResponse(os.path.join(WEB_DIR, "xfyun.html"))


@app.get("/health", tags=["系统"], summary="健康检查")
def health():
    return {"ok": True}


@app.get("/api/asr/status", tags=["语音识别"], summary="语音识别状态")
def asr_status():
    """
    检查 WhisperX 语音识别和 GTCRN 降噪是否可用
    返回: { "available": true/false, "enhancer": true/false, "duration_min": 3, "duration_max": 15 }
    """
    return {
        "available": _check_whisperx_available(),
        "enhancer": _check_enhancer_available(),
        "duration_min": int(REF_AUDIO_DURATION_MIN),
        "duration_max": int(REF_AUDIO_DURATION_MAX),
    }


@app.post("/api/asr", tags=["语音识别"], summary="语音识别（WhisperX）")
async def recognize_audio(file: UploadFile = File(...)):
    """
    使用 WhisperX 识别参考音频文本
    流程：上传音频 → GTCRN降噪增强(可选) → WhisperX 识别
    """
    if not _check_whisperx_available():
        raise HTTPException(
            status_code=503,
            detail="WhisperX 未就绪。请安装: pip install whisperx"
        )
    if not file.filename or not file.content_type:
        raise HTTPException(status_code=400, detail="请上传有效的音频文件")
    allowed = {"audio/", "application/octet-stream"}
    if not any(file.content_type.startswith(t) for t in allowed):
        pass
    
    tmp_path = None
    enhanced_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[-1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            duration = _get_audio_duration_seconds(tmp_path)
            _validate_ref_audio_duration(duration)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        if _check_enhancer_available():
            try:
                print(f"[识别] 降噪增强 → WhisperX 识别 ({duration:.1f}s)")
                enhanced_path = _enhance_audio_for_asr(tmp_path)
                asr_input = enhanced_path
            except Exception as e:
                print(f"[识别] 降噪失败，直接识别: {e}")
                asr_input = tmp_path
        else:
            print(f"[识别] WhisperX 直接识别 ({duration:.1f}s)")
            asr_input = tmp_path
        
        text = _recognize_audio(asr_input)
        print(f"[OK] 识别完成: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        return {"text": text.strip() if text else ""}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if enhanced_path and enhanced_path != tmp_path:
            try:
                os.remove(enhanced_path)
            except OSError:
                pass


def _process_voice_clone_task(user_id: str, task_id: str, params: Dict):
    """后台处理语音克隆任务"""
    ref_audio_tmp = None
    try:
        # 更新任务状态为处理中
        task_data = _get_task(user_id, task_id)
        if not task_data:
            return
        
        task_data["status"] = "processing"
        task_data["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_task(user_id, task_id, task_data)
        
        # 加载模型
        model = _get_model(params["model"])
        
        # 处理文本（移除拼音标注）
        text = _apply_polyphonic(params["text"])
        ref_text = _apply_polyphonic(params["ref_text"])
        
        # 将base64转为临时文件
        ref_audio_b64 = params["ref_audio_b64"]
        if ref_audio_b64.startswith("data:"):
            ref_audio_b64 = ref_audio_b64.split(",", 1)[1]
        
        audio_bytes = base64.b64decode(ref_audio_b64)
        ref_audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        ref_audio_tmp.write(audio_bytes)
        ref_audio_tmp.close()
        
        # 生成语音
        print(f"[任务 {task_id}] 生成语音中...")
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=params["lang"],
            ref_audio=ref_audio_tmp.name,
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
                    print(f"[字幕] SRT 已生成: {os.path.basename(srt_path)}")
                if json_path:
                    subtitle_json_url = f"/api/download/{user_id}/{task_id}?type=json"
                    print(f"[字幕] JSON 已生成: {os.path.basename(json_path)}")
        except Exception as e:
            subtitle_error = str(e)
            print(f"[字幕] 生成跳过: {e}")
        
        # 更新任务状态为完成
        task_data["status"] = "completed"
        task_data["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        
        print(f"[OK] 任务 {task_id} 完成 (时长: {len(wav_out)/sr:.1f}s)")
        
    except Exception as e:
        # 更新任务状态为失败
        task_data = _get_task(user_id, task_id)
        if task_data:
            task_data["status"] = "failed"
            task_data["error"] = str(e)
            task_data["failed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _save_task(user_id, task_id, task_data)
        print(f"[FAIL] 任务 {task_id} 失败: {str(e)}")
    
    finally:
        # 清理临时文件
        if ref_audio_tmp and os.path.exists(ref_audio_tmp.name):
            try:
                os.unlink(ref_audio_tmp.name)
            except:
                pass


@app.post("/api/voice-clone", tags=["语音克隆"], summary="语音克隆（同步）")
def voice_clone_sync(req: VoiceCloneRequest):
    """
    同步语音克隆接口（立即返回音频 base64）
    
    生成完成后立即返回音频数据，无需轮询任务状态
    """
    try:
        # 验证参考音频时长（3-15 秒）
        try:
            duration = _get_audio_duration_seconds(req.ref_audio_b64)
            _validate_ref_audio_duration(duration)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # 加载模型
        model = _get_model(req.model_size)
        
        # 处理文本
        text = _apply_polyphonic(req.text)
        ref_text = _apply_polyphonic(req.ref_text)
        
        # 生成语音
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=req.language,
            ref_audio=req.ref_audio_b64,
            ref_text=ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
        )
        
        # 应用语速调整
        wav_out = _apply_speed(
            wavs[0],
            req.speed if req.speed_enabled else 1.0,
            sr,
        )
        
        # 编码为 base64 返回
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


@app.post("/api/clone/upload", tags=["语音克隆"], summary="提交语音克隆任务（异步上传）")
async def submit_voice_clone_task_upload(
    user_id: str = Form(...),
    text: str = Form(...),
    ref_text: str = Form(...),
    ref_audio: UploadFile = File(...),
    task_id: Optional[str] = Form(None),
    speed: float = Form(1.0),
    lang: str = Form("English"),
    model: str = Form("1.7B"),
):
    """
    提交语音克隆任务（异步，文件上传方式）
    
    参数说明：
    - user_id: 用户标识（必填）
    - text: 要合成的文本内容（必填）
    - ref_text: 参考音频对应的文本（必填）
    - ref_audio: 参考音频文件（必填，3-15秒）
    - task_id: 任务ID（可选，不填则自动生成）
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
        if model not in ("0.6B", "1.7B"):
            raise ValueError(f"不支持的模型: {model}，请使用 0.6B 或 1.7B")
        
        # 读取上传的音频文件
        audio_content = await ref_audio.read()
        
        # 转换为base64
        ref_audio_b64 = base64.b64encode(audio_content).decode('utf-8')
        
        # 验证参考音频时长（3-15 秒）
        try:
            duration = _get_audio_duration_seconds(ref_audio_b64)
            _validate_ref_audio_duration(duration)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # 生成任务ID
        task_id_final = task_id or _generate_task_id()
        
        # 检查任务是否已存在
        existing_task = _get_task(user_id, task_id_final)
        if existing_task:
            raise HTTPException(
                status_code=400, 
                detail=f"任务ID已存在: {task_id_final}"
            )
        
        # 创建任务记录
        task_data = {
            "task_id": task_id_final,
            "user_id": user_id,
            "status": "pending",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": {
                "text": text,
                "ref_audio_b64": ref_audio_b64,
                "ref_text": ref_text,
                "speed": float(speed),
                "lang": lang,
                "model": model,
            }
        }
        
        # 保存任务
        _save_task(user_id, task_id_final, task_data)
        
        # 提交到队列
        _TASK_QUEUE.put({
            "user_id": user_id,
            "task_id": task_id_final,
            "params": task_data["params"]
        })
        
        # 启动队列工作线程（如果尚未启动）
        _start_queue_worker()
        
        return {
            "task_id": task_id_final,
            "user_id": user_id,
            "status": task_data["status"],
            "created_at": task_data["created_at"],
            "queue_position": _TASK_QUEUE.qsize(),
            "message": "任务已提交，请轮询 /api/task/{user_id}/{task_id} 查询状态"
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _base_url(request: Request) -> str:
    """从 request 中获取服务的 base URL"""
    scheme = request.url.scheme
    netloc = request.headers.get("host", request.client.host)
    return f"{scheme}://{netloc}"


@app.get("/api/task/{user_id}/{task_id}", tags=["任务管理"], summary="查询任务状态")
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
        
        # 添加禁用缓存的响应头，确保每次获取最新状态
        return JSONResponse(
            content=out,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/tasks/{user_id}", tags=["任务管理"], summary="查询用户所有任务")
def list_user_tasks(user_id: str, request: Request):
    """
    查询用户的所有任务
    
    返回任务列表，按创建时间倒序
    """
    try:
        tasks = _get_user_tasks(user_id)
        base_url = _base_url(request)
        
        # 处理每个任务的 audio_url
        for task in tasks:
            if "params" in task and "ref_audio_b64" in task["params"]:
                task["params"]["ref_audio_b64"] = "[已隐藏]"
            if task.get("audio_url") and not task["audio_url"].startswith("http"):
                task["audio_url"] = base_url + task["audio_url"]
            if task.get("subtitle_srt") and not task["subtitle_srt"].startswith("http"):
                task["subtitle_srt"] = base_url + task["subtitle_srt"]
            if task.get("subtitle_json") and not task["subtitle_json"].startswith("http"):
                task["subtitle_json"] = base_url + task["subtitle_json"]
        
        return {
            "user_id": user_id,
            "total": len(tasks),
            "tasks": tasks
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/task/{user_id}/{task_id}", tags=["任务管理"], summary="删除任务")
def delete_task(user_id: str, task_id: str):
    """删除指定任务（包括音频文件）"""
    try:
        success = _delete_task(user_id, task_id)
        if not success:
            raise HTTPException(status_code=404, detail="任务不存在")
        return {"message": "任务已删除", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/download/{user_id}/{task_id}", tags=["任务管理"], summary="下载任务文件")
def download_file(user_id: str, task_id: str, type: str = Query("audio", regex="^(audio|srt|json)$")):
    """
    下载任务生成的文件

    参数:
    - type: 文件类型，audio（默认）/ srt / json
    """
    task = _get_task(user_id, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"任务尚未完成，当前状态: {task['status']}"
        )
    
    audio_file = task.get("audio_file")
    if not audio_file:
        raise HTTPException(status_code=404, detail="音频文件不存在")
    
    audio_path = os.path.join(AUDIO_OUTPUT_DIR, audio_file)
    srt_path, json_path = _subtitle_paths_for_audio(audio_path)

    if type == "srt":
        if not os.path.exists(srt_path):
            raise HTTPException(status_code=404, detail="SRT 字幕文件不存在（WhisperX 可能未安装或生成失败）")
        return FileResponse(
            srt_path,
            media_type="text/plain; charset=utf-8",
            filename=f"{task_id}.srt",
        )
    
    if type == "json":
        if not os.path.exists(json_path):
            raise HTTPException(status_code=404, detail="JSON 字幕文件不存在（WhisperX 可能未安装或生成失败）")
        return FileResponse(
            json_path,
            media_type="application/json; charset=utf-8",
            filename=f"{task_id}_subtitle.json",
        )
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="音频文件已被删除")
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"{task_id}.wav",
    )


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Qwen3-TTS 语音克隆服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址（默认: 0.0.0.0）")
    parser.add_argument("--port", type=int, default=9778, help="监听端口（默认: 9770）")
    args = parser.parse_args()
    
    HOST = args.host
    PORT = args.port
    
    # 启动前打印服务信息
    print("="*60)
    print("[OK] 服务启动完成！")
    print(f"[INFO] WhisperX 模型目录: models/WhisperX")
    print(f"[INFO] 监听地址: http://{HOST}:{PORT}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host=HOST, port=PORT)
