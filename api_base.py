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
import shutil
import sys
import time
import threading
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, List, Optional, Any
from queue import Queue

import requests as http_requests

# ----------------------------
# GPU 选择（必须在 import torch 前生效）
# 用法：
#   - 环境变量：QWEN_TTS_GPU=1  或 CUDA_VISIBLE_DEVICES=1
#   - 命令行：python api_base.py --gpu 1 --port 9770
# 说明：
#   - 设置 CUDA_VISIBLE_DEVICES 后，程序内看到的第一张卡会变成 cuda:0
#   - 例如 --gpu 1 表示只暴露物理第 2 张卡给本进程
# ----------------------------
def _preparse_gpu_from_argv(argv: List[str]) -> Optional[str]:
    try:
        i = argv.index("--gpu")
    except ValueError:
        return None
    if i + 1 >= len(argv):
        return ""
    return str(argv[i + 1]).strip()


def _apply_cuda_visible_devices() -> None:
    # 若用户已显式设置 CUDA_VISIBLE_DEVICES，则不覆盖
    if os.getenv("CUDA_VISIBLE_DEVICES", "").strip():
        return
    gpu = os.getenv("QWEN_TTS_GPU", "").strip()
    if not gpu:
        gpu = _preparse_gpu_from_argv(sys.argv)
    if gpu is None:
        return
    if gpu == "":
        # 提供了 --gpu 但没给值：不做任何事，后续 argparse 会报错/用户可看到
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


_apply_cuda_visible_devices()

# 屏蔽 pyannote/torchcodec 的 torchcodec 加载失败警告（不影响 WhisperX 功能）
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")
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
# 任务保留天数，超过则自动删除（含僵死的 processing，视为失败）；可用环境变量覆盖
TASK_RETENTION_DAYS = int(os.getenv("QWEN_TTS_TASK_RETENTION_DAYS", "7"))
AUDIO_OUTPUT_DIR = os.path.join(TASKS_DIR, "audio")
REF_AUDIO_DIR = os.path.join(TASKS_DIR, "ref_audio")
SCRIPTS_DIR = os.path.join(APP_DIR, "scripts")
GTCRN_MODEL_PATH = os.path.join(SCRIPTS_DIR, "model_trained_on_dns3.tar")

# 参考音频时长限制（秒）
REF_AUDIO_DURATION_MIN = 3.0  # 官方支持3秒快速克隆
REF_AUDIO_DURATION_MAX = 15.0
# 参考音频目标采样率：上传直接落盘；仅当检测不是 16kHz 时才用 ffmpeg 转码
REF_AUDIO_TARGET_SR = 16000

# 确保任务目录存在
os.makedirs(TASKS_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(REF_AUDIO_DIR, exist_ok=True)


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
        if os.getenv("QWEN_TTS_FORCE_FP32", "").strip().lower() in ("1", "true", "yes"):
            return "cuda:0", torch.float32
        # 2080Ti(Turing, SM75) 不支持 bfloat16；强制 bfloat16 可能导致异常/极慢退化
        try:
            major, _minor = torch.cuda.get_device_capability(0)
        except Exception:
            major = 0
        # 稳定优先：SM80+ 用 bf16；更老架构默认用 fp32（仍在 GPU 上）以避免 fp16 数值不稳导致的概率 NaN/Inf。
        dtype = torch.bfloat16 if major >= 8 else torch.float32
        return "cuda:0", dtype
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


def _delete_ref_audio_file_if_any(params: Optional[Dict]) -> None:
    """删除任务参数中保存的本地参考音频（tasks/ref_audio/ 下）。"""
    if not params:
        return
    rel = params.get("ref_audio_rel")
    if not rel or not isinstance(rel, str):
        return
    p = os.path.join(TASKS_DIR, rel.replace("/", os.sep))
    if os.path.isfile(p):
        try:
            os.remove(p)
        except OSError:
            pass


def _delete_task(user_id: str, task_id: str) -> bool:
    """删除任务：从内存和文件移除，并删除输出音频与参考音频文件。返回是否成功。"""
    # 判定是否真实存在（避免删除不存在的任务也返回成功）
    key = _get_task_key(user_id, task_id)
    task_file = os.path.join(TASKS_DIR, f"{user_id}_{task_id}.json")
    exists = False
    with _TASKS_LOCK:
        exists = key in _TASKS_STORE
    if not exists and not os.path.exists(task_file):
        return False

    task_data = _get_task(user_id, task_id)
    if task_data:
        _delete_ref_audio_file_if_any(task_data.get("params"))
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
        if key in _TASKS_STORE:
            del _TASKS_STORE[key]
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


def _clear_all_tasks() -> Dict[str, int]:
    """清除所有非 processing 状态的任务及其文件，返回统计。"""
    deleted = 0
    skipped = 0

    with _TASKS_LOCK:
        keys_to_delete = []
        for key, task in _TASKS_STORE.items():
            if task.get("status") == "processing":
                skipped += 1
                continue
            keys_to_delete.append(key)

        for key in keys_to_delete:
            task = _TASKS_STORE.pop(key)
            _delete_ref_audio_file_if_any(task.get("params"))
            if task.get("audio_file"):
                audio_path = os.path.join(AUDIO_OUTPUT_DIR, task["audio_file"])
                srt_path, json_path = _subtitle_paths_for_audio(audio_path)
                for p in (audio_path, srt_path, json_path):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            deleted += 1

    for filename in os.listdir(TASKS_DIR):
        if not filename.endswith(".json"):
            continue
        task_file = os.path.join(TASKS_DIR, filename)
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            if task_data.get("status") == "processing":
                skipped += 1
                continue
            os.remove(task_file)
            _delete_ref_audio_file_if_any(task_data.get("params"))
            if task_data.get("audio_file"):
                audio_path = os.path.join(AUDIO_OUTPUT_DIR, task_data["audio_file"])
                srt_path, json_path = _subtitle_paths_for_audio(audio_path)
                for p in (audio_path, srt_path, json_path):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            deleted += 1
        except Exception:
            pass

    return {"deleted": deleted, "skipped_processing": skipped}


def _parse_task_created_at(task: Dict) -> Optional[datetime]:
    """解析任务 created_at 为本地 naive datetime。"""
    raw = task.get("created_at")
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(raw[:26], fmt)
        except ValueError:
            continue
    return None


def _purge_tasks_older_than_days(days: Optional[int] = None) -> int:
    """删除创建时间超过指定天数的任务（含仍为 processing 的僵死任务，视为失败）。返回删除数量。"""
    d = days if days is not None else TASK_RETENTION_DAYS
    if d <= 0:
        return 0
    if not os.path.isdir(TASKS_DIR):
        return 0
    cutoff = datetime.now() - timedelta(days=d)
    removed = 0
    for filename in list(os.listdir(TASKS_DIR)):
        if not filename.endswith(".json"):
            continue
        task_file = os.path.join(TASKS_DIR, filename)
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        created = _parse_task_created_at(task_data)
        if created is None or created >= cutoff:
            continue
        uid = task_data.get("user_id")
        tid = task_data.get("task_id")
        if not uid or not tid:
            continue
        _delete_task(uid, tid)
        removed += 1
    if removed:
        print(f"[任务] 已自动清除超过 {d} 天的任务: {removed} 个")
    return removed


def _task_stats(tasks: List[Dict]) -> Dict[str, int]:
    s = {"completed": 0, "processing": 0, "pending": 0, "failed": 0}
    for t in tasks:
        st = t.get("status")
        if st in s:
            s[st] += 1
    return s


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


def _is_prob_tensor_assert_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "probability tensor contains either" in msg
        or "device-side assert triggered" in msg
        or "cuda error: device-side assert triggered" in msg
    )


def _load_model_with_dtype(model_size: str, dtype: torch.dtype) -> Qwen3TTSModel:
    """Load a fresh Base model instance with specified dtype (no cache)."""
    model_path = _resolve_model_path(model_size)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation="eager",
    )


def _generate_voice_clone_with_fallback(
    model_size: str,
    text: str,
    language: str,
    ref_audio: str,
    ref_text: str,
    x_vector_only_mode: bool,
):
    """
    Generate voice clone audio. If CUDA sampling becomes numerically unstable under fp16
    (probability tensor assert), retry once with fp32 for stability.
    """
    model = _get_model(model_size)
    try:
        return model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )
    except Exception as e:
        msg = str(e).lower()
        if torch.cuda.is_available() and _is_prob_tensor_assert_error(e):
            # device-side assert 会污染当前进程的 CUDA 上下文，后续任何 torch.cuda 调用/加载都可能继续失败。
            if "device-side assert triggered" in msg:
                raise RuntimeError(
                    "CUDA device-side assert triggered. "
                    "该错误会污染当前进程的 CUDA 上下文，无法在同一进程内恢复。"
                    "请重启服务进程后重试；建议设置 QWEN_TTS_FORCE_FP32=1 以提高稳定性（仍在 GPU 上运行）。"
                ) from e

            print("[WARN] CUDA 采样概率异常，尝试使用 GPU(float32) 重新生成一次（更稳定但更慢）")
            # 注意：这里不要强依赖 torch.cuda.empty_cache()，避免极端情况下再次触发 CUDA 错误
            model_fp32 = _load_model_with_dtype(model_size, torch.float32)
            return model_fp32.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
        raise


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
    tail_pad = np.zeros(int(sr * 1.0), dtype=np.float32)
    if speed is None or speed == 1.0:
        return np.concatenate([wav, tail_pad])
    speed = float(speed)
    if speed < 0.1:
        speed = 0.1
    wav = np.concatenate([wav, np.zeros(int(sr * 0.4), dtype=np.float32)])
    stretched = _apply_speed_ffmpeg(wav, sr, speed)
    return np.concatenate([stretched, tail_pad])


def _clean_b64(raw: str) -> str:
    """去除 base64 字符串中的非 ASCII 字符和空白，防止 b64decode 报错。"""
    return re.sub(r'[^A-Za-z0-9+/=]', '', raw)


def _is_riff_wave(audio_bytes: bytes) -> bool:
    return len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"


def _ref_audio_ok_for_model(path: str) -> bool:
    """
    参考音频可直接交给模型：soundfile 能读且采样率为 16kHz（常见为 WAV 16kHz）。
    不满足时再走 ffmpeg 转单声道 16kHz WAV。
    """
    try:
        info = sf.info(path)
        return int(info.samplerate) == REF_AUDIO_TARGET_SR
    except Exception:
        return False


def _guess_ref_audio_suffix(
    params: Dict,
    audio_bytes: bytes,
    data_uri_suffix: Optional[str] = None,
) -> str:
    """为参考音频二进制选择临时文件扩展名（浏览器常为 webm/ogg，误用 .wav 会导致 librosa 加载失败）。"""
    if data_uri_suffix:
        return data_uri_suffix

    ct = (params.get("ref_audio_content_type") or "").lower()
    fn = (params.get("ref_audio_filename") or "").lower()

    if "webm" in ct or "matroska" in ct:
        return ".webm"
    if "ogg" in ct and "video" not in ct:
        return ".ogg"
    if "mpeg" in ct or "/mp3" in ct or ct == "audio/mp3":
        return ".mp3"
    if "mp4" in ct or "m4a" in ct or "aac" in ct:
        return ".m4a"
    if "flac" in ct:
        return ".flac"
    if "wav" in ct or "wave" in ct or ct.endswith("audio/x-wav"):
        return ".wav"

    for ext in (".webm", ".weba"):
        if fn.endswith(ext):
            return ".webm"
    if fn.endswith(".opus"):
        return ".ogg"
    for ext in (".ogg", ".oga"):
        if fn.endswith(ext):
            return ".ogg"
    if fn.endswith(".mp3"):
        return ".mp3"
    if fn.endswith((".m4a", ".mp4", ".aac")):
        return ".m4a"
    if fn.endswith(".flac"):
        return ".flac"
    if fn.endswith(".wav"):
        return ".wav"

    if len(audio_bytes) < 16:
        return ".wav"

    if _is_riff_wave(audio_bytes):
        return ".wav"
    if audio_bytes[:4] == b"\x1aE\xdf\xa3":
        return ".webm"
    if audio_bytes[:4] == b"OggS":
        return ".ogg"
    if audio_bytes[:3] == b"ID3" or (audio_bytes[0] == 0xFF and len(audio_bytes) > 1 and (audio_bytes[1] & 0xE0) == 0xE0):
        return ".mp3"
    if audio_bytes[4:8] == b"ftyp":
        return ".m4a"
    if audio_bytes[:4] == b"fLaC":
        return ".flac"

    # 常见：浏览器录音为 WebM，无可靠文件名时按 webm 处理
    return ".webm"


def _write_ref_audio_temp_for_clone(
    params: Dict,
    audio_bytes: bytes,
    data_uri_suffix: Optional[str] = None,
) -> str:
    """
    将参考音频写入临时文件；若已是 16kHz（可直接读）则不改；否则用 ffmpeg 转单声道 16kHz WAV。
    """
    suffix = _guess_ref_audio_suffix(params, audio_bytes, data_uri_suffix)
    tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tf.write(audio_bytes)
    tf.close()
    raw_path = tf.name

    if _ref_audio_ok_for_model(raw_path):
        return raw_path

    ff = shutil.which("ffmpeg")
    if ff:
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out.close()
        wav_path = out.name
        try:
            cmd = [
                ff, "-y", "-loglevel", "error", "-i", raw_path,
                "-ac", "1", "-ar", str(REF_AUDIO_TARGET_SR), "-f", "wav", wav_path,
            ]
            run_kw: Dict[str, Any] = {"check": True, "timeout": 120}
            if os.name == "nt":
                run_kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(cmd, **run_kw)
            try:
                os.remove(raw_path)
            except OSError:
                pass
            return wav_path
        except Exception:
            try:
                os.remove(wav_path)
            except OSError:
                pass
            return raw_path
    return raw_path


def _ensure_ref_audio_path_for_model(stored_path: str) -> Tuple[str, bool]:
    """
    已落盘的参考音频 → 交给 Qwen 的路径。
    直接保存的文件若已是 16kHz（soundfile 可读）则原样使用；否则 ffmpeg 转单声道 16kHz WAV。
    返回 (路径, 是否需在调用方 finally 中删除该临时文件)。
    """
    if _ref_audio_ok_for_model(stored_path):
        return stored_path, False

    ff = shutil.which("ffmpeg")
    if ff:
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out.close()
        wav_path = out.name
        try:
            cmd = [
                ff, "-y", "-loglevel", "error", "-i", stored_path,
                "-ac", "1", "-ar", str(REF_AUDIO_TARGET_SR), "-f", "wav", wav_path,
            ]
            run_kw: Dict[str, Any] = {"check": True, "timeout": 120}
            if os.name == "nt":
                run_kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(cmd, **run_kw)
            return wav_path, True
        except Exception:
            try:
                os.remove(wav_path)
            except OSError:
                pass
    # 无 ffmpeg 或转换失败：使用原文件（扩展名已在落盘时按类型选对）
    return stored_path, False


def _get_audio_duration_seconds(path_or_b64: str) -> float:
    """
    获取音频时长（秒）。
    path_or_b64: 文件路径 或 base64 字符串（支持 data:audio/...;base64,XXX 或纯 base64）
    """
    path_or_b64 = path_or_b64.strip()

    # 1. 处理 data:audio 格式
    if path_or_b64.startswith("data:") or "," in path_or_b64:
        raw = path_or_b64.split(",", 1)[1] if "," in path_or_b64 else path_or_b64
        data_bytes = base64.b64decode(_clean_b64(raw))
        data, sr = sf.read(io.BytesIO(data_bytes), dtype="float32", always_2d=False)

    # 2. 检查是否为文件路径（必须实际存在）
    elif os.path.isfile(path_or_b64):
        data, sr = sf.read(path_or_b64, dtype="float32", always_2d=False)

    # 3. 作为纯base64解码
    else:
        data_bytes = base64.b64decode(_clean_b64(path_or_b64))
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


def _ensure_trailing_guard(text: str) -> str:
    """在每个片段末尾追加 '…嗯~' 防止模型吃掉最后一个字（社区方案）。"""
    text = text.rstrip()
    if not text:
        return text
    return text + "…嗯~"


def _trim_trailing_guard(wav: np.ndarray, sr: int, search_sec: float = 2.0) -> np.ndarray:
    """裁掉末尾 '…嗯~' 产生的牺牲音频。
    从音频末尾反向检测：跳过尾部静音 → 跳过'嗯'语音 → 找到'…'的静音间隙 → 在此处切断。
    """
    min_samples = int(sr * 0.5)
    if len(wav) < min_samples:
        return wav

    frame_len = int(sr * 0.02)
    hop = frame_len // 2
    search_samples = min(int(sr * search_sec), len(wav))
    tail = wav[-search_samples:]

    n_frames = max(1, (len(tail) - frame_len) // hop + 1)
    rms = np.array([
        np.sqrt(np.mean(tail[i * hop:i * hop + frame_len] ** 2))
        for i in range(n_frames)
    ])

    threshold = max(np.median(rms) * 0.12, np.max(rms) * 0.03)
    is_voice = rms > threshold

    i = n_frames - 1
    while i >= 0 and not is_voice[i]:
        i -= 1
    en_voice_end = i

    while i >= 0 and is_voice[i]:
        i -= 1
    en_silence_end = i

    if en_silence_end <= 0 or en_voice_end <= en_silence_end:
        return wav

    cut_frame = en_silence_end
    cut_sample = len(wav) - search_samples + cut_frame * hop

    if cut_sample < len(wav) // 2:
        return wav

    fade_len = min(int(sr * 0.03), cut_sample)
    fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    result = wav[:cut_sample].copy()
    result[-fade_len:] *= fade
    return result


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


tags_metadata = [
    {"name": "系统", "description": "健康检查、页面等基础接口"},
    {"name": "语音识别", "description": "WhisperX 语音识别（ASR）"},
    {"name": "语音克隆", "description": "Qwen3-TTS 语音合成与克隆"},
    {"name": "任务管理", "description": "异步语音克隆任务的查询、下载、删除"},
]


def _startup_preload_models():
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

    # 4. 清除超过保留期的任务
    try:
        _purge_tasks_older_than_days()
    except Exception as e:
        print(f"[WARN] 过期任务清理失败: {e}")


@asynccontextmanager
async def _lifespan(_app):
    _startup_preload_models()
    yield


app = FastAPI(
    title="Qwen3-TTS 语音克隆 API",
    description="语音合成、语音克隆、语音识别",
    openapi_tags=tags_metadata,
    lifespan=_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["系统"], summary="主页")
def index():
    return FileResponse(INDEX_FILE)


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
        import traceback
        traceback.print_exc()
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
    cleanup_ref_path = None  # 仅删除临时转码的 wav，不删 tasks/ref_audio 下已保存文件
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
        
        # 处理文本（移除拼音标注 + 末尾防截断）
        text = _ensure_trailing_guard(_apply_polyphonic(params["text"]))
        ref_text = _apply_polyphonic(params["ref_text"])
        
        # 参考音频：优先使用提交时已保存的本地文件（避免排队后 base64 再解码丢格式）
        rel = params.get("ref_audio_rel")
        if rel and isinstance(rel, str):
            stored_path = os.path.join(TASKS_DIR, rel.replace("/", os.sep))
            if not os.path.isfile(stored_path):
                raise FileNotFoundError(f"参考音频不存在或已删除: {rel}")
            model_ref_path, is_temp = _ensure_ref_audio_path_for_model(stored_path)
            if is_temp:
                cleanup_ref_path = model_ref_path
        else:
            # 兼容旧任务：仅有 base64
            ref_audio_b64 = (params.get("ref_audio_b64") or "").strip()
            if not ref_audio_b64:
                raise ValueError("任务缺少参考音频（ref_audio_rel 或 ref_audio_b64）")
            data_uri_suffix = None
            if "," in ref_audio_b64:
                header = ref_audio_b64.split(",", 1)[0].lower()
                ref_audio_b64 = ref_audio_b64.split(",", 1)[1]
                for mime, ext in [("webm", ".webm"), ("ogg", ".ogg"), ("mp3", ".mp3"),
                                  ("mpeg", ".mp3"), ("mp4", ".m4a"), ("flac", ".flac")]:
                    if mime in header:
                        data_uri_suffix = ext
                        break
            ref_audio_b64 = re.sub(r"[^A-Za-z0-9+/=]", "", ref_audio_b64)
            audio_bytes = base64.b64decode(ref_audio_b64)
            model_ref_path = _write_ref_audio_temp_for_clone(
                params, audio_bytes, data_uri_suffix=data_uri_suffix
            )
            cleanup_ref_path = model_ref_path

        # 生成语音
        print(f"[任务 {task_id}] 生成语音中...")
        wavs, sr = _generate_voice_clone_with_fallback(
            model_size=params["model"],
            text=text,
            language=params["lang"],
            ref_audio=model_ref_path,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        
        # 裁掉末尾牺牲音频（…嗯~），再应用语速调整
        wav_raw = _trim_trailing_guard(np.asarray(wavs[0], dtype=np.float32), sr)
        wav_out = _apply_speed(wav_raw, params["speed"], sr)
        
        # 保存音频文件
        audio_filename = f"{user_id}_{task_id}.wav"
        audio_path = os.path.join(AUDIO_OUTPUT_DIR, audio_filename)
        
        sf.write(audio_path, wav_out, sr, format="WAV")
        
        # 先标记任务完成（避免字幕生成耗时导致一直卡 processing）
        task_data["status"] = "completed"
        task_data["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task_data["audio_url"] = f"/api/download/{user_id}/{task_id}"
        task_data["audio_file"] = audio_filename
        task_data["sample_rate"] = int(sr)
        _save_task(user_id, task_id, task_data)
        
        print(f"[OK] 任务 {task_id} 完成 (时长: {len(wav_out)/sr:.1f}s)")

        # 生成字幕（可选，失败不影响任务完成状态；生成后再写回字幕字段）
        try:
            if not _check_whisperx_available():
                task_data = _get_task(user_id, task_id) or task_data
                task_data["subtitle_error"] = "WhisperX 未安装，无法为合成结果生成字幕"
                _save_task(user_id, task_id, task_data)
            else:
                t0 = time.time()
                subtitle_data = _generate_subtitles_for_audio(
                    audio_path,
                    params["lang"],
                    original_text=params.get("text", ""),
                )
                if subtitle_data:
                    srt_path, json_path = _write_subtitle_files(subtitle_data, audio_path)
                    task_data = _get_task(user_id, task_id) or task_data
                    if srt_path:
                        task_data["subtitle_srt"] = f"/api/download/{user_id}/{task_id}?type=srt"
                        print(f"[字幕] SRT 已生成: {os.path.basename(srt_path)}")
                    if json_path:
                        task_data["subtitle_json"] = f"/api/download/{user_id}/{task_id}?type=json"
                        print(f"[字幕] JSON 已生成: {os.path.basename(json_path)}")
                    task_data["subtitle_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    task_data["subtitle_elapsed_sec"] = round(time.time() - t0, 3)
                    _save_task(user_id, task_id, task_data)
        except Exception as e:
            task_data = _get_task(user_id, task_id) or task_data
            task_data["subtitle_error"] = str(e)
            _save_task(user_id, task_id, task_data)
            print(f"[字幕] 生成失败(不影响完成): {e}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        err_msg = str(e) or repr(e)
        task_data = _get_task(user_id, task_id)
        if task_data:
            task_data["status"] = "failed"
            task_data["error"] = err_msg
            task_data["failed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _save_task(user_id, task_id, task_data)
        print(f"[FAIL] 任务 {task_id} 失败: {err_msg}")
    
    finally:
        if cleanup_ref_path and os.path.exists(cleanup_ref_path):
            try:
                os.unlink(cleanup_ref_path)
            except OSError:
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
        
        # 处理文本（末尾防截断）
        text = _ensure_trailing_guard(_apply_polyphonic(req.text))
        ref_text = _apply_polyphonic(req.ref_text)
        
        # 生成语音
        wavs, sr = _generate_voice_clone_with_fallback(
            model_size=req.model_size,
            text=text,
            language=req.language,
            ref_audio=req.ref_audio_b64,
            ref_text=ref_text,
            x_vector_only_mode=req.x_vector_only_mode,
        )
        
        # 裁掉末尾牺牲音频（…嗯~），再应用语速调整
        wav_raw = _trim_trailing_guard(np.asarray(wavs[0], dtype=np.float32), sr)
        wav_out = _apply_speed(
            wav_raw,
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
        
        # 读取上传的音频并立即落盘（队列处理时直接读文件，避免 base64 再解码导致格式/扩展名错误）
        audio_content = await ref_audio.read()
        if not audio_content:
            raise HTTPException(status_code=400, detail="参考音频文件为空")

        task_id_final = task_id or _generate_task_id()
        existing_task = _get_task(user_id, task_id_final)
        if existing_task:
            raise HTTPException(
                status_code=400,
                detail=f"任务ID已存在: {task_id_final}",
            )

        meta = {
            "ref_audio_filename": ref_audio.filename or "",
            "ref_audio_content_type": ref_audio.content_type or "",
        }
        suffix = _guess_ref_audio_suffix(meta, audio_content, None)
        ref_basename = f"{user_id}_{task_id_final}{suffix}"
        ref_abs_path = os.path.join(REF_AUDIO_DIR, ref_basename)
        ref_rel = f"ref_audio/{ref_basename}"
        try:
            with open(ref_abs_path, "wb") as out_f:
                out_f.write(audio_content)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"无法保存参考音频: {e}") from e

        try:
            duration = _get_audio_duration_seconds(ref_abs_path)
            _validate_ref_audio_duration(duration)
        except ValueError as e:
            try:
                os.remove(ref_abs_path)
            except OSError:
                pass
            raise HTTPException(status_code=400, detail=str(e)) from e

        # 创建任务记录（不再存 ref_audio_b64，减小 JSON 与内存占用）
        task_data = {
            "task_id": task_id_final,
            "user_id": user_id,
            "status": "pending",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": {
                "text": text,
                "ref_audio_rel": ref_rel,
                "ref_audio_filename": meta["ref_audio_filename"],
                "ref_audio_content_type": meta["ref_audio_content_type"],
                "ref_text": ref_text,
                "speed": float(speed),
                "lang": lang,
                "model": model,
            },
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


@app.get("/api/tasks/{user_id}", tags=["任务管理"], summary="查询用户任务（分页）")
def list_user_tasks(
    user_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(10, ge=1, le=100, description="每页条数，最大 100"),
):
    """
    查询用户的任务列表，按创建时间倒序；支持分页。
    每次调用会先清理全局过期任务：创建超过保留期的任务一律删除（含仍为 processing 的僵死任务）。
    """
    try:
        _purge_tasks_older_than_days()
        tasks = _get_user_tasks(user_id)
        total = len(tasks)
        stats = _task_stats(tasks)
        total_pages = max(1, (total + page_size - 1) // page_size) if total else 1
        if page > total_pages and total_pages >= 1:
            page = total_pages
        start = (page - 1) * page_size
        page_tasks = tasks[start : start + page_size]

        base_url = _base_url(request)
        for task in page_tasks:
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
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "retention_days": TASK_RETENTION_DAYS,
            "stats": stats,
            "tasks": page_tasks,
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


@app.post("/api/task/{user_id}/{task_id}/retry", tags=["任务管理"], summary="重试任务")
def retry_task(user_id: str, task_id: str):
    """使用原任务参数重新排队（复用原 task_id，不创建新任务）。"""
    try:
        task = _get_task(user_id, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task.get("status") == "processing":
            raise HTTPException(status_code=400, detail="任务正在处理中，无法重试（如为僵死任务请先删除后再提交）")
        old_params = task.get("params")
        if not old_params:
            raise HTTPException(status_code=400, detail="无法重试：缺少任务参数")

        params = copy.deepcopy(old_params)
        rel = params.get("ref_audio_rel")
        b64 = params.get("ref_audio_b64")

        # 确保参考音频可用：优先使用已落盘的 ref_audio_rel；否则尝试从旧版 base64 落盘（复用当前 task_id）。
        if rel and isinstance(rel, str):
            src = os.path.join(TASKS_DIR, rel.replace("/", os.sep))
            if not os.path.isfile(src):
                raise HTTPException(status_code=400, detail="参考音频文件已丢失，无法重试")
            params.pop("ref_audio_b64", None)
        elif b64 and isinstance(b64, str) and b64.strip() and b64 != "[已隐藏]":
            meta = {
                "ref_audio_filename": params.get("ref_audio_filename") or "",
                "ref_audio_content_type": params.get("ref_audio_content_type") or "",
            }
            raw = b64.strip()
            if "," in raw:
                raw = raw.split(",", 1)[1]
            raw = re.sub(r"[^A-Za-z0-9+/=]", "", raw)
            audio_bytes = base64.b64decode(raw)
            suffix = _guess_ref_audio_suffix(meta, audio_bytes, None)
            new_rel = f"ref_audio/{user_id}_{task_id}{suffix}"
            dst = os.path.join(TASKS_DIR, new_rel.replace("/", os.sep))
            with open(dst, "wb") as f:
                f.write(audio_bytes)
            params["ref_audio_rel"] = new_rel
            params.pop("ref_audio_b64", None)
        else:
            raise HTTPException(status_code=400, detail="无法重试：缺少参考音频文件或数据")

        # 清理旧输出文件（若存在）
        if task.get("audio_file"):
            audio_path = os.path.join(AUDIO_OUTPUT_DIR, task["audio_file"])
            srt_path, json_path = _subtitle_paths_for_audio(audio_path)
            for p in (audio_path, srt_path, json_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        # 重置任务字段（复用原 task_id）
        task["status"] = "pending"
        task["params"] = params
        task["retry_count"] = int(task.get("retry_count") or 0) + 1
        task["retried_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for k in (
            "started_at",
            "completed_at",
            "failed_at",
            "error",
            "audio_url",
            "audio_file",
            "sample_rate",
            "subtitle_srt",
            "subtitle_json",
            "subtitle_error",
            "subtitle_generated_at",
            "subtitle_elapsed_sec",
        ):
            task.pop(k, None)
        _save_task(user_id, task_id, task)

        _TASK_QUEUE.put({"user_id": user_id, "task_id": task_id, "params": params})
        _start_queue_worker()
        return {
            "ok": True,
            "task_id": task_id,
            "user_id": user_id,
            "status": "pending",
            "message": "已重新加入队列",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/tasks/clear", tags=["任务管理"], summary="一键清除所有任务")
def clear_all_tasks(include_processing: bool = Query(False)):
    """
    清除所有任务及其音频/字幕文件。

    默认保留正在执行中（processing）的任务；若 include_processing=true 则一并删除。
    """
    try:
        if include_processing:
            deleted = 0
            skipped = 0
            # 复制一份 key 列表避免遍历时修改
            with _TASKS_LOCK:
                keys = list(_TASKS_STORE.keys())
            for key in keys:
                try:
                    uid, tid = key.split(":", 1)
                except ValueError:
                    continue
                if _delete_task(uid, tid):
                    deleted += 1
            # 再扫一遍磁盘残留（内存里未加载的）
            for filename in list(os.listdir(TASKS_DIR)):
                if not filename.endswith(".json") or "_" not in filename:
                    continue
                try:
                    uid, rest = filename.split("_", 1)
                    tid = rest[:-5]  # strip .json
                except Exception:
                    continue
                if _delete_task(uid, tid):
                    deleted += 1
            result = {"deleted": deleted, "skipped_processing": skipped}
        else:
            result = _clear_all_tasks()
        return {
            "message": "清除完成",
            "deleted": result["deleted"],
            "skipped_processing": result["skipped_processing"],
            "include_processing": include_processing,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"清除失败: {str(exc)}")


@app.get("/api/download/{user_id}/{task_id}", tags=["任务管理"], summary="下载任务文件")
def download_file(user_id: str, task_id: str, type: str = Query("audio", pattern="^(audio|srt|json)$")):
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
    parser.add_argument("--gpu", type=str, default=os.getenv("QWEN_TTS_GPU", "").strip(),
                        help="使用第几张显卡（写入 CUDA_VISIBLE_DEVICES）。例如 --gpu 1；留空则不设置")
    args = parser.parse_args()
    
    HOST = args.host
    PORT = args.port
    if args.gpu:
        # 此处仅用于打印/与 bat 对齐；真正生效需在 import torch 前设置（文件顶部已做预解析）
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu).strip()
    
    # 启动前打印服务信息
    print("="*60)
    print("[OK] 服务启动完成！")
    print(f"[INFO] WhisperX 模型目录: models/WhisperX")
    print(f"[INFO] 监听地址: http://{HOST}:{PORT}")
    if os.getenv("CUDA_VISIBLE_DEVICES", "").strip():
        print(f"[INFO] CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host=HOST, port=PORT)
