# Qwen3-TTS 语音克隆服务

基于 [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) 构建的语音克隆服务，使用 CUDA Graph 加速推理，内置异步任务队列、Web 界面、WhisperX 语音识别和 GTCRN 音频降噪。

## 功能特性

- **语音克隆 (Voice Clone)** — 3~15 秒参考音频即可复刻任意音色
- **CUDA Graph 加速** — 基于 faster-qwen3-tts，推理速度 6-10x 提升
- **异步任务队列** — 提交即返回，后台排队处理，支持状态轮询和任务管理
- **ASR/字幕** — 集成 WhisperX 用于语音识别与字幕生成（SRT/JSON）
- **GTCRN 降噪** — 参考音频自动降噪，提升克隆质量
- **语速调节** — 基于 FFmpeg 的变速播放（0.1x ~ 5.0x）
- **双模型切换** — 0.6B（快速）和 1.7B（高质量）
- **Web 界面** — 暗色主题单页应用，支持录音、上传、任务管理

## 环境要求

- Python >= 3.10
- CUDA GPU（推荐 8GB+ 显存）
- FFmpeg（语速调节功能）
- Anaconda / Miniconda（推荐）

## 快速开始

### 1. 创建环境

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

### 2. 安装依赖

```bash
# 安装 faster-qwen3-tts（会自动安装 qwen-tts、transformers 等依赖）
cd faster-qwen3-tts
pip install -e .
cd ..

# 安装 API 服务依赖
pip install fastapi uvicorn[standard] python-multipart soundfile numpy requests

# 可选：安装 WhisperX（语音识别 + 字幕生成）
pip install whisperx

# 可选：安装 FlashAttention 2（减少显存占用）
pip install -U flash-attn --no-build-isolation
```

### 3. 下载模型

模型文件需放置在 `models/` 目录下：

```bash
# 通过 ModelScope 下载（国内推荐）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz --local_dir ./models/Qwen3-TTS-Tokenizer-12Hz
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./models/Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir ./models/Qwen3-TTS-12Hz-0.6B-Base
```

### 4. 启动服务

```bash
# 默认启动（4090 / 3090 等新卡）
python api_base.py --port 9770 --gpu 0 --workers 1

# 2080Ti 等较老 GPU（强制 FP32 防止 CUDA assert）
python api_base.py --port 9770 --gpu 0 --workers 1 --fp32

# 或使用 bat 脚本（Windows）
RunQwen3-TTS-9770.bat           # 使用 Anaconda 环境
RunQwen3-TTS.bat                # 使用 runtime/ 内嵌 Python
```

启动后访问 `http://localhost:9770` 打开 Web 界面。

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/clone/upload` | POST | 提交语音克隆任务（文件上传，异步） |
| `/api/voice-clone` | POST | 同步语音克隆（立即返回音频 base64） |
| `/api/task/{user_id}/{task_id}` | GET | 查询任务状态 |
| `/api/tasks/{user_id}` | GET | 查询用户任务列表（分页） |
| `/api/task/{user_id}/{task_id}` | DELETE | 删除任务 |
| `/api/task/{user_id}/{task_id}/retry` | POST | 重试失败任务 |
| `/api/tasks/clear` | DELETE | 一键清除所有任务 |
| `/api/download/{user_id}/{task_id}` | GET | 下载音频/字幕文件 |
| `/api/asr` | POST | 语音识别（WhisperX + GTCRN） |
| `/api/asr/status` | GET | ASR 功能状态 |
| `/health` | GET | 健康检查 |

## 调用示例

### Python — 提交克隆任务并下载

```python
import requests, time

BASE_URL = "http://localhost:9770"

with open("reference.wav", "rb") as f:
    resp = requests.post(f"{BASE_URL}/api/clone/upload", files={
        "ref_audio": ("reference.wav", f, "audio/wav")
    }, data={
        "user_id": "user001",
        "text": "今天天气真不错，适合出去走走。",
        "ref_text": "这是参考音频的文本内容。",
        "lang": "Chinese",
        "model": "1.7B",
    })

task_id = resp.json()["task_id"]

while True:
    status = requests.get(f"{BASE_URL}/api/task/user001/{task_id}").json()
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        raise Exception(status["error"])
    time.sleep(3)

audio = requests.get(f"{BASE_URL}/api/download/user001/{task_id}")
with open("output.wav", "wb") as f:
    f.write(audio.content)
```

## 项目结构

```
Qwen3-TTS/
├── api_base.py                  # API 服务（FastAPI）
├── faster-qwen3-tts/            # TTS 推理引擎（CUDA Graph 加速）
├── models/                      # 模型权重（需自行下载）
│   ├── Qwen3-TTS-12Hz-1.7B-Base/
│   ├── Qwen3-TTS-12Hz-0.6B-Base/
│   ├── Qwen3-TTS-Tokenizer-12Hz/
│   └── WhisperX/
├── scripts/
│   ├── subtitle.py              # WhisperX 字幕生成
│   └── gtcrn.py                 # GTCRN 语音降噪
├── web/index_base.html          # Web 界面
├── tasks/                       # 运行时任务数据与生成音频
├── runtime/                     # 可选：内嵌 Python 环境（不提交 git）
├── RunQwen3-TTS-9770.bat        # Windows 启动脚本（Anaconda）
└── RunQwen3-TTS.bat             # Windows 启动脚本（runtime/）
```

## 参考音频要求

| 要求 | 说明 |
|------|------|
| 时长 | 3 ~ 15 秒（推荐 5 ~ 10 秒） |
| 格式 | WAV / MP3 等常见格式 |
| 采样率 | 建议 16kHz 以上 |
| 质量 | 清晰无噪音、单说话人、音量适中 |
| 文本匹配 | `ref_text` 必须与音频内容精确对应 |

## 致谢

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — 阿里通义千问团队
- [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) — CUDA Graph 加速推理
