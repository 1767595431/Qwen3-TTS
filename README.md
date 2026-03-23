# Qwen3-TTS 语音合成服务

基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 构建的**生产级语音合成服务平台**，提供语音克隆、自定义音色、音色设计三种能力，内置异步任务队列、Web 管理界面、ASR 语音识别和音频降噪。

## 功能特性

- **语音克隆 (Voice Clone)** — 3~15 秒参考音频即可复刻任意音色
- **自定义音色 (Custom Voice)** — 9 种预设音色 + 自然语言指令控制情感/风格
- **音色设计 (Voice Design)** — 用文字描述想要的音色，模型自动生成
- **10 语种支持** — 中/英/日/韩/德/法/俄/葡/西/意，支持自动语言检测
- **异步任务队列** — 提交即返回，后台排队处理，支持状态轮询和任务管理
- **ASR 辅助** — 集成 SenseVoice 自动识别参考音频文本，降低使用门槛
- **GTCRN 降噪** — 参考音频自动降噪，提升克隆质量
- **语速调节** — 基于 FFmpeg 的变速播放（0.1x ~ 5.0x）
- **多音字纠正** — 内置中文多音字映射表
- **双模型切换** — 同时加载 0.6B（快速）和 1.7B（高质量）模型
- **Web 界面** — 暗色主题单页应用，支持录音、上传、任务管理

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                      Web 前端                            │
│  index.html (全功能 3 模式)    index_base.html (克隆专用)  │
└────────────┬─────────────────────────────┬───────────────┘
             │                             │
     ┌───────▼────────┐           ┌───────▼────────┐
     │ api_server.py  │           │  api_base.py   │
     │  端口 8001      │           │ 端口 9770/9771  │
     │ CustomVoice     │           │ Voice Clone    │
     │ VoiceDesign     │           │ + SenseVoice   │
     │ Voice Clone     │           │ + GTCRN 降噪    │
     └───────┬────────┘           └───────┬────────┘
             │                             │
             └─────────────┬───────────────┘
                           │
             ┌─────────────▼─────────────┐
             │    qwen_tts (核心引擎)     │
             │  Talker LLM (0.6B / 1.7B) │
             │  Speaker Encoder (ECAPA)   │
             │  Speech Tokenizer (12Hz)   │
             └───────────────────────────┘
```

## 环境要求

- Python >= 3.9
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
# 从源码安装核心包
pip install -e .

# 安装 FlashAttention 2（减少显存占用，推荐）
pip install -U flash-attn --no-build-isolation

# 安装 ASR 功能（可选，用于自动识别参考音频文本）
pip install -r requirements-asr.txt
```

### 3. 下载模型

模型文件需放置在 `models/` 目录下：

```bash
# 通过 ModelScope 下载（国内推荐）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz --local_dir ./models/Qwen3-TTS-Tokenizer-12Hz
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./models/Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir ./models/Qwen3-TTS-12Hz-0.6B-Base

# 可选：下载 CustomVoice 和 VoiceDesign 模型（api_server.py 使用）
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local_dir ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

### 4. 启动服务

**语音克隆专用服务**（推荐，含 ASR + 降噪）：

```bash
# 默认端口 9770
python api_base.py --port 9770

# 或使用 bat 脚本（Windows）
RunQwen3-TTS-9770.bat
```

**全功能服务**（CustomVoice + VoiceDesign + Clone）：

```bash
python api_server.py --port 8001
```

启动后访问 `http://localhost:9770`（或 `http://localhost:8001`）打开 Web 界面。

## 两套 API 服务

| 特性 | `api_server.py` | `api_base.py` |
|------|:---------------:|:-------------:|
| 默认端口 | 8001 | 9770 / 9771 |
| CustomVoice | ✅ | - |
| VoiceDesign | ✅ | - |
| Voice Clone | ✅ | ✅ |
| ASR 语音识别 | - | ✅ (SenseVoice) |
| 音频降噪 | - | ✅ (GTCRN) |
| 异步任务队列 | 基础 | 完整 (持久化) |
| 任务管理 CRUD | 基础 | 完整 + 重试 |
| 前端页面 | `web/index.html` | `web/index_base.html` |
| 模型切换 | 所有 5 个模型 | 0.6B / 1.7B Base |

## API 接口

### 语音克隆服务 (api_base.py)

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/clone/upload` | POST | 提交语音克隆任务（文件上传） |
| `/api/voice-clone` | POST | 同步语音克隆 |
| `/api/task/{user_id}/{task_id}` | GET | 查询任务状态 |
| `/api/tasks/{user_id}` | GET | 查询用户任务（分页；超保留期自动清理，含僵死 processing） |
| `/api/task/{user_id}/{task_id}` | DELETE | 删除任务 |
| `/api/download/{user_id}/{task_id}` | GET | 下载生成的音频 |
| `/api/queue/status` | GET | 查询队列状态 |
| `/api/asr` | POST | 语音识别（SenseVoice + GTCRN） |
| `/api/asr/status` | GET | ASR 功能状态 |
| `/health` | GET | 健康检查 |

### 全功能服务 (api_server.py)

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/custom-voice` | POST | 预设音色合成（含指令控制） |
| `/api/voice-design` | POST | 音色设计合成 |
| `/api/voice-clone` | POST | 同步语音克隆 |
| `/api/clone` | POST | 异步语音克隆任务 |

> 完整接口文档见 [语音克隆接口文档](qwen3-tts语音克隆文档.md)

## 调用示例

### Python — 提交克隆任务并下载

```python
import requests
import time

BASE_URL = "http://localhost:9770"

# 1. 提交任务
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

task = resp.json()
task_id = task["task_id"]

# 2. 轮询等待
while True:
    status = requests.get(f"{BASE_URL}/api/task/user001/{task_id}").json()
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        raise Exception(status["error"])
    time.sleep(3)

# 3. 下载音频
audio = requests.get(f"{BASE_URL}/api/download/user001/{task_id}")
with open("output.wav", "wb") as f:
    f.write(audio.content)
```

### Python — 直接使用核心引擎

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "models/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

wavs, sr = model.generate_voice_clone(
    text="你好，很高兴认识你。",
    language="Chinese",
    ref_audio="reference.wav",
    ref_text="这是参考音频的文本。",
)
sf.write("output.wav", wavs[0], sr)
```

### cURL

```bash
# 提交克隆任务
curl -X POST "http://localhost:9770/api/clone/upload" \
  -F "user_id=user001" \
  -F "text=今天天气真不错" \
  -F "ref_text=参考音频的文本" \
  -F "ref_audio=@reference.wav" \
  -F "lang=Chinese" \
  -F "model=1.7B"

# 查询状态
curl "http://localhost:9770/api/task/user001/{task_id}"

# 下载音频
curl -o output.wav "http://localhost:9770/api/download/user001/{task_id}"
```

## 预设音色

`api_server.py` 的 CustomVoice 模式支持以下预设音色：

| 音色 | 描述 | 母语 |
|------|------|------|
| Vivian | 明亮、略带锋芒的年轻女声 | 中文 |
| Serena | 温暖、温柔的年轻女声 | 中文 |
| Uncle_Fu | 低沉醇厚的成熟男声 | 中文 |
| Dylan | 清澈自然的北京腔男声 | 中文 (北京话) |
| Eric | 略带沙哑的活泼成都腔男声 | 中文 (四川话) |
| Ryan | 富有节奏感的动感男声 | 英文 |
| Aiden | 阳光明朗的美式男声 | 英文 |
| Ono_Anna | 活泼轻盈的日本女声 | 日文 |
| Sohee | 情感丰富的温暖韩国女声 | 韩文 |

## 模型说明

| 模型 | 参数量 | 功能 | 流式 | 指令控制 |
|------|:------:|------|:----:|:--------:|
| Qwen3-TTS-12Hz-1.7B-Base | 1.7B | 语音克隆、微调基座 | ✅ | - |
| Qwen3-TTS-12Hz-0.6B-Base | 0.6B | 语音克隆（轻量） | ✅ | - |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 1.7B | 预设音色 + 情感控制 | ✅ | ✅ |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 0.6B | 预设音色 | ✅ | - |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 1.7B | 自然语言音色设计 | ✅ | ✅ |
| Qwen3-TTS-Tokenizer-12Hz | - | 语音编解码器 | - | - |

## 微调训练

支持基于 Base 模型的 SFT 微调，训练自定义音色：

```bash
# 1. 准备数据（JSONL 格式，提取 audio_codes）
python finetuning/prepare_data.py \
    --input_jsonl data/train.jsonl \
    --output_jsonl data/train_processed.jsonl \
    --model_path models/Qwen3-TTS-Tokenizer-12Hz

# 2. 训练
accelerate launch finetuning/sft_12hz.py \
    --init_model_path models/Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path models/my-custom-voice \
    --train_jsonl data/train_processed.jsonl \
    --speaker_name "my_speaker"
```

> 详细说明见 [finetuning/README.md](finetuning/README.md)

## 项目结构

```
Qwen3-TTS/
├── api_base.py                  # 语音克隆专用 API 服务
├── api_server.py                # 全功能 API 服务
├── web/
│   ├── index.html               # 全功能 Web 界面
│   └── index_base.html          # 克隆专用 Web 界面
├── qwen_tts/                    # 核心 TTS 引擎
│   ├── core/
│   │   ├── models/              # Talker 模型 + Speaker Encoder
│   │   ├── tokenizer_12hz/      # 12Hz 语音编解码器
│   │   └── tokenizer_25hz/      # 25Hz 语音编解码器
│   ├── inference/               # 高层推理 API
│   └── cli/demo.py              # Gradio 演示
├── models/                      # 模型权重（需自行下载）
├── SenseVoiceSmall/             # SenseVoice ASR 模型
├── scripts/gtcrn.py             # GTCRN 语音降噪
├── finetuning/                  # 微调训练工具
├── configs/polyphonic_map.json  # 多音字映射表
├── examples/                    # 示例脚本
├── tasks/                       # 运行时任务数据与生成音频
├── RunQwen3-TTS-9770.bat        # Windows 启动脚本 (端口 9770)
└── RunQwen3-TTS-9771.bat        # Windows 启动脚本 (端口 9771)
```

## 参考音频要求

| 要求 | 说明 |
|------|------|
| 时长 | 3 ~ 15 秒（推荐 5 ~ 10 秒） |
| 格式 | WAV / MP3 等常见格式 |
| 采样率 | 建议 16kHz 以上 |
| 质量 | 清晰无噪音、单说话人、音量适中 |
| 文本匹配 | `ref_text` 必须与音频内容精确对应 |

## 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习框架 | PyTorch + Transformers 4.57.3 |
| 推理加速 | Accelerate, Flash Attention 2 |
| API 框架 | FastAPI + Uvicorn |
| 前端 | 原生 HTML/CSS/JS（暗色主题 SPA） |
| ASR | FunASR + SenseVoice (ModelScope) |
| 音频降噪 | GTCRN |
| 音频处理 | librosa, soundfile, torchaudio, FFmpeg |
| 训练 | Accelerate + TensorBoard |

## 致谢

本项目基于阿里通义千问团队开源的 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 构建。

```bibtex
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and others},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```

## 许可证

[Apache-2.0](LICENSE)
