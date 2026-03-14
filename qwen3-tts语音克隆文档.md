# 语音克隆接口文档（api_base.py）

> 本文档详细说明 `api_base.py` 提供的所有语音合成、管理及字幕生成接口

## 📋 目录

- [系统架构](#系统架构)
- [接口总览](#接口总览)
- [核心接口详解](#核心接口详解)
  - [1. 提交语音克隆任务（异步）](#1-提交语音克隆任务异步)
  - [2. 语音克隆（同步）](#2-语音克隆同步)
  - [3. 查询任务状态](#3-查询任务状态)
  - [4. 查询用户所有任务](#4-查询用户所有任务)
  - [5. 删除任务](#5-删除任务)
  - [6. 下载任务文件（音频/字幕）](#6-下载任务文件音频字幕)
  - [7. 查询队列状态](#7-查询队列状态)
- [辅助接口](#辅助接口)
  - [8. 语音识别（ASR）](#8-语音识别asr)
  - [9. 检查ASR状态](#9-检查asr状态)
  - [10. 健康检查](#10-健康检查)
- [字幕生成说明](#字幕生成说明)
- [完整示例](#完整示例)
- [错误处理](#错误处理)
- [注意事项](#注意事项)

---

## 系统架构

### 🎯 异步任务队列系统

本API采用**异步任务队列架构**，确保高效稳定的语音合成服务：

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  前端/客户端 │ ──→ │ 提交任务接口  │ ──→ │  任务队列       │
└─────────────┘     └──────────────┘     │  (pending)     │
                                         └────────────────┘
                                                 ↓
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  轮询查询    │ ←── │ 查询任务状态  │ ←── │  任务处理器     │
│  (2-5秒/次)  │     └──────────────┘     │  (processing)  │
└─────────────┘                           └────────────────┘
                                                 ↓
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  下载音频    │ ←── │ 下载接口      │ ←── │  任务完成       │
└─────────────┘     └──────────────┘     │  (completed)   │
                                         └────────────────┘
```

### ✨ 核心特性

1. **队列管理**：所有任务按提交顺序排队，避免资源竞争
2. **单线程处理**：同时只处理一个任务，确保稳定性
3. **实时状态**：支持查询任务状态、队列位置、进度
4. **持久化存储**：任务信息保存到文件，服务重启后可恢复
5. **音频缓存**：生成的音频文件可随时下载

---

## 接口总览

| 接口路径 | 方法 | 功能说明 |
|---------|------|---------|
| `/api/clone/upload` | POST | 🔹 提交语音克隆任务（异步，文件上传） |
| `/api/voice-clone` | POST | 🔹 语音克隆（同步，立即返回音频+字幕） |
| `/api/task/{user_id}/{task_id}` | GET | 🔹 查询指定任务状态 |
| `/api/tasks/{user_id}` | GET | 🔹 查询用户所有任务列表 |
| `/api/task/{user_id}/{task_id}` | DELETE | 🔹 删除任务及音频/字幕文件 |
| `/api/download/{user_id}/{task_id}?type=audio\|srt\|json` | GET | 🔹 下载任务文件（音频/SRT字幕/JSON字幕） |
| `/api/queue/status` | GET | 🔹 查询任务队列状态 |
| `/api/asr` | POST | 🔸 语音识别（WhisperX + GTCRN降噪） |
| `/api/asr/status` | GET | 🔸 检查ASR功能是否可用 |
| `/health` | GET | 🔸 健康检查 |

> 🔹 核心接口 | 🔸 辅助接口

---

## 核心接口详解

### 1. 提交语音克隆任务（异步）

**接口地址**
```
POST /api/clone/upload
```

**功能说明**

直接上传音频文件提交语音克隆任务到队列，系统将按顺序异步处理。任务完成后会自动生成 SRT 和 JSON 格式的字幕文件（需要 WhisperX 支持）。

**请求方式**

`multipart/form-data` 表单上传

**请求参数**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|-------|------|------|--------|------|
| `user_id` | string | ✅ | - | 用户唯一标识 |
| `text` | string | ✅ | - | 要合成的目标文本 |
| `ref_text` | string | ✅ | - | 参考音频对应的转写文本 |
| `ref_audio` | File | ✅ | - | 参考音频文件（3-15秒，WAV/MP3等） |
| `task_id` | string | ❌ | 自动生成 | 任务ID（可自定义或系统生成9位数字） |
| `speed` | float | ❌ | 1.0 | 语速调整系数（0.1-5.0），<1慢速，>1快速 |
| `lang` | string | ❌ | English | 目标语言，支持10种主要语言及自动检测（详见[语言选项](#b-语言选项完整列表)） |
| `model` | string | ❌ | 1.7B | 模型大小：`0.6B`（快）或`1.7B`（高质量） |

**参考音频要求**

- ✅ **时长**：3-15秒（强制要求，官方支持3秒快速克隆）
- ✅ **格式**：支持WAV、MP3等常见格式
- ✅ **采样率**：建议24000Hz或更高
- ✅ **质量**：清晰无噪音，单说话人
- ✅ **内容**：`ref_text`必须与音频内容精确匹配

**响应示例**

```json
{
  "task_id": "123456789",
  "user_id": "user001",
  "status": "pending",
  "created_at": "2026-02-01 10:30:00",
  "queue_position": 1,
  "message": "任务已提交，请轮询 /api/task/{user_id}/{task_id} 查询状态"
}
```

**响应字段说明**

| 字段 | 类型 | 说明 |
|-----|------|------|
| `task_id` | string | 任务唯一ID |
| `user_id` | string | 用户标识 |
| `status` | string | 任务状态：`pending`（等待中） |
| `created_at` | string | 任务创建时间 |
| `queue_position` | integer | 队列位置（1表示下一个处理） |
| `message` | string | 提示信息 |

**错误响应**

```json
{
  "detail": "参考音频时长为 2.5 秒，必须在 3-15 秒之间"
}
```

**HTML表单示例**

```html
<form id="cloneForm">
  <input type="text" name="user_id" value="user001" required>
  <textarea name="text" placeholder="要合成的文本" required></textarea>
  <textarea name="ref_text" placeholder="参考音频的文本" required></textarea>
  <input type="file" name="ref_audio" accept="audio/*" required>
  <input type="number" name="speed" value="1.0" step="0.1" min="0.1" max="5.0">
  <select name="lang">
    <option value="Auto">自动检测</option>
    <option value="Chinese">中文</option>
    <option value="English">英语</option>
  </select>
  <button type="submit">提交任务</button>
</form>

<script>
document.getElementById('cloneForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  
  const response = await fetch('/api/clone/upload', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  console.log('任务已提交:', result.task_id);
});
</script>
```

**JavaScript示例**

```javascript
async function submitTask(audioFile, text, refText) {
  const formData = new FormData();
  formData.append('user_id', 'user001');
  formData.append('text', text);
  formData.append('ref_text', refText);
  formData.append('ref_audio', audioFile);
  formData.append('speed', '1.0');
  formData.append('lang', 'Chinese');
  formData.append('model', '1.7B');
  
  const response = await fetch('http://localhost:8001/api/clone/upload', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// 使用示例
const fileInput = document.getElementById('audioInput');
const audioFile = fileInput.files[0];
const result = await submitTask(
  audioFile, 
  '今天天气真不错', 
  '这是参考音频的文本'
);
console.log('任务ID:', result.task_id);
```

**Python示例**

```python
import requests

# 读取音频文件
with open('reference.wav', 'rb') as f:
    files = {'ref_audio': ('reference.wav', f, 'audio/wav')}
    data = {
        'user_id': 'user001',
        'text': '今天天气真不错，适合出去走走。',
        'ref_text': '这是参考音频的文本内容。',
        'speed': '1.0',
        'lang': 'Chinese',
        'model': '1.7B'
    }
    
    response = requests.post(
        'http://localhost:8001/api/clone/upload',
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"任务ID: {result['task_id']}")
```

**cURL示例**

```bash
curl -X POST "http://localhost:8001/api/clone/upload" \
  -F "user_id=user001" \
  -F "text=今天天气真不错，适合出去走走。" \
  -F "ref_text=这是参考音频的文本内容。" \
  -F "ref_audio=@reference.wav" \
  -F "speed=1.0" \
  -F "lang=Chinese" \
  -F "model=1.7B"
```

---

### 2. 语音克隆（同步）

**接口地址**
```
POST /api/voice-clone
```

**功能说明**

同步语音克隆接口，立即返回音频 base64 和内联字幕数据，无需轮询。适合短文本快速合成场景。

**请求方式**

`application/json`

**请求参数**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|-------|------|------|--------|------|
| `text` | string | ✅ | - | 要合成的目标文本 |
| `ref_audio_b64` | string | ✅ | - | 参考音频的 base64 编码（纯字符串，不含 DataURL 前缀） |
| `ref_text` | string | ✅ | - | 参考音频对应的转写文本 |
| `speed` | float | ❌ | 1.0 | 语速调整系数（0.1-5.0） |
| `speed_enabled` | boolean | ❌ | true | 是否启用语速调整 |
| `language` | string | ❌ | English | 目标语言 |
| `model_size` | string | ❌ | 1.7B | 模型大小：`0.6B` 或 `1.7B` |
| `x_vector_only_mode` | boolean | ❌ | false | 是否只使用 x-vector 模式 |

**响应示例（含字幕）**

```json
{
  "sample_rate": 24000,
  "audio_b64": "UklGRiQAAABXQVZFZm10IBAAAA...",
  "subtitle_generated": true,
  "subtitle_language": "zh",
  "subtitle_srt": "1\n00:00:00,031 --> 00:00:03,668\n今天天气真不错...\n",
  "subtitle_json": {
    "language": "zh",
    "segments": [
      {
        "start": 0.031,
        "end": 3.668,
        "text": "今天天气真不错"
      }
    ]
  }
}
```

**字幕相关响应字段**

| 字段 | 类型 | 说明 |
|-----|------|------|
| `subtitle_generated` | boolean | 字幕是否成功生成 |
| `subtitle_language` | string | 字幕检测到的语言代码（如 `zh`、`en`） |
| `subtitle_srt` | string | SRT 格式字幕文本（完整内容） |
| `subtitle_json` | object | JSON 格式字幕数据（含 segments 数组） |
| `subtitle_error` | string | 字幕生成失败时的错误信息（仅在失败时出现） |

**响应示例（字幕不可用）**

```json
{
  "sample_rate": 24000,
  "audio_b64": "UklGRiQAAABXQVZFZm10IBAAAA...",
  "subtitle_generated": false,
  "subtitle_error": "WhisperX 未安装，无法为合成结果生成字幕"
}
```

**cURL示例**

```bash
curl -X POST "http://localhost:8001/api/voice-clone" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "今天天气真不错，适合出去走走。",
    "ref_audio_b64": "UklGRiQAAABXQVZFZm10IBAAAA...",
    "ref_text": "这是参考音频的文本内容",
    "speed": 1.0,
    "language": "Chinese",
    "model_size": "1.7B"
  }'
```

---

### 3. 查询任务状态

**接口地址**
```
GET /api/task/{user_id}/{task_id}
```

**功能说明**

查询指定任务的当前状态、进度和结果。

**路径参数**

| 参数名 | 类型 | 说明 |
|-------|------|------|
| `user_id` | string | 用户标识 |
| `task_id` | string | 任务ID |

**响应示例（不同状态）**

**状态1：等待中（pending）**
```json
{
  "task_id": "123456789",
  "user_id": "user001",
  "status": "pending",
  "created_at": "2026-02-01 10:30:00"
}
```

**状态2：处理中（processing）**
```json
{
  "task_id": "123456789",
  "user_id": "user001",
  "status": "processing",
  "created_at": "2026-02-01 10:30:00",
  "started_at": "2026-02-01 10:30:05"
}
```

**状态3：已完成（completed）**
```json
{
  "task_id": "123456789",
  "user_id": "user001",
  "status": "completed",
  "created_at": "2026-02-01 10:30:00",
  "started_at": "2026-02-01 10:30:05",
  "completed_at": "2026-02-01 10:30:20",
  "audio_url": "http://localhost:8001/api/download/user001/123456789",
  "audio_file": "user001_123456789.wav",
  "sample_rate": 24000,
  "subtitle_srt": "http://localhost:8001/api/download/user001/123456789?type=srt",
  "subtitle_json": "http://localhost:8001/api/download/user001/123456789?type=json"
}
```

> `subtitle_srt` 和 `subtitle_json` 仅在 WhisperX 可用且字幕生成成功时出现。若字幕生成失败，会返回 `subtitle_error` 字段说明原因。

**状态4：失败（failed）**
```json
{
  "task_id": "123456789",
  "user_id": "user001",
  "status": "failed",
  "created_at": "2026-02-01 10:30:00",
  "error": "参考音频解码失败: Invalid base64 string",
  "failed_at": "2026-02-01 10:30:10"
}
```

**任务状态流转图**

```
pending → processing → completed
   ↓           ↓
   └─────→ failed
```

**cURL示例**

```bash
curl "http://localhost:8001/api/task/user001/123456789"
```

---

### 4. 查询用户所有任务

**接口地址**
```
GET /api/tasks/{user_id}
```

**功能说明**

获取指定用户的所有任务列表（按创建时间倒序）。

**路径参数**

| 参数名 | 类型 | 说明 |
|-------|------|------|
| `user_id` | string | 用户标识 |

**响应示例**

```json
{
  "user_id": "user001",
  "total": 5,
  "tasks": [
    {
      "task_id": "123456789",
      "status": "completed",
      "created_at": "2026-02-01 10:30:00",
      "audio_url": "http://localhost:8001/api/download/user001/123456789"
    },
    {
      "task_id": "123456788",
      "status": "processing",
      "created_at": "2026-02-01 10:25:00"
    },
    {
      "task_id": "123456787",
      "status": "failed",
      "created_at": "2026-02-01 10:20:00",
      "error": "音频时长不符合要求"
    }
  ]
}
```

**cURL示例**

```bash
curl "http://localhost:8001/api/tasks/user001"
```

---

### 5. 删除任务

**接口地址**
```
DELETE /api/task/{user_id}/{task_id}
```

**功能说明**

删除指定任务及其生成的音频文件。

**路径参数**

| 参数名 | 类型 | 说明 |
|-------|------|------|
| `user_id` | string | 用户标识 |
| `task_id` | string | 任务ID |

**响应示例**

```json
{
  "message": "任务已删除",
  "task_id": "123456789"
}
```

**cURL示例**

```bash
curl -X DELETE "http://localhost:8001/api/task/user001/123456789"
```

---

### 6. 下载任务文件（音频/字幕）

**接口地址**
```
GET /api/download/{user_id}/{task_id}?type=audio|srt|json
```

**功能说明**

下载任务生成的文件，支持三种类型：音频（WAV）、SRT 字幕、JSON 字幕。

**路径参数**

| 参数名 | 类型 | 说明 |
|-------|------|------|
| `user_id` | string | 用户标识 |
| `task_id` | string | 任务ID |

**查询参数**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|-------|------|------|--------|------|
| `type` | string | ❌ | `audio` | 文件类型：`audio`（WAV音频）、`srt`（SRT字幕）、`json`（JSON字幕） |

**各类型响应**

| type 值 | Content-Type | 文件名示例 | 说明 |
|---------|-------------|-----------|------|
| `audio` | `audio/wav` | `user001_123456789.wav` | 生成的语音音频文件 |
| `srt` | `text/plain; charset=utf-8` | `123456789.srt` | SRT 格式字幕文件 |
| `json` | `application/json; charset=utf-8` | `123456789_subtitle.json` | JSON 格式字幕数据 |

**SRT 字幕示例**

```
1
00:00:00,031 --> 00:00:03,668
今天天气真不错，适合出去走走。

2
00:00:03,890 --> 00:00:07,250
阳光明媚，万里无云。
```

**JSON 字幕示例**

```json
{
  "language": "zh",
  "segments": [
    {
      "start": 0.031,
      "end": 3.668,
      "text": "今天天气真不错，适合出去走走。"
    },
    {
      "start": 3.890,
      "end": 7.250,
      "text": "阳光明媚，万里无云。"
    }
  ]
}
```

**cURL示例**

```bash
# 下载音频（默认）
curl -O "http://localhost:8001/api/download/user001/123456789"

# 下载 SRT 字幕
curl "http://localhost:8001/api/download/user001/123456789?type=srt" -o output.srt

# 下载 JSON 字幕
curl "http://localhost:8001/api/download/user001/123456789?type=json" -o subtitle.json
```

**错误响应**

```json
{
  "detail": "SRT 字幕文件不存在（WhisperX 可能未安装或生成失败）"
}
```

> 字幕文件在任务完成时由 WhisperX 自动生成。如果 WhisperX 未安装或字幕生成失败，对应的字幕文件将不存在，但不影响音频文件的正常下载。

---

### 7. 查询队列状态

**接口地址**
```
GET /api/queue/status
```

**功能说明**

查询任务队列的当前状态（等待任务数、工作线程状态）。

**响应示例**

```json
{
  "queue_size": 3,
  "worker_running": true
}
```

**响应字段说明**

| 字段 | 类型 | 说明 |
|-----|------|------|
| `queue_size` | integer | 队列中等待处理的任务数量 |
| `worker_running` | boolean | 工作线程是否正在运行 |

**cURL示例**

```bash
curl "http://localhost:8001/api/queue/status"
```

---

## 辅助接口

### 8. 语音识别（ASR）

**接口地址**
```
POST /api/asr
```

**功能说明**

使用 **WhisperX + GTCRN降噪** 识别音频文本，用于自动生成 `ref_text`。WhisperX 同时也被用于字幕生成中的时间戳对齐。

**请求方式**

`multipart/form-data` 文件上传

**请求参数**

| 参数名 | 类型 | 必填 | 说明 |
|-------|------|------|------|
| `file` | File | ✅ | 音频文件（3-15秒，支持WAV/MP3等） |

**处理流程**

```
上传音频 → GTCRN降噪增强(可选) → WhisperX 识别 → 返回文本
```

**响应示例**

```json
{
  "text": "这是识别出的音频文本内容"
}
```

**错误响应**

```json
{
  "detail": "WhisperX 未就绪。请安装: pip install whisperx"
}
```

**JavaScript示例**

```javascript
const formData = new FormData();
formData.append("file", audioFile);

const response = await fetch("http://localhost:8001/api/asr", {
  method: "POST",
  body: formData
});

const data = await response.json();
console.log("识别结果:", data.text);
```

**cURL示例**

```bash
curl -X POST "http://localhost:8001/api/asr" \
  -F "file=@reference.wav"
```

---

### 9. 检查ASR状态

**接口地址**
```
GET /api/asr/status
```

**功能说明**

检查 WhisperX 语音识别和 GTCRN 降噪功能是否可用。WhisperX 同时负责语音识别和字幕时间戳生成。

**响应示例**

```json
{
  "available": true,
  "enhancer": true,
  "duration_min": 3,
  "duration_max": 15
}
```

**响应字段说明**

| 字段 | 类型 | 说明 |
|-----|------|------|
| `available` | boolean | WhisperX 是否可用（影响 ASR 和字幕生成） |
| `enhancer` | boolean | GTCRN 降噪是否可用 |
| `duration_min` | integer | 最小音频时长（秒） |
| `duration_max` | integer | 最大音频时长（秒） |

**cURL示例**

```bash
curl "http://localhost:8001/api/asr/status"
```

---

### 10. 健康检查

**接口地址**
```
GET /health
```

**功能说明**

检查API服务是否正常运行。

**响应示例**

```json
{
  "ok": true
}
```

**cURL示例**

```bash
curl "http://localhost:8001/health"
```

---

## 字幕生成说明

### 工作原理

系统在语音合成完成后，自动使用 **WhisperX** 对生成的音频进行语音识别和强制对齐，从而产生精确的时间戳字幕。如果提交任务时提供了原文（`text` 字段），字幕文本将使用原文而非 ASR 识别结果，确保文字完全准确。

```
TTS 合成音频 → WhisperX 识别 + 强制对齐 → 原文对齐匹配 → 生成 SRT + JSON 字幕
```

### 两种接口的字幕返回方式

| 接口 | 字幕返回方式 | 说明 |
|-----|-------------|------|
| `POST /api/voice-clone`（同步） | 响应 JSON 中内联返回 | `subtitle_srt`（SRT文本）、`subtitle_json`（JSON对象）直接包含在响应体中 |
| `POST /api/clone/upload`（异步） | 通过下载接口获取 | 任务完成后，`subtitle_srt` 和 `subtitle_json` 字段包含下载 URL |

### 字幕输出格式

**SRT 格式**（标准字幕格式，兼容各类播放器）

```
1
00:00:00,031 --> 00:00:03,668
3月5日，参加江苏代表团审议，

2
00:00:03,890 --> 00:00:09,123
习近平总书记就抓好"十五五"经济社会发展提出明确要求，
```

**JSON 格式**（结构化数据，方便程序处理）

```json
{
  "language": "zh",
  "segments": [
    {
      "start": 0.031,
      "end": 3.668,
      "text": "3月5日，参加江苏代表团审议，"
    },
    {
      "start": 3.890,
      "end": 9.123,
      "text": "习近平总书记就抓好"十五五"经济社会发展提出明确要求，"
    }
  ]
}
```

### 前置条件

- 需要安装 WhisperX：`pip install whisperx`
- 首次使用时会自动下载 WhisperX 模型（需要网络访问 HuggingFace）
- 字幕生成失败不会影响 TTS 主流程，音频文件依然正常返回

---

## 完整示例

### Python完整调用示例

```python
import base64
import requests
import time

# ============ 配置 ============
BASE_URL = "http://localhost:8001"
USER_ID = "user001"
REF_AUDIO_PATH = "reference.wav"

# ============ 1. 读取参考音频 ============
print("📂 读取参考音频...")
with open(REF_AUDIO_PATH, "rb") as f:
    ref_audio_data = f.read()
    ref_audio_b64 = base64.b64encode(ref_audio_data).decode('utf-8')

# ============ 2. ASR识别参考文本（可选）============
print("🔊 识别参考音频文本...")
with open(REF_AUDIO_PATH, "rb") as f:
    asr_resp = requests.post(
        f"{BASE_URL}/api/asr",
        files={"file": f}
    )

if asr_resp.status_code == 200:
    ref_text = asr_resp.json()["text"]
    print(f"✅ 识别结果: {ref_text}")
else:
    # 如果ASR不可用，手动输入
    ref_text = input("请输入参考音频的文本内容: ")

# ============ 3. 提交语音克隆任务 ============
print("\n📤 提交语音克隆任务...")
payload = {
    "user_id": USER_ID,
    "text": "今天天气真不错，适合出去走走。",
    "ref_audio_b64": ref_audio_b64,
    "ref_text": ref_text,
    "speed": 1.0,
    "lang": "Chinese",
    "model": "1.7B"
}

submit_resp = requests.post(f"{BASE_URL}/api/clone", json=payload)
if submit_resp.status_code != 200:
    print(f"❌ 提交失败: {submit_resp.json()}")
    exit(1)

result = submit_resp.json()
task_id = result["task_id"]
print(f"✅ 任务已提交！")
print(f"   Task ID: {task_id}")
print(f"   队列位置: {result.get('queue_position', 'N/A')}")

# ============ 4. 轮询查询任务状态 ============
print("\n⏳ 等待任务完成...")
query_url = f"{BASE_URL}/api/task/{USER_ID}/{task_id}"
max_wait = 180  # 最多等待3分钟
start_time = time.time()

while True:
    response = requests.get(query_url)
    task_info = response.json()
    status = task_info["status"]
    
    if status == "completed":
        print("✅ 任务完成！")
        audio_url = task_info["audio_url"]
        if not audio_url.startswith("http"):
            audio_url = BASE_URL + audio_url
        
        print(f"🎵 音频信息:")
        print(f"   下载地址: {audio_url}")
        print(f"   文件名: {task_info['audio_file']}")
        print(f"   采样率: {task_info['sample_rate']}Hz")
        
        # ============ 5. 下载音频文件 ============
        print("\n📥 正在下载音频...")
        audio_resp = requests.get(audio_url)
        if audio_resp.status_code == 200:
            output_file = f"output_{task_id}.wav"
            with open(output_file, "wb") as f:
                f.write(audio_resp.content)
            print(f"💾 音频已保存: {output_file}")
        break
    
    elif status == "failed":
        print(f"❌ 任务失败: {task_info.get('error', '未知错误')}")
        break
    
    elif status in ["pending", "processing"]:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            print(f"⏰ 超时：任务执行时间超过{max_wait}秒")
            break
        
        status_icon = "⏳" if status == "pending" else "🔄"
        print(f"{status_icon} 状态: {status} (已等待 {int(elapsed)}秒)")
        time.sleep(3)  # 每3秒查询一次
    
    else:
        print(f"❓ 未知状态: {status}")
        break

# ============ 6. 查询用户所有任务（可选）============
print("\n📋 查询用户所有任务...")
tasks_resp = requests.get(f"{BASE_URL}/api/tasks/{USER_ID}")
tasks_info = tasks_resp.json()
print(f"总任务数: {tasks_info['total']}")
for task in tasks_info['tasks'][:3]:
    print(f"  - {task['task_id']}: {task['status']}")
```

---

### JavaScript完整调用示例

```javascript
const BASE_URL = 'http://localhost:8001';
const USER_ID = 'user001';

// 将File对象转为base64
async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // 移除 "data:audio/wav;base64," 前缀
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// 主函数
async function cloneVoice(refAudioFile, targetText) {
  try {
    // 1. 转换音频为base64
    console.log('📂 读取参考音频...');
    const refAudioB64 = await fileToBase64(refAudioFile);
    
    // 2. ASR识别参考文本（可选）
    console.log('🔊 识别参考音频文本...');
    const formData = new FormData();
    formData.append('file', refAudioFile);
    
    let refText;
    const asrResp = await fetch(`${BASE_URL}/api/asr`, {
      method: 'POST',
      body: formData
    });
    
    if (asrResp.ok) {
      const asrData = await asrResp.json();
      refText = asrData.text;
      console.log(`✅ 识别结果: ${refText}`);
    } else {
      refText = prompt('ASR不可用，请输入参考音频的文本内容:');
    }
    
    // 3. 提交语音克隆任务
    console.log('\n📤 提交语音克隆任务...');
    const payload = {
      user_id: USER_ID,
      text: targetText,
      ref_audio_b64: refAudioB64,
      ref_text: refText,
      speed: 1.0,
      lang: 'Chinese',
      model: '1.7B'
    };
    
    const submitResp = await fetch(`${BASE_URL}/api/clone`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!submitResp.ok) {
      throw new Error(`提交失败: ${await submitResp.text()}`);
    }
    
    const result = await submitResp.json();
    const taskId = result.task_id;
    console.log(`✅ 任务已提交！Task ID: ${taskId}`);
    console.log(`   队列位置: ${result.queue_position}`);
    
    // 4. 轮询查询任务状态
    console.log('\n⏳ 等待任务完成...');
    const queryUrl = `${BASE_URL}/api/task/${USER_ID}/${taskId}`;
    const maxWait = 180; // 最多等待3分钟
    const startTime = Date.now();
    
    while (true) {
      const response = await fetch(queryUrl);
      const taskInfo = await response.json();
      const status = taskInfo.status;
      
      if (status === 'completed') {
        console.log('✅ 任务完成！');
        console.log(`🎵 音频下载地址: ${taskInfo.audio_url}`);
        
        // 触发浏览器下载
        const a = document.createElement('a');
        a.href = taskInfo.audio_url;
        a.download = taskInfo.audio_file;
        a.click();
        
        return taskInfo;
      } else if (status === 'failed') {
        throw new Error(`任务失败: ${taskInfo.error}`);
      } else if (status === 'pending' || status === 'processing') {
        const elapsed = (Date.now() - startTime) / 1000;
        if (elapsed > maxWait) {
          throw new Error(`任务超时（>${maxWait}秒）`);
        }
        console.log(`${status === 'pending' ? '⏳' : '🔄'} 状态: ${status} (已等待 ${Math.floor(elapsed)}秒)`);
        await new Promise(resolve => setTimeout(resolve, 3000)); // 每3秒查询一次
      } else {
        throw new Error(`未知状态: ${status}`);
      }
    }
    
  } catch (error) {
    console.error('❌ 错误:', error);
    throw error;
  }
}

// 使用示例（前端调用）
const refAudioInput = document.getElementById('refAudio');
const targetText = document.getElementById('targetText').value;

cloneVoice(refAudioInput.files[0], targetText)
  .then(result => console.log('🎉 成功!', result))
  .catch(error => console.error('💥 失败:', error));
```

---

## 错误处理

### 常见错误码

| HTTP状态码 | 错误原因 | 解决方案 |
|-----------|---------|---------|
| 400 | 参数错误 | 检查请求参数格式、类型、范围 |
| 404 | 任务不存在 | 确认`user_id`和`task_id`正确 |
| 503 | 服务不可用 | 检查ASR/模型是否加载成功 |

### 错误响应格式

```json
{
  "detail": "具体的错误信息"
}
```

### Python错误处理示例

```python
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # 自动抛出HTTP错误
    result = response.json()
    
except requests.exceptions.HTTPError as e:
    print(f"HTTP错误 {e.response.status_code}: {e.response.json()['detail']}")
except requests.exceptions.Timeout:
    print("请求超时")
except requests.exceptions.ConnectionError:
    print("连接失败，请检查服务是否启动")
except Exception as e:
    print(f"未知错误: {str(e)}")
```

---

## 注意事项

### ⚠️ 重要限制

1. **参考音频时长**
   - 必须在 **3-15秒** 之间（官方支持3秒快速克隆）
   - 少于3秒：不支持
   - 超过15秒：处理超时
   - 推荐 **5-10秒**：质量最佳

2. **队列处理机制**
   - ⚠️ 任务**按提交顺序依次执行**，一个一个处理
   - 高峰期可能需要较长等待时间
   - 建议先调用 `/api/queue/status` 查看队列负载

3. **Base64编码格式**
   - ✅ 正确：纯base64字符串（`SUQzAwA...`）
   - ❌ 错误：包含DataURL前缀（`data:audio/wav;base64,SUQzAwA...`）
   - 前端获取时需要 `.split(',')[1]` 移除前缀

4. **轮询间隔**
   - 建议 **2-5秒** 查询一次
   - 避免频繁请求（<1秒）造成服务器压力

5. **任务超时**
   - 短文本（<50字）：约10-30秒
   - 长文本（>200字）：可能需要60-180秒
   - 建议设置合理的超时时间（如180秒）

### 📝 最佳实践

1. **参考音频准备**
   ```
   ✅ 清晰无噪音
   ✅ 单说话人
   ✅ 音量适中
   ✅ 采样率 ≥ 16000Hz
   ✅ ref_text 与音频内容精确匹配
   ```

2. **任务提交流程**
   ```
   1. 检查 /api/queue/status（队列负载）
   2. 准备参考音频（3-15秒）
   3. ASR识别或手动输入 ref_text
   4. 提交任务获取 task_id
   5. 轮询查询状态（3秒/次）
   6. 下载完成的音频文件
   ```

3. **错误重试策略**
   ```python
   max_retries = 3
   for attempt in range(max_retries):
       try:
           response = requests.post(url, json=payload, timeout=30)
           response.raise_for_status()
           break
       except Exception as e:
           if attempt == max_retries - 1:
               raise
           print(f"重试 {attempt+1}/{max_retries}...")
           time.sleep(2)
   ```

### 🔧 性能优化建议

1. **模型缓存**
   - 首次使用某个模型时会有加载时间（10-30秒）
   - 后续使用同一模型会自动缓存，速度更快

2. **音频格式**
   - 推荐使用 WAV 格式（无损）
   - 如果使用 MP3，确保比特率 ≥ 128kbps

3. **文本长度**
   - 单次合成不超过 **500字**
   - 超长文本建议分段合成后拼接

### 🔐 安全建议

1. **用户标识**
   - 使用唯一的用户ID（如UUID、邮箱hash）
   - 避免使用敏感信息（如明文邮箱、手机号）

2. **任务管理**
   - 定期清理历史任务（使用 `DELETE /api/task`）
   - 避免音频文件堆积占用磁盘空间

3. **生产环境部署**
   - 使用 Nginx 反向代理
   - 启用 HTTPS 加密传输
   - 设置请求频率限制（Rate Limiting）

---

## 附录

### A. 任务状态完整定义

| 状态 | 英文 | 说明 | 下一步操作 |
|-----|------|------|-----------|
| 等待中 | pending | 任务已提交，在队列中等待 | 继续轮询 |
| 处理中 | processing | 任务正在执行 | 继续轮询 |
| 已完成 | completed | 任务成功完成 | 下载音频 |
| 失败 | failed | 任务执行失败 | 查看error字段 |

### B. 语言选项完整列表

Qwen3-TTS 支持 **10种主要语言** 以及自动语言检测：

| 语言代码 | 语言名称 | 说明 |
|---------|---------|------|
| `Auto` | 自动检测 | 推荐：根据文本内容自动识别语言 |
| `Chinese` | 中文 | 支持普通话及多种方言（北京话、四川话等） |
| `English` | 英语 | 支持美式英语、英式英语 |
| `Japanese` | 日语 | 日本语 |
| `Korean` | 韩语 | 한국어 |
| `German` | 德语 | Deutsch |
| `French` | 法语 | Français |
| `Russian` | 俄语 | Русский |
| `Portuguese` | 葡萄牙语 | Português |
| `Spanish` | 西班牙语 | Español |
| `Italian` | 意大利语 | Italiano |

**使用建议：**
- ✅ **推荐使用 `Auto`**：模型会根据文本内容自动选择最合适的语言
- ✅ **明确已知语言时**：直接指定语言代码可获得更好的效果
- ✅ **多语言混合文本**：使用 `Auto` 让模型自动处理
- ❌ **避免语言不匹配**：如文本是中文但设置为 `English` 会导致发音不准确

### C. 模型对比

| 模型 | 参数量 | 质量 | 速度 | 显存占用 | 推荐场景 |
|-----|-------|------|------|---------|---------|
| 0.6B | 6亿 | ⭐⭐⭐ | 快 | ~2GB | 快速测试、批量生成 |
| 1.7B | 17亿 | ⭐⭐⭐⭐⭐ | 较慢 | ~4GB | 高质量生产环境 |

### D. 常见问题（FAQ）

**Q1: 为什么任务一直是 pending 状态？**
- 检查队列状态 `/api/queue/status`
- 确认 `worker_running` 是否为 `true`
- 查看服务器日志是否有错误

**Q2: 如何加快合成速度？**
- 使用 `0.6B` 模型
- 减少文本长度
- 避免高峰期提交

**Q3: 音频质量不满意怎么办？**
- 使用 `1.7B` 模型
- 提供高质量参考音频（清晰、无噪音、24000Hz采样率）
- 确保 `ref_text` 与音频内容精确匹配

**Q4: 支持批量任务吗？**
- 支持，可连续提交多个任务
- 任务会依次排队处理
- 建议使用脚本批量提交并管理

---

## 联系与支持

- **项目地址**：[Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- **文档更新日期**：2026-03-06
- **API版本**：v1.1

---

**文档结束** 🎉
