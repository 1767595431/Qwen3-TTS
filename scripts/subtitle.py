"""
字幕生成模块：基于 WhisperX 对 TTS 生成的音频做识别 + 强制对齐，
输出带时间戳的 SRT 和 JSON 字幕文件，支持 sentence / word 两种粒度。
"""
import json
import os
import re
from typing import Dict, List

import torch
import whisperx

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_WHITESPACE_RE = re.compile(r"\s+")
_STRONG_END_PUNCT = "。！？!?；;"
_CJK_SOFT_SPLIT_PUNCT = "，,、：:"
_LATIN_SOFT_SPLIT_PUNCT = ",;:"

_WHISPERX_MODEL = None
_WHISPERX_ALIGN_MODELS: Dict[str, tuple] = {}
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHISPERX_MODEL_DIR = os.getenv(
    "QWEN_TTS_SUBTITLE_MODEL_DIR",
    os.path.join(APP_DIR, "models", "WhisperX"),
)

LANG_MAP = {
    "Auto": None,
    "Chinese": "zh",
    "English": "en",
    "Japanese": "ja",
    "Korean": "ko",
    "German": "de",
    "French": "fr",
    "Russian": "ru",
    "Portuguese": "pt",
    "Spanish": "es",
    "Italian": "it",
}


def _subtitle_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _subtitle_compute_type(device: str) -> str:
    if device == "cuda":
        return os.getenv("QWEN_TTS_SUBTITLE_COMPUTE_TYPE", "float16")
    return os.getenv("QWEN_TTS_SUBTITLE_COMPUTE_TYPE_CPU", "int8")


def _subtitle_model_name() -> str:
    return os.getenv("QWEN_TTS_SUBTITLE_MODEL", "large-v3")


def _subtitle_batch_size() -> int:
    try:
        return max(1, int(os.getenv("QWEN_TTS_SUBTITLE_BATCH_SIZE", "16")))
    except ValueError:
        return 16


def _subtitle_preload_align_language() -> str | None:
    raw = os.getenv("QWEN_TTS_SUBTITLE_PRELOAD_ALIGN_LANGUAGE", "zh").strip().lower()
    return raw or None


def _ensure_whisperx_model_dir() -> str:
    os.makedirs(WHISPERX_MODEL_DIR, exist_ok=True)
    return WHISPERX_MODEL_DIR


def get_whisperx_cache_status() -> Dict[str, object]:
    """Return local WhisperX cache directory status for startup checks."""
    model_dir = _ensure_whisperx_model_dir()
    entries = []
    try:
        entries = [entry.name for entry in os.scandir(model_dir)]
    except OSError:
        entries = []

    return {
        "model_dir": model_dir,
        "model_name": _subtitle_model_name(),
        "has_cache": len(entries) > 0,
        "entries": sorted(entries)[:20],
    }


def preload_whisperx_assets() -> Dict[str, object]:
    """
    Preload WhisperX main model and optional default alignment model.
    Raises on failure so callers can report it during startup.
    """
    model = _get_whisperx_model()
    align_language = _subtitle_preload_align_language()
    align_loaded = False
    if align_language:
        _get_align_model(align_language)
        align_loaded = True

    return {
        "model_name": _subtitle_model_name(),
        "model_dir": _ensure_whisperx_model_dir(),
        "device": _subtitle_device(),
        "compute_type": _subtitle_compute_type(_subtitle_device()),
        "align_language": align_language,
        "align_loaded": align_loaded,
        "model_loaded": model is not None,
    }


def _get_whisperx_model():
    global _WHISPERX_MODEL
    if _WHISPERX_MODEL is not None:
        return _WHISPERX_MODEL

    device = _subtitle_device()
    compute_type = _subtitle_compute_type(device)
    model_dir = _ensure_whisperx_model_dir()
    _WHISPERX_MODEL = whisperx.load_model(
        _subtitle_model_name(),
        device,
        compute_type=compute_type,
        download_root=model_dir,
    )
    return _WHISPERX_MODEL


def _get_align_model(language_code: str):
    if language_code in _WHISPERX_ALIGN_MODELS:
        return _WHISPERX_ALIGN_MODELS[language_code]

    device = _subtitle_device()
    model_dir = _ensure_whisperx_model_dir()
    model_a, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device,
        model_dir=model_dir,
    )
    _WHISPERX_ALIGN_MODELS[language_code] = (model_a, metadata)
    return model_a, metadata


def generate_subtitles(audio_path: str, language: str = "Auto") -> Dict:
    """
    对音频做 WhisperX 识别 + forced alignment，返回带时间戳的段落和词列表。

    Returns:
        {
            "segments": [
                {
                    "start": 0.0, "end": 2.5, "text": "...",
                    "words": [{"start": 0.0, "end": 0.4, "word": "..."}]
                }
            ],
            "language": "zh"
        }
    """
    device = _subtitle_device()
    lang_code = LANG_MAP.get(language)

    model = _get_whisperx_model()
    audio = whisperx.load_audio(audio_path)

    transcribe_kwargs = {"batch_size": _subtitle_batch_size()}
    if lang_code:
        transcribe_kwargs["language"] = lang_code

    result = model.transcribe(audio, **transcribe_kwargs)
    detected_lang = result.get("language", lang_code or "en")

    try:
        model_a, metadata = _get_align_model(detected_lang)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as e:
        print(f"[字幕] forced alignment 失败 ({detected_lang}): {e}，使用原始时间戳")

    segments: List[Dict] = []
    for seg in result.get("segments", []):
        segment = {
            "start": round(seg.get("start", 0.0), 3),
            "end": round(seg.get("end", 0.0), 3),
            "text": seg.get("text", "").strip(),
            "words": [],
        }
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                segment["words"].append({
                    "start": round(float(w["start"]), 3),
                    "end": round(float(w["end"]), 3),
                    "word": w.get("word", "").strip(),
                })
        if segment["text"]:
            segments.append(segment)

    return {"segments": segments, "language": detected_lang}


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _clean_subtitle_text(text: str) -> str:
    """Normalize spacing while keeping Chinese punctuation compact."""
    text = text.strip()
    if not text:
        return ""
    text = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", text)
    text = re.sub(r"([(\[{\"'“‘])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _visible_text_len(text: str) -> int:
    return len(_WHITESPACE_RE.sub("", text))


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _build_segment_from_words(words: List[Dict]) -> Dict | None:
    if not words:
        return None
    start = float(words[0].get("start", 0.0))
    end = float(words[-1].get("end", start))
    text = _clean_subtitle_text("".join(str(word.get("word", "")) for word in words))
    if not text:
        return None
    return {
        "start": round(start, 3),
        "end": round(max(end, start), 3),
        "text": text,
        "words": words,
    }


def _split_plain_text(text: str, prefer_cjk: bool) -> List[str]:
    text = _clean_subtitle_text(text)
    if not text:
        return []

    ideal_chars = 22 if prefer_cjk else 44
    hard_max = 45 if prefer_cjk else 80
    min_soft_chars = 10 if prefer_cjk else 24
    all_split_punct = _STRONG_END_PUNCT + (_CJK_SOFT_SPLIT_PUNCT if prefer_cjk else _LATIN_SOFT_SPLIT_PUNCT)

    chunks: List[str] = []
    current = ""
    over_ideal = False
    for char in text:
        current += char
        visible_len = _visible_text_len(current)

        if char in _STRONG_END_PUNCT:
            chunks.append(_clean_subtitle_text(current))
            current = ""
            over_ideal = False
            continue

        if visible_len >= ideal_chars:
            over_ideal = True

        if over_ideal and char in all_split_punct and visible_len >= min_soft_chars:
            chunks.append(_clean_subtitle_text(current))
            current = ""
            over_ideal = False
            continue

        if visible_len >= hard_max:
            chunks.append(_clean_subtitle_text(current))
            current = ""
            over_ideal = False

    if current.strip():
        chunks.append(_clean_subtitle_text(current))
    return [c for c in chunks if c]


def _split_segment_by_text(seg: Dict) -> List[Dict]:
    text = seg.get("text", "").strip()
    if not text:
        return []

    prefer_cjk = _contains_cjk(text)
    chunks = _split_plain_text(text, prefer_cjk=prefer_cjk)
    if len(chunks) <= 1:
        return [{
            "start": round(float(seg.get("start", 0.0)), 3),
            "end": round(float(seg.get("end", seg.get("start", 0.0))), 3),
            "text": _clean_subtitle_text(text),
            "words": seg.get("words", []),
        }]

    total_units = sum(max(1, _visible_text_len(chunk)) for chunk in chunks)
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    duration = max(end - start, 0.0)

    out: List[Dict] = []
    cursor = start
    consumed_units = 0
    for index, chunk in enumerate(chunks):
        units = max(1, _visible_text_len(chunk))
        consumed_units += units
        if index == len(chunks) - 1 or total_units <= 0:
            chunk_end = end
        else:
            chunk_end = start + duration * (consumed_units / total_units)
        out.append({
            "start": round(cursor, 3),
            "end": round(max(chunk_end, cursor), 3),
            "text": chunk,
            "words": [],
        })
        cursor = chunk_end
    return out


def _split_segment_by_words(seg: Dict) -> List[Dict]:
    words = [
        word for word in seg.get("words", [])
        if "start" in word and "end" in word and str(word.get("word", "")).strip()
    ]
    if not words:
        return _split_segment_by_text(seg)

    sample_text = seg.get("text", "") or "".join(str(word.get("word", "")) for word in words)
    prefer_cjk = _contains_cjk(sample_text)
    max_chars = 24 if prefer_cjk else 48
    min_soft_chars = 10 if prefer_cjk else 24
    max_duration = 6.0 if prefer_cjk else 8.0
    soft_split_punct = _CJK_SOFT_SPLIT_PUNCT if prefer_cjk else _LATIN_SOFT_SPLIT_PUNCT

    out: List[Dict] = []
    current_words: List[Dict] = []
    visible_len = 0

    for word in words:
        token = str(word.get("word", ""))
        current_words.append(word)
        visible_len += max(1, _visible_text_len(token))
        current_duration = float(current_words[-1]["end"]) - float(current_words[0]["start"])
        token_tail = token.strip()
        should_flush = (
            any(char in token_tail for char in _STRONG_END_PUNCT)
            or (
                any(char in token_tail for char in soft_split_punct)
                and visible_len >= min_soft_chars
            )
            or visible_len >= max_chars
            or (current_duration >= max_duration and visible_len >= min_soft_chars)
        )
        if should_flush:
            chunk = _build_segment_from_words(current_words)
            if chunk:
                out.append(chunk)
            current_words = []
            visible_len = 0

    if current_words:
        chunk = _build_segment_from_words(current_words)
        if chunk:
            out.append(chunk)

    return out or _split_segment_by_text(seg)


def _prepare_sentence_segments(segments: List[Dict]) -> List[Dict]:
    prepared: List[Dict] = []
    for seg in segments:
        prepared.extend(_split_segment_by_words(seg))
    return prepared


def build_srt_content(
    segments: List[Dict],
    granularity: str = "sentence",
) -> str:
    """Build SRT content from subtitle segments."""
    lines: List[str] = []
    index = 1

    if granularity == "word":
        for seg in segments:
            for w in seg.get("words", []):
                start_str = _format_srt_time(w["start"])
                end_str = _format_srt_time(w["end"])
                lines.append(str(index))
                lines.append(f"{start_str} --> {end_str}")
                lines.append(w["word"])
                lines.append("")
                index += 1
    else:
        for seg in _prepare_sentence_segments(segments):
            start_str = _format_srt_time(seg["start"])
            end_str = _format_srt_time(seg["end"])
            lines.append(str(index))
            lines.append(f"{start_str} --> {end_str}")
            lines.append(seg["text"])
            lines.append("")
            index += 1

    return "\n".join(lines)


def build_subtitle_json_payload(subtitle_data: Dict) -> Dict:
    """Build the JSON payload returned by the subtitle endpoint."""
    prepared_segments = _prepare_sentence_segments(subtitle_data.get("segments", []))
    out = {
        "language": subtitle_data.get("language", ""),
        "segments": [],
    }
    for i, seg in enumerate(prepared_segments, start=1):
        out["segments"].append({
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "words": seg.get("words", []),
        })
    return out


def write_srt(
    segments: List[Dict],
    output_path: str,
    granularity: str = "sentence",
) -> str:
    """
    Write SRT subtitle file.

    Args:
        segments: list from generate_subtitles()["segments"]
        output_path: .srt file path
        granularity: "sentence" (one subtitle per segment) or "word" (one per word)

    Returns:
        The output file path.
    """
    content = build_srt_content(segments, granularity=granularity)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


def write_subtitle_json(
    subtitle_data: Dict,
    output_path: str,
) -> str:
    """
    Write JSON subtitle file containing both sentence-level and word-level data.

    Args:
        subtitle_data: full dict from generate_subtitles()
        output_path: .json file path

    Returns:
        The output file path.
    """
    out = build_subtitle_json_payload(subtitle_data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return output_path


def _align_original_text_to_asr(
    original_text: str,
    asr_segments: List[Dict],
) -> List[Dict]:
    """
    Map original text chunks onto ASR-detected time regions.

    Strategy:
      1. Merge all ASR segments into one flat character-level timeline
         (each ASR segment contributes its text length and time span).
      2. Split original text into display-ready chunks.
      3. Walk through the ASR timeline character-by-character, assigning
         each original chunk a start/end from the ASR position that best
         matches how far through the total text we've read.
    This keeps timestamps anchored to actual speech rather than a naive
    chars-per-second estimate.
    """
    clean_text = _clean_subtitle_text(original_text)
    prefer_cjk = _contains_cjk(clean_text)
    chunks = _split_plain_text(clean_text, prefer_cjk=prefer_cjk)
    if not chunks:
        chunks = [clean_text] if clean_text else [""]

    asr_total_chars = sum(
        max(1, _visible_text_len(seg.get("text", "")))
        for seg in asr_segments
    )
    orig_total_chars = sum(max(1, _visible_text_len(c)) for c in chunks)

    timeline: List[dict] = []
    for seg in asr_segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        seg_chars = max(1, _visible_text_len(seg.get("text", "")))
        dur = max(e - s, 0.0)
        for ci in range(seg_chars):
            frac = ci / seg_chars
            timeline.append({
                "t": s + dur * frac,
                "t_end": s + dur * ((ci + 1) / seg_chars),
            })
    if not timeline:
        s = float(asr_segments[0].get("start", 0.0))
        e = float(asr_segments[-1].get("end", s))
        timeline = [{"t": s, "t_end": e}]

    total_end = float(asr_segments[-1].get("end", 0.0))

    out: List[Dict] = []
    consumed_orig = 0
    for i, chunk in enumerate(chunks):
        chunk_chars = max(1, _visible_text_len(chunk))
        frac_start = consumed_orig / orig_total_chars
        consumed_orig += chunk_chars
        frac_end = consumed_orig / orig_total_chars

        idx_start = min(int(frac_start * asr_total_chars), len(timeline) - 1)
        idx_end = min(int(frac_end * asr_total_chars), len(timeline) - 1)

        t_start = timeline[idx_start]["t"]
        t_end = timeline[idx_end]["t_end"] if idx_end < len(timeline) else total_end
        if i == len(chunks) - 1:
            t_end = total_end

        out.append({
            "start": round(t_start, 3),
            "end": round(max(t_end, t_start + 0.1), 3),
            "text": chunk,
            "words": [],
        })

    return out


def generate_subtitles_with_original_text(
    audio_path: str,
    original_text: str,
    language: str = "Auto",
) -> Dict:
    """
    Use WhisperX for timestamps only; replace recognized text with the original.
    WhisperX provides the speech timeline, original text provides the content.
    """
    device = _subtitle_device()
    lang_code = LANG_MAP.get(language)

    model = _get_whisperx_model()
    audio = whisperx.load_audio(audio_path)

    transcribe_kwargs = {"batch_size": _subtitle_batch_size()}
    if lang_code:
        transcribe_kwargs["language"] = lang_code

    result = model.transcribe(audio, **transcribe_kwargs)
    detected_lang = result.get("language", lang_code or "en")

    try:
        model_a, metadata = _get_align_model(detected_lang)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as e:
        print(f"[字幕] forced alignment 失败 ({detected_lang}): {e}，使用原始时间戳")

    asr_segments = result.get("segments", [])
    if not asr_segments:
        audio_duration = len(audio) / 16000.0
        return {
            "segments": [{
                "start": 0.0,
                "end": round(audio_duration, 3),
                "text": original_text.strip(),
                "words": [],
            }],
            "language": detected_lang,
        }

    segments = _align_original_text_to_asr(original_text, asr_segments)

    return {"segments": segments, "language": detected_lang}


def generate_subtitle_inline(
    audio_path: str,
    language: str = "Auto",
    original_text: str = "",
) -> Dict:
    """
    Generate subtitle data and return it inline (no files written).
    If original_text is provided, use it instead of WhisperX recognition.
    """
    if original_text.strip():
        return generate_subtitles_with_original_text(audio_path, original_text, language)
    return generate_subtitles(audio_path, language)


def recognize_text(audio_path: str, language: str = "Auto") -> str:
    """
    Use WhisperX to transcribe audio and return plain text (no timestamps).
    Reuses the same cached model as subtitle generation.
    """
    lang_code = LANG_MAP.get(language)
    model = _get_whisperx_model()
    audio = whisperx.load_audio(audio_path)

    transcribe_kwargs = {"batch_size": _subtitle_batch_size()}
    if lang_code:
        transcribe_kwargs["language"] = lang_code

    result = model.transcribe(audio, **transcribe_kwargs)
    segments = result.get("segments", [])
    return " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip())
