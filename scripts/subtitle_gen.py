#!/usr/bin/env python3
"""
subtitle_gen.py — Auto-generate Spanish subtitles for videos.

Workflow:
  1. Analyze video frames to detect regions with existing text/subtitles
  2. OCR-scan for burned-in subtitles; if found, offer to sync timings
  3. Transcribe audio using Whisper (if no OCR sync)
  4. Translate to target language if needed (via deep-translator)
  5. Generate ASS subtitles positioned in the detected free region
  6. Burn subtitles into the video using ffmpeg
"""

import argparse
import asyncio
import difflib
import glob
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


# ---------------------------------------------------------------------------
# Video info helpers
# ---------------------------------------------------------------------------

def get_video_info(video_path, ffprobe_path):
    """Get video dimensions, duration, and any embedded subtitle streams."""
    cmd = [
        ffprobe_path, "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,width,height",
        "-show_entries", "format=duration",
        "-of", "json", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    width = height = 0
    duration = float(info["format"]["duration"])
    subtitle_streams = []

    for s in info.get("streams", []):
        if s["codec_type"] == "video" and width == 0:
            width, height = int(s["width"]), int(s["height"])
        elif s["codec_type"] == "subtitle":
            subtitle_streams.append(s)

    return width, height, duration, subtitle_streams


# ---------------------------------------------------------------------------
# Frame extraction & region analysis
# ---------------------------------------------------------------------------

def extract_frames(video_path, num_frames, ffmpeg_path, ffprobe_path):
    """Extract evenly-spaced frames as PNG files. Returns (paths, w, h)."""
    width, height, duration, _ = get_video_info(video_path, ffprobe_path)
    tmpdir = tempfile.mkdtemp(prefix="sub_analysis_")
    timestamps = [duration * i / (num_frames + 1) for i in range(1, num_frames + 1)]

    paths = []
    for i, ts in enumerate(timestamps):
        out = os.path.join(tmpdir, f"frame_{i:03d}.png")
        subprocess.run(
            [ffmpeg_path, "-v", "error", "-ss", str(ts), "-i", video_path,
             "-vframes", "1", "-y", out],
            capture_output=True, check=True,
        )
        if os.path.exists(out):
            paths.append(out)

    return paths, width, height, tmpdir


def analyze_text_regions(frame_paths, width, height):
    """
    Score horizontal bands for text presence.
    Combined score = mean + 2*std  →  lower is freer.
    """
    num_bands = 5
    band_h = height // num_bands
    scores = np.zeros((len(frame_paths), num_bands))

    for fi, fp in enumerate(frame_paths):
        img = Image.open(fp).convert("L")
        edges = np.array(img.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        for b in range(num_bands):
            y0 = b * band_h
            y1 = min((b + 1) * band_h, height)
            scores[fi, b] = edges[y0:y1, :].mean()

    mean = scores.mean(axis=0)
    std = scores.std(axis=0)
    combined = mean + 2 * std
    return combined, mean, std


def find_subtitle_placement(video_path, ffmpeg_path, ffprobe_path, log=print):
    """
    Analyze frame regions and decide where to place new subtitles.
    Returns (new_sub_position, margin, existing_sub_region, width, height).
    """
    frames, w, h, tmpdir = extract_frames(
        video_path, num_frames=10,
        ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path,
    )

    if not frames:
        log("  (frame extraction failed — defaulting to top)")
        return "top", max(10, int(h * 0.03)), "bottom", w, h

    combined, means, stds = analyze_text_regions(frames, w, h)

    labels = ["top", "upper-mid", "center", "lower-mid", "bottom"]
    log("  Region scores (lower = more free):")
    for i, (label, score) in enumerate(zip(labels, combined)):
        tag = " <- candidate" if i in (0, 4) else ""
        log(f"    {label:>10}: {score:6.1f}{tag}")

    top_score = combined[0]
    bot_score = combined[4]

    if top_score <= bot_score:
        new_position = "top"
        existing_region = "bottom"
    else:
        new_position = "bottom"
        existing_region = "top"

    margin = max(10, int(h * 0.03))

    # cleanup
    for f in frames:
        os.unlink(f)
    os.rmdir(tmpdir)

    log(f"  -> Best placement for new subs: {new_position} (margin {margin}px)")
    log(f"  -> Existing content detected at: {existing_region}")
    return new_position, margin, existing_region, w, h


# ---------------------------------------------------------------------------
# OCR subtitle detection
# ---------------------------------------------------------------------------

def extract_frames_dense(video_path, fps, region, ffmpeg_path, ffprobe_path):
    """
    Extract frames at given fps, cropped to the subtitle region.
    Returns (frame_paths, tmpdir, frame_interval).
    """
    w, h, duration, _ = get_video_info(video_path, ffprobe_path)
    tmpdir = tempfile.mkdtemp(prefix="ocr_frames_")

    # Crop to ~22% of frame height at the detected region
    crop_h = int(h * 0.22)
    if region == "bottom":
        crop_y = h - crop_h
    else:
        crop_y = 0

    cmd = [
        ffmpeg_path, "-v", "error", "-i", video_path,
        "-vf", f"fps={fps},crop={w}:{crop_h}:0:{crop_y}",
        "-q:v", "2",
        os.path.join(tmpdir, "frame_%05d.png"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    frames = sorted(glob.glob(os.path.join(tmpdir, "frame_*.png")))
    return frames, tmpdir, 1.0 / fps


_TESSERACT_LANG_MAP = {
    "en": "eng", "es": "spa", "fr": "fra", "de": "deu", "pt": "por",
    "it": "ita", "zh": "chi_sim", "ja": "jpn", "ko": "kor", "ar": "ara",
    "ru": "rus", "hi": "hin", "nl": "nld", "pl": "pol", "tr": "tur",
    "vi": "vie", "th": "tha", "sv": "swe", "da": "dan", "fi": "fin",
}


def get_tesseract_lang(iso_code):
    """Map ISO 639-1 code to Tesseract language code, with fallback to eng."""
    return _TESSERACT_LANG_MAP.get(iso_code, "eng")


def ocr_frame(image_path, lang="eng"):
    """Run Tesseract OCR on a single frame. Returns extracted text.

    Args:
        image_path: Path to the frame image.
        lang: Tesseract language code (e.g. "eng", "spa"). Falls back to
              "eng" if the specified language pack is not installed.
    """
    import pytesseract

    img = Image.open(image_path)
    # Preprocess: convert to grayscale, increase contrast, scale up
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img, cutoff=1)
    # Scale up 2x for better OCR on small text
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)

    try:
        text = pytesseract.image_to_string(
            img, lang=lang,
            config="--psm 6",  # Assume a single uniform block of text
        )
    except pytesseract.TesseractError:
        # Language pack not installed — fall back to English
        text = pytesseract.image_to_string(
            img, lang="eng",
            config="--psm 6",
        )
    return text.strip()


def group_ocr_results(results, frame_interval, min_duration=0.4):
    """
    Group consecutive frames with similar text into timed segments.
    Uses SequenceMatcher to handle minor OCR variations between frames.
    """
    if not results:
        return []

    segments = []
    current_text = results[0][1]
    start_time = results[0][0]

    for timestamp, text in results[1:]:
        # Check similarity between current and new text
        if current_text and text:
            similarity = difflib.SequenceMatcher(
                None, current_text.lower(), text.lower()
            ).ratio()
            same = similarity > 0.65
        else:
            same = (not current_text and not text)

        if not same:
            # Text changed — save previous segment if it had content
            if current_text.strip():
                segments.append({
                    "start": start_time,
                    "end": timestamp,
                    "text": current_text.strip(),
                })
            current_text = text
            start_time = timestamp
        else:
            # Keep the longer/better version
            if len(text) > len(current_text):
                current_text = text

    # Last segment
    if current_text.strip():
        segments.append({
            "start": start_time,
            "end": results[-1][0] + frame_interval,
            "text": current_text.strip(),
        })

    # Filter out very short segments (OCR noise)
    segments = [s for s in segments if s["end"] - s["start"] >= min_duration]
    return segments


def detect_burned_in_subs(video_path, region, ffmpeg_path, ffprobe_path,
                          log=print, ocr_lang="eng"):
    """
    OCR-scan the given region for burned-in subtitles.
    Returns list of {start, end, text} segments, or empty list.
    """
    log(f"  Extracting frames from '{region}' region at 2 fps...")
    frames, tmpdir, interval = extract_frames_dense(
        video_path, fps=2, region=region,
        ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path,
    )

    if not frames:
        return []

    log(f"  Running OCR on {len(frames)} frames (lang={ocr_lang})...")
    results = []
    for i, fp in enumerate(frames):
        timestamp = i * interval
        text = ocr_frame(fp, lang=ocr_lang)
        results.append((timestamp, text))
        # Progress indicator every 20 frames
        if (i + 1) % 20 == 0:
            log(f"    ... {i + 1}/{len(frames)} frames processed")

    # Cleanup temp frames
    for f in frames:
        os.unlink(f)
    os.rmdir(tmpdir)

    segments = group_ocr_results(results, interval)
    return segments


def assess_ocr_quality(segments):
    """
    Evaluate whether OCR'd segments contain readable subtitle text.

    For each segment we check:
      - alpha_ratio: fraction of characters that are letters or spaces
      - real_words:  words with >= 3 alphabetic characters
      - length:      total stripped length

    Returns (is_usable, confidence, good_segments).
    good_segments contains only the segments that passed quality checks.
    """
    if not segments:
        return False, 0.0, []

    good = []
    for seg in segments:
        text = seg["text"]
        stripped = text.strip()
        if not stripped:
            continue

        # Ratio of letters + spaces to total characters
        friendly = sum(1 for c in stripped if c.isalpha() or c == " ")
        alpha_ratio = friendly / len(stripped)

        # Count words that are mostly alphabetic (>= 3 alpha chars)
        words = stripped.split()
        real_words = [
            w for w in words
            if sum(1 for c in w if c.isalpha()) >= 3
        ]

        # A readable subtitle line typically has:
        #   > 65% letters/spaces, at least 3 real words, length >= 10
        if alpha_ratio >= 0.65 and len(real_words) >= 3 and len(stripped) >= 10:
            good.append(seg)

    confidence = len(good) / len(segments) if segments else 0.0

    # Consider usable if >= 3 good segments AND they represent >= 25% of total
    is_usable = len(good) >= 3 and confidence >= 0.25
    return is_usable, confidence, good


# ---------------------------------------------------------------------------
# Whisper model helpers
# ---------------------------------------------------------------------------

def get_whisper_device():
    """Auto-detect best device for Whisper inference."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


def load_whisper_model(model_size="small", log=print):
    """Load a Whisper model with GPU auto-detection."""
    from faster_whisper import WhisperModel

    device, compute_type = get_whisper_device()
    log(f"  Loading Whisper model '{model_size}' on {device}...")
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def detect_language_quick(video_path, model_size="small", model=None):
    """Detect audio language without full transcription (~5-10s)."""
    if model is None:
        model = load_whisper_model(model_size)
    _, info = model.transcribe(video_path, language=None)
    return info.language, info.language_probability


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(video_path, language, model_size, model=None, log=print):
    """Transcribe with faster-whisper. Returns (segments, detected_lang)."""
    if model is None:
        model = load_whisper_model(model_size)

    log("  Transcribing audio...")
    segments_iter, info = model.transcribe(
        video_path,
        language=language if language != "auto" else None,
        word_timestamps=True,
    )

    detected = info.language
    log(f"  Detected language: {detected} (p={info.language_probability:.2f})")

    segments = []
    for seg in segments_iter:
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })

    return segments, detected


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

class UnsupportedLanguageError(Exception):
    """Raised when the detected language is not supported by the translator."""
    def __init__(self, lang):
        self.lang = lang
        super().__init__(
            f"Detected language '{lang}' is not supported by the translation service"
        )


_GOOGLE_LANG_MAP = {
    "zh": "zh-CN",
    "he": "iw",
    "jw": "jv",
}


def translate_segments(segments, source_lang, target_lang="es", log=print):
    """Translate subtitle text using deep-translator (Google free API)."""
    from deep_translator import GoogleTranslator

    # Normalize language codes for GoogleTranslator compatibility
    source_lang = _GOOGLE_LANG_MAP.get(source_lang, source_lang)
    target_lang = _GOOGLE_LANG_MAP.get(target_lang, target_lang)

    log(f"  Translating {len(segments)} segments: {source_lang} -> {target_lang}")
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
    except Exception:
        raise UnsupportedLanguageError(source_lang)

    for seg in segments:
        try:
            translated = translator.translate(seg["text"])
            if translated:
                seg["text"] = translated
        except Exception as e:
            log(f"  Warning: Translation failed for segment: {e}")

    return segments


# ---------------------------------------------------------------------------
# TTS audio dubbing (edge-tts)
# ---------------------------------------------------------------------------

_EDGE_TTS_VOICES = {
    "es": {"male": "es-MX-JorgeNeural", "female": "es-MX-DaliaNeural"},
    "en": {"male": "en-US-GuyNeural", "female": "en-US-JennyNeural"},
    "fr": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
    "de": {"male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"},
    "pt": {"male": "pt-BR-AntonioNeural", "female": "pt-BR-FranciscaNeural"},
    "it": {"male": "it-IT-DiegoNeural", "female": "it-IT-ElsaNeural"},
    "ja": {"male": "ja-JP-KeitaNeural", "female": "ja-JP-NanamiNeural"},
    "ko": {"male": "ko-KR-InJoonNeural", "female": "ko-KR-SunHiNeural"},
    "zh": {"male": "zh-CN-YunxiNeural", "female": "zh-CN-XiaoxiaoNeural"},
    "ru": {"male": "ru-RU-DmitryNeural", "female": "ru-RU-SvetlanaNeural"},
    "ar": {"male": "ar-SA-HamedNeural", "female": "ar-SA-ZariyahNeural"},
    "hi": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
}

SAMPLE_RATE = 24000


def synthesize_dubbed_audio(segments, target_lang, video_duration, output_path,
                            ffmpeg_path, voice_gender="male", log=print):
    """Generate a full-length WAV with TTS speech placed at segment timestamps.

    For each segment, edge-tts produces an MP3 which is decoded to raw PCM via
    ffmpeg, then mixed (additive) into a silent buffer spanning the entire video
    duration.  The result is written as a 16-bit WAV file.
    """
    import edge_tts

    lang_voices = _EDGE_TTS_VOICES.get(target_lang)
    if not lang_voices:
        lang_name = _LANG_NAMES.get(target_lang, target_lang)
        raise ValueError(
            f"No TTS voice configured for '{lang_name}' ({target_lang}). "
            f"Supported languages for dubbing: "
            f"{', '.join(sorted(_EDGE_TTS_VOICES))}. "
            f"Try using subtitles-only mode instead."
        )
    voice = lang_voices.get(voice_gender, lang_voices["male"])

    total_samples = int(video_duration * SAMPLE_RATE)
    mixed = np.zeros(total_samples, dtype=np.float32)

    tmpdir = tempfile.mkdtemp(prefix="tts_dub_")
    mp3_paths = []

    try:
        # Sort by start time so each segment's audio window is bounded
        # by the next segment's start, preventing overlapping speech.
        segments = sorted(segments, key=lambda s: s["start"])

        # --- Pass 1: generate all TTS clips and measure durations ---
        seg_info = []  # [(mp3_path, tts_samples, max_samples), ...]
        for i, seg in enumerate(segments):
            text = seg["text"].strip()
            if not text:
                seg_info.append(None)
                continue

            mp3_path = os.path.join(tmpdir, f"seg_{i:04d}.mp3")
            mp3_paths.append(mp3_path)

            comm = edge_tts.Communicate(text, voice)
            asyncio.run(comm.save(mp3_path))

            # Decode to get raw duration
            decode_cmd = [
                ffmpeg_path, "-v", "error",
                "-i", mp3_path,
                "-f", "f32le",
                "-acodec", "pcm_f32le",
                "-ar", str(SAMPLE_RATE),
                "-ac", "1",
                "pipe:1",
            ]
            proc = subprocess.run(decode_cmd, capture_output=True, check=True)
            tts_samples = len(proc.stdout) // 4  # float32 = 4 bytes

            next_start = (segments[i + 1]["start"]
                          if i + 1 < len(segments) else video_duration)
            max_samples = int((next_start - seg["start"]) * SAMPLE_RATE)

            seg_info.append((mp3_path, tts_samples, max_samples))

            if (i + 1) % 10 == 0 or i + 1 == len(segments):
                log(f"  TTS: {i + 1}/{len(segments)} segments generated")

        # --- Compute the minimum uniform speed that fits everything ---
        # Each segment can delay its start if the previous one hasn't
        # finished, so tight segments borrow slack from generous ones.
        # Binary-search for the lowest speed where the last segment's
        # audio ends before the video does — no trimming needed.
        active = [(i, info) for i, info in enumerate(seg_info) if info]

        def _fits(speed):
            """Check if all segments fit at the given speed."""
            cursor = 0.0
            for idx, (_, tts_s, _) in active:
                orig_start = segments[idx]["start"]
                start = max(orig_start, cursor)
                cursor = start + tts_s / (speed * SAMPLE_RATE)
            return cursor <= video_duration

        lo, hi = 1.0, 3.0
        if _fits(lo):
            uniform_speed = 1.0
        else:
            # Binary search (20 iterations gives ~1e-6 precision)
            for _ in range(20):
                mid = (lo + hi) / 2
                if _fits(mid):
                    hi = mid
                else:
                    lo = mid
            uniform_speed = hi

        if uniform_speed > 1.001:
            log(f"  TTS uniform speed: {uniform_speed:.2f}x")

        # --- Compute adjusted start times with the chosen speed ---
        adjusted_starts = {}
        cursor = 0.0
        for idx, (_, tts_s, _) in active:
            orig_start = segments[idx]["start"]
            start = max(orig_start, cursor)
            adjusted_starts[idx] = start
            cursor = start + tts_s / (uniform_speed * SAMPLE_RATE)

        # --- Pass 2: decode with uniform atempo and place in buffer ---
        for idx, (mp3_path, tts_samples, max_samples) in active:
            if uniform_speed > 1.001:
                # Build atempo filter chain (each filter limited to 2.0x)
                parts, s = [], uniform_speed
                while s > 2.0:
                    parts.append("atempo=2.0")
                    s /= 2.0
                parts.append(f"atempo={s:.4f}")
                af = ",".join(parts)

                tempo_cmd = [
                    ffmpeg_path, "-v", "error",
                    "-i", mp3_path,
                    "-af", af,
                    "-f", "f32le",
                    "-acodec", "pcm_f32le",
                    "-ar", str(SAMPLE_RATE),
                    "-ac", "1",
                    "pipe:1",
                ]
                proc = subprocess.run(tempo_cmd, capture_output=True,
                                      check=True)
                pcm = np.frombuffer(proc.stdout, dtype=np.float32)
            else:
                decode_cmd = [
                    ffmpeg_path, "-v", "error",
                    "-i", mp3_path,
                    "-f", "f32le",
                    "-acodec", "pcm_f32le",
                    "-ar", str(SAMPLE_RATE),
                    "-ac", "1",
                    "pipe:1",
                ]
                proc = subprocess.run(decode_cmd, capture_output=True,
                                      check=True)
                pcm = np.frombuffer(proc.stdout, dtype=np.float32)

            # Place at the adjusted start offset (additive mix)
            offset = int(adjusted_starts[idx] * SAMPLE_RATE)
            end = min(offset + len(pcm), total_samples)
            length = end - offset
            if length > 0:
                mixed[offset:end] += pcm[:length]

        # Clip to [-1, 1] and convert to int16
        np.clip(mixed, -1.0, 1.0, out=mixed)
        pcm_int16 = (mixed * 32767).astype(np.int16)

        # Write WAV with manual 44-byte header
        data_bytes = pcm_int16.tobytes()
        num_channels = 1
        sample_width = 2  # 16-bit
        byte_rate = SAMPLE_RATE * num_channels * sample_width
        block_align = num_channels * sample_width

        with open(output_path, "wb") as f:
            # RIFF header
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + len(data_bytes)))
            f.write(b"WAVE")
            # fmt chunk
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))               # chunk size
            f.write(struct.pack("<H", 1))                 # PCM format
            f.write(struct.pack("<H", num_channels))
            f.write(struct.pack("<I", SAMPLE_RATE))
            f.write(struct.pack("<I", byte_rate))
            f.write(struct.pack("<H", block_align))
            f.write(struct.pack("<H", sample_width * 8))  # bits per sample
            # data chunk
            f.write(b"data")
            f.write(struct.pack("<I", len(data_bytes)))
            f.write(data_bytes)

        log(f"  TTS WAV written: {output_path}")

    finally:
        # Clean up temp MP3 files
        for p in mp3_paths:
            if os.path.exists(p):
                os.unlink(p)
        if os.path.isdir(tmpdir):
            os.rmdir(tmpdir)


# ---------------------------------------------------------------------------
# ASS subtitle generation
# ---------------------------------------------------------------------------

def _ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


_LANG_NAMES = {
    "es": "Spanish", "en": "English", "fr": "French", "de": "German",
    "pt": "Portuguese", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese",
    "th": "Thai", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
}


def generate_ass(segments, position, margin, width, height, output_path,
                 font_size=None, target_lang=None):
    """Write an ASS subtitle file with styling and positioning."""
    if font_size is None:
        font_size = max(18, int(height * 0.05))

    alignment = 8 if position == "top" else 2
    lang_name = _LANG_NAMES.get(target_lang, target_lang or "Translated")

    header = (
        "[Script Info]\n"
        f"Title: Auto-generated {lang_name} Subtitles\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial,{font_size},"
        "&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        f"-1,0,0,0,100,100,0,0,1,2,1,{alignment},20,20,{margin},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    lines = [header]
    for seg in segments:
        start = _ass_time(seg["start"])
        end = _ass_time(seg["end"])
        text = seg["text"].replace("\n", "\\N")
        lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# SRT export
# ---------------------------------------------------------------------------

def _srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")


# ---------------------------------------------------------------------------
# Burn-in
# ---------------------------------------------------------------------------

def burn_subtitles(video_path, ass_path, output_path, ffmpeg_path,
                   output_codec="h264"):
    """Burn ASS subtitles into the video.

    Note: subtitle burn-in requires a video filter, so re-encoding is
    mandatory even when output_codec is "copy".
    """
    tmp_ass = tempfile.NamedTemporaryFile(
        suffix=".ass", delete=False, prefix="subs_"
    )
    tmp_ass.close()
    with open(ass_path, "rb") as src, open(tmp_ass.name, "wb") as dst:
        dst.write(src.read())

    escaped = tmp_ass.name.replace("\\", "/").replace(":", "\\:")

    # Video filter present → always re-encode video as H.264
    v_codec = ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
    # Audio can be copied when no new audio is mixed in
    if output_codec == "copy":
        a_codec = ["-c:a", "copy"]
    else:
        a_codec = ["-c:a", "aac", "-b:a", "128k"]

    cmd = [
        ffmpeg_path, "-i", video_path,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", f"ass={escaped}",
    ] + v_codec + a_codec + [
        "-movflags", "+faststart",
        "-y", output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        os.unlink(tmp_ass.name)


# ---------------------------------------------------------------------------
# Interactive prompt helper
# ---------------------------------------------------------------------------

def ask_user(prompt, options=("y", "n"), default="y"):
    """Ask the user a yes/no question. Works even when stdin is a pipe."""
    try:
        tty = open("/dev/tty", "r")
    except OSError:
        tty = sys.stdin

    hint = "/".join(o.upper() if o == default else o for o in options)
    try:
        while True:
            print(f"{prompt} [{hint}]: ", end="", flush=True)
            answer = tty.readline().strip().lower()
            if not answer:
                return default
            if answer in options:
                return answer
            print(f"  Please enter one of: {', '.join(options)}")
    finally:
        if tty is not sys.stdin:
            tty.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate and burn Spanish subtitles into a video.",
    )
    ap.add_argument("input", help="Input video file")
    ap.add_argument("-o", "--output", help="Output video file")
    ap.add_argument(
        "--lang", default="auto",
        help="Force source language for Whisper (default: auto-detect)",
    )
    ap.add_argument(
        "--target-lang", default="es",
        help="Target subtitle language (default: es/Spanish)",
    )
    ap.add_argument(
        "--model", default="small",
        help="Whisper model size: tiny/base/small/medium/large (default: small)",
    )
    ap.add_argument("--font-size", type=int, help="Override subtitle font size")
    ap.add_argument(
        "--srt-only", action="store_true",
        help="Only generate SRT + ASS files, don't burn in",
    )
    ap.add_argument(
        "--position", choices=["top", "bottom", "auto"], default="auto",
        help="Force subtitle position (default: auto-detect)",
    )
    ap.add_argument(
        "--no-ocr", action="store_true",
        help="Skip OCR detection, go straight to Whisper transcription",
    )
    ap.add_argument(
        "--ocr-sync", action="store_true",
        help="Auto-accept OCR sync prompt (non-interactive)",
    )
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--ffprobe", default=None)
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # media-tools/
    ffmpeg = args.ffmpeg or os.path.join(base_dir, "bin", "ffmpeg")
    ffprobe = args.ffprobe or os.path.join(base_dir, "bin", "ffprobe")

    input_path = os.path.abspath(args.input)
    basename = os.path.splitext(os.path.basename(input_path))[0]

    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        out_dir = os.path.join(base_dir, "subbed")
        output_path = os.path.join(out_dir, f"{basename}_subtitled.mp4")

    target = args.target_lang
    use_ocr_sync = False
    ocr_segments = None
    raw_ocr = None

    # ---- Step 1: Analyze frame regions ----------------------------------------
    if args.position == "auto":
        print("[1/4] Analyzing video for existing subtitles...")
        position, margin, existing_region, width, height = \
            find_subtitle_placement(input_path, ffmpeg, ffprobe)
    else:
        position = args.position
        existing_region = "bottom" if position == "top" else "top"
        w, h, _, _ = get_video_info(input_path, ffprobe)
        width, height = w, h
        margin = max(10, int(height * 0.03))
        print(f"[1/4] Using forced position: {position}")

    # ---- Step 1b: OCR scan for burned-in subtitles ----------------------------
    if not args.no_ocr:
        print(f"\n[1b] Scanning for burned-in subtitles ({existing_region} region)...")
        raw_ocr = detect_burned_in_subs(
            input_path, existing_region, ffmpeg, ffprobe,
        )

        if raw_ocr:
            is_usable, confidence, good_segments = assess_ocr_quality(raw_ocr)
            print(f"\n  OCR quality: {len(good_segments)}/{len(raw_ocr)} "
                  f"readable segments ({confidence:.0%} confidence)")

            if is_usable:
                ocr_segments = good_segments

                # If the burned-in subs are already in the target language,
                # there's nothing useful to add.
                if target == "en":
                    print(f"\n  Video already has burned-in English subtitles "
                          f"and target is '{target}'.")
                    print("  Nothing to do — skipping subtitle generation.")
                    sys.exit(0)

                print(f"\n  Readable burned-in subtitles detected:")
                print("  " + "-" * 60)
                for i, seg in enumerate(ocr_segments, 1):
                    start = _srt_time(seg["start"])
                    end = _srt_time(seg["end"])
                    preview = seg["text"].replace("\n", " ")[:80]
                    if len(seg["text"]) > 80:
                        preview += "..."
                    print(f"  {i:2d}. [{start} -> {end}]")
                    print(f"      {preview}")
                print("  " + "-" * 60)

                if args.ocr_sync:
                    answer = "y"
                    print(f"\n  --ocr-sync: auto-accepting OCR sync")
                else:
                    answer = ask_user(
                        f"\n  Sync {target} subtitles with these timings?",
                        options=("y", "n"), default="y",
                    )
                if answer == "y":
                    use_ocr_sync = True
                    print(f"  -> Will use OCR timings + translate to {target}")
                else:
                    print("  -> Will use Whisper transcription instead")
            else:
                print("  OCR text is too noisy — skipping to Whisper transcription.")
        else:
            print("  No burned-in subtitles detected.")

    # ---- Step 2: Get subtitle segments ----------------------------------------
    if use_ocr_sync and ocr_segments:
        print(f"\n[2/4] Translating OCR subtitles to {target}...")
        segments = translate_segments(ocr_segments, "en", target)
        print(f"  Translated {len(segments)} segments")
    else:
        print(f"\n[2/4] Transcribing audio...")
        segments, detected_lang = transcribe_audio(
            input_path, language=args.lang, model_size=args.model,
        )

        if not segments:
            print("No speech detected — nothing to subtitle.")
            sys.exit(1)

        print(f"  Found {len(segments)} subtitle segments")

        # If audio is already in the target language AND the OCR scan found
        # burned-in subtitles (even noisy ones), the video is already subbed.
        if detected_lang == target and raw_ocr:
            print(f"\n  Audio is already in '{target}' and the video has "
                  f"existing burned-in subtitles.")
            print("  Nothing to do — skipping subtitle generation.")
            sys.exit(0)

        # Translate if needed
        if detected_lang != target:
            print(f"\n[2b] Source ({detected_lang}) differs from target ({target})...")
            segments = translate_segments(segments, detected_lang, target)

    # ---- Step 3: Generate subtitle files --------------------------------------
    srt_path = os.path.join(os.path.dirname(output_path), f"{basename}.srt")
    ass_path = os.path.join(os.path.dirname(output_path), f"{basename}.ass")

    print(f"\n[3/4] Generating subtitles (position: {position})...")
    save_srt(segments, srt_path)
    print(f"  SRT saved: {srt_path}")
    generate_ass(segments, position, margin, width, height, ass_path,
                 font_size=args.font_size, target_lang=target)
    print(f"  ASS saved: {ass_path}")

    if args.srt_only:
        print("\nDone (subtitle files only).")
        return

    # ---- Step 4: Burn in ------------------------------------------------------
    print(f"\n[4/4] Burning subtitles into video...")
    burn_subtitles(input_path, ass_path, output_path, ffmpeg)
    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
