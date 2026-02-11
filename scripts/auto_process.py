#!/usr/bin/env python3
"""
auto_process.py — Smart video assessment & processing.

Automatically detects aspect ratio, audio language, and burned-in subtitles,
then decides what processing is needed and executes it.

Optimizations:
  - Whisper model loaded once, reused for detection and transcription
  - GPU auto-detection (CUDA when available)
  - Parallel assessment (language + OCR run concurrently)
  - Video info fetched once and passed through
  - Single ffmpeg pass when both conversion and subtitle burn are needed

Phases:
  1. Assessment  — dimensions, audio language, quick OCR sample
  2. Decision    — what conversion/subtitle work is needed
  3. Execution   — direct Python calls into subtitle_gen functions

Usage:
  CLI:  python scripts/auto_process.py <video_or_url> [--target-lang es] [--model small] [--dry-run]
  API:  from scripts.auto_process import process_video
        result = process_video("video.mp4", on_progress=my_callback)
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Allow importing from the same scripts/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from progress import ProgressUpdate, StepTracker, print_progress
from subtitle_gen import (
    assess_ocr_quality,
    burn_subtitles,
    detect_burned_in_subs,
    detect_language_quick,
    find_subtitle_placement,
    generate_ass,
    get_video_info,
    load_whisper_model,
    ocr_frame,
    save_srt,
    synthesize_dubbed_audio,
    transcribe_audio,
    translate_segments,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # media-tools/


# ---------------------------------------------------------------------------
# Phase 1 — Assessment
# ---------------------------------------------------------------------------

def assess_aspect_ratio(width, height):
    """Check if video is already 10:9."""
    is_10_9 = (width * 9 == height * 10)
    actual_ratio = width / height if height else 0
    return is_10_9, actual_ratio


def _detect_language(video_path, model):
    """Language detection wrapper for parallel execution (quiet)."""
    return detect_language_quick(video_path, model=model)


def _check_existing_subs(video_path, base_dir):
    """Check if an SRT file already exists for this video and detect its language.

    Returns (srt_path, detected_lang) or (None, None) if no SRT found.
    """
    from langdetect import detect, LangDetectException

    basename = os.path.splitext(os.path.basename(video_path))[0]
    srt_path = os.path.join(base_dir, "subbed", f"{basename}.srt")

    if not os.path.isfile(srt_path):
        return None, None

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract text lines (skip sequence numbers and timestamps)
    text_lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\d+$', line):
            continue
        if re.match(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}', line):
            continue
        text_lines.append(line)

    combined = " ".join(text_lines)
    if len(combined) < 20:
        return srt_path, None

    try:
        detected = detect(combined)
    except LangDetectException:
        return srt_path, None

    return srt_path, detected


def _quick_ocr_sample(video_path, ffmpeg_path, width, height, duration):
    """
    Quick OCR sample: extract ~10 frames from the bottom subtitle region,
    OCR each, check if >=30% have readable text.
    Also detects the language of the burned-in subtitles.
    Returns (has_subs, readable_count, total_count, sub_lang). No stdout output.
    """
    from langdetect import detect, LangDetectException

    num_samples = 10
    tmpdir = tempfile.mkdtemp(prefix="auto_ocr_")
    crop_h = int(height * 0.22)
    crop_y = height - crop_h

    readable = 0
    total = 0
    all_text = []

    for i in range(num_samples):
        ts = duration * (i + 1) / (num_samples + 1)
        out = os.path.join(tmpdir, f"sample_{i:03d}.png")
        subprocess.run(
            [ffmpeg_path, "-v", "error", "-ss", str(ts), "-i", video_path,
             "-vf", f"crop={width}:{crop_h}:0:{crop_y}",
             "-vframes", "1", "-y", out],
            capture_output=True,
        )
        if not os.path.exists(out):
            continue

        total += 1
        text = ocr_frame(out)
        words = text.split()
        real_words = [w for w in words if sum(1 for c in w if c.isalpha()) >= 3]
        if len(real_words) >= 3:
            readable += 1
            all_text.append(" ".join(real_words))

        os.unlink(out)

    os.rmdir(tmpdir)

    has_subs = total > 0 and (readable / total) >= 0.30

    # Detect subtitle language from collected OCR text
    sub_lang = None
    if has_subs and all_text:
        # Filter to only clean words (mostly alphabetic) to reduce OCR noise
        clean_words = []
        for text in all_text:
            for word in text.split():
                alpha_chars = sum(1 for c in word if c.isalpha())
                if len(word) >= 3 and alpha_chars / len(word) >= 0.8:
                    clean_words.append(word)
        clean_text = " ".join(clean_words)
        if len(clean_words) >= 5:
            try:
                sub_lang = detect(clean_text)
            except LangDetectException:
                pass

    return has_subs, readable, total, sub_lang


# ---------------------------------------------------------------------------
# Phase 2 — Decision
# ---------------------------------------------------------------------------

def decide_actions(is_10_9, actual_ratio, audio_lang, has_burned_subs, target_lang,
                   width, height, sub_lang=None, convert_portrait=True,
                   existing_srt_lang=None):
    """
    Decision matrix:
      Audio Language | Has Burned-in Subs | Subs Language | Action
      target         | Yes                | *             | Skip subtitles
      target         | No                 | *             | Transcribe only (no translation)
      other          | Yes                | target        | Skip subtitles (already in target)
      other          | Yes                | other/unknown | Whisper transcribe + translate
      other          | No                 | *             | Whisper transcribe + translate

    Additionally: convert to 10:9 if portrait (height > width) and not already 10:9.
    Landscape videos are left at their original aspect ratio.
    """
    actions = []

    # When subtitles are disabled, only consider conversion
    if target_lang is None:
        is_portrait = height > width
        if not is_10_9 and is_portrait and convert_portrait:
            actions.append({
                "type": "convert",
                "reason": f"Portrait video ({width}x{height}, ratio {actual_ratio:.4f}) — stretch to 10:9",
            })
        actions.append({
            "type": "skip_subs",
            "reason": "Subtitles disabled by user",
        })
        return actions

    is_portrait = height > width
    if not is_10_9 and is_portrait and convert_portrait:
        actions.append({
            "type": "convert",
            "reason": f"Portrait video ({width}x{height}, ratio {actual_ratio:.4f}) — stretch to 10:9",
        })
    elif not is_10_9 and is_portrait and not convert_portrait:
        actions.append({
            "type": "skip_convert",
            "reason": f"Portrait video ({width}x{height}, ratio {actual_ratio:.4f}) — keeping original format",
        })
    elif not is_10_9 and not is_portrait:
        actions.append({
            "type": "skip_convert",
            "reason": f"Landscape video ({width}x{height}, ratio {actual_ratio:.4f}) — skipping 10:9 conversion",
        })

    # If existing SRT already matches target language, skip subtitle generation
    if existing_srt_lang == target_lang:
        actions.append({
            "type": "skip_subs",
            "reason": f"Subtitles already exist in '{existing_srt_lang}'",
        })
        return actions

    audio_is_target = (audio_lang == target_lang)
    subs_are_target = (sub_lang == target_lang) if sub_lang else False

    if audio_is_target and has_burned_subs:
        actions.append({
            "type": "skip_subs",
            "reason": f"Audio is '{audio_lang}' (target) and video has burned-in subs",
        })
    elif audio_is_target and not has_burned_subs:
        actions.append({
            "type": "transcribe_only",
            "reason": f"Audio is '{audio_lang}' (target), no burned-in subs — transcribe without translation",
        })
    elif not audio_is_target and has_burned_subs and subs_are_target:
        actions.append({
            "type": "skip_subs",
            "reason": f"Burned-in subs detected in '{sub_lang}' (target language) — no new subtitles needed",
        })
    elif not audio_is_target and has_burned_subs:
        sub_info = f" (detected as '{sub_lang}')" if sub_lang else ""
        actions.append({
            "type": "whisper_translate",
            "reason": f"Audio is '{audio_lang}', burned-in subs{sub_info} not in target — Whisper + translate to {target_lang}",
        })
    else:
        actions.append({
            "type": "whisper_translate",
            "reason": f"Audio is '{audio_lang}', no burned-in subs — Whisper + translate to {target_lang}",
        })

    return actions


def _format_plan(actions, video_path, dry_run=False):
    """Format the execution plan as a string."""
    label = "DRY RUN — " if dry_run else ""
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"  {label}EXECUTION PLAN")
    lines.append(f"{'=' * 60}")
    lines.append(f"  Input: {video_path}")

    step = 1
    for action in actions:
        atype = action["type"]
        reason = action["reason"]
        if atype == "skip_subs":
            lines.append(f"  Step {step}: SKIP subtitles — {reason}")
        elif atype == "skip_convert":
            lines.append(f"  Step {step}: SKIP 10:9 conversion — {reason}")
        elif atype == "convert":
            lines.append(f"  Step {step}: Convert to 10:9 — {reason}")
        elif atype == "transcribe_only":
            lines.append(f"  Step {step}: Generate subtitles (transcribe, no translation) — {reason}")
        elif atype == "ocr_sync_translate":
            lines.append(f"  Step {step}: Generate subtitles (OCR sync + translate) — {reason}")
        elif atype == "whisper_translate":
            lines.append(f"  Step {step}: Generate subtitles (Whisper + translate) — {reason}")
        step += 1

    lines.append(f"{'=' * 60}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 3 — Execution helpers
# ---------------------------------------------------------------------------

MAX_HEIGHT = 1920  # Cap output resolution for fast encoding


def compute_10_9_dimensions(src_w, src_h):
    """Compute target dimensions for 10:9 pixel ratio (horizontal stretch only).

    Caps height at MAX_HEIGHT to keep encoding fast. The 10:9 stretch is
    applied after downscaling.
    """
    h = min(src_h, MAX_HEIGHT)
    # Ensure even height
    if h % 2 != 0:
        h -= 1
    new_w = round(h * 10 / 9)
    # Ensure even width for H.264
    if new_w % 2 != 0:
        new_w += 1
    return new_w, h


def _get_subtitle_segments(sub_action_type, video_path, target_lang, audio_lang,
                           whisper_model, existing_region, ffmpeg, ffprobe,
                           on_progress=print_progress, tracker=None):
    """
    Produce subtitle segments based on the decided strategy.
    Reuses the already-loaded whisper_model. Falls back to Whisper if OCR
    quality is too low.
    """
    def _log(msg):
        on_progress(ProgressUpdate("execution", msg, -1))

    if sub_action_type == "transcribe_only":
        if tracker:
            tracker.begin("Transcribing audio")
        _log("  Transcribing audio (same language, no translation)...")
        segments, _ = transcribe_audio(
            video_path, language=audio_lang, model_size=None,
            model=whisper_model, log=lambda m: _log(m),
        )
        if tracker:
            tracker.finish("Transcribing audio")
        return segments

    elif sub_action_type == "ocr_sync_translate":
        if tracker:
            tracker.begin("Transcribing audio")
        _log(f"  Running full OCR scan on '{existing_region}' region...")
        raw_ocr = detect_burned_in_subs(
            video_path, existing_region, ffmpeg, ffprobe,
            log=lambda m: _log(m),
        )
        is_usable, confidence, good_segments = assess_ocr_quality(raw_ocr)
        _log(f"  OCR quality: {len(good_segments)}/{len(raw_ocr)} "
             f"readable ({confidence:.0%})")

        if is_usable:
            if tracker:
                tracker.finish("Transcribing audio")
                tracker.begin("Translating subtitles")
            _log(f"  Translating {len(good_segments)} OCR segments "
                 f"to {target_lang}...")
            result = translate_segments(good_segments, audio_lang, target_lang,
                                       log=lambda m: _log(m))
            if tracker:
                tracker.finish("Translating subtitles")
            return result
        else:
            _log("  OCR quality too low — falling back to Whisper...")
            segments, detected = transcribe_audio(
                video_path, language=audio_lang, model_size=None,
                model=whisper_model, log=lambda m: _log(m),
            )
            if tracker:
                tracker.finish("Transcribing audio")
            if detected != target_lang:
                if tracker:
                    tracker.begin("Translating subtitles")
                segments = translate_segments(segments, detected, target_lang,
                                             log=lambda m: _log(m))
                if tracker:
                    tracker.finish("Translating subtitles")
            else:
                if tracker:
                    tracker.skip("Translating subtitles")
            return segments

    elif sub_action_type == "whisper_translate":
        if tracker:
            tracker.begin("Transcribing audio")
        _log("  Transcribing audio with Whisper...")
        segments, detected = transcribe_audio(
            video_path, language=audio_lang, model_size=None,
            model=whisper_model, log=lambda m: _log(m),
        )
        if tracker:
            tracker.finish("Transcribing audio")
        if detected != target_lang and segments:
            if tracker:
                tracker.begin("Translating subtitles")
            _log(f"  Translating: {detected} -> {target_lang}...")
            segments = translate_segments(segments, detected, target_lang,
                                         log=lambda m: _log(m))
            if tracker:
                tracker.finish("Translating subtitles")
        else:
            if tracker:
                tracker.skip("Translating subtitles")
        return segments

    return []


def _ffmpeg_convert_and_burn(input_path, ass_path, output_path,
                             width, height, ffmpeg_path):
    """Single ffmpeg pass: scale to 10:9 + burn ASS subtitles."""
    tmp_ass = tempfile.NamedTemporaryFile(
        suffix=".ass", delete=False, prefix="subs_",
    )
    tmp_ass.close()
    with open(ass_path, "rb") as src, open(tmp_ass.name, "wb") as dst:
        dst.write(src.read())

    escaped = tmp_ass.name.replace("\\", "/").replace(":", "\\:")

    cmd = [
        ffmpeg_path, "-i", input_path,
        "-vf", f"scale={width}:{height},setsar=1,ass={escaped}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-y", output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        os.unlink(tmp_ass.name)


def _ffmpeg_encode_with_dub(input_path, wav_path, ass_path, output_path,
                            scale_w=None, scale_h=None, ffmpeg_path="ffmpeg"):
    """Encode video with dubbed audio WAV, optionally scaling + burning subtitles."""
    tmp_ass = tempfile.NamedTemporaryFile(
        suffix=".ass", delete=False, prefix="subs_",
    )
    tmp_ass.close()
    with open(ass_path, "rb") as src, open(tmp_ass.name, "wb") as dst:
        dst.write(src.read())

    escaped = tmp_ass.name.replace("\\", "/").replace(":", "\\:")

    # Build video filter chain
    vf_parts = []
    if scale_w and scale_h:
        vf_parts.append(f"scale={scale_w}:{scale_h},setsar=1")
    vf_parts.append(f"ass={escaped}")
    vf_str = ",".join(vf_parts)

    cmd = [
        ffmpeg_path,
        "-i", input_path,
        "-i", wav_path,
        "-map", "0:v:0",
        "-map", "1:a",
        "-vf", vf_str,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-y", output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        os.unlink(tmp_ass.name)


def execute_pipeline(video_path, actions, target_lang, audio_lang,
                     whisper_model, width, height, ffmpeg, ffprobe,
                     base_dir=None, on_progress=print_progress, tracker=None,
                     duration=0, dub_audio=False, voice_gender="male"):
    """
    Execute the decided actions using direct Python calls.
    Combines convert + burn into a single ffmpeg pass when both are needed.
    Returns dict with output paths.
    """
    if base_dir is None:
        base_dir = BASE_DIR

    def _log(msg):
        on_progress(ProgressUpdate("execution", msg, -1))

    needs_convert = any(a["type"] == "convert" for a in actions)
    sub_action = next(
        (a for a in actions
         if a["type"] in ("transcribe_only", "ocr_sync_translate", "whisper_translate")),
        None,
    )

    basename = os.path.splitext(os.path.basename(video_path))[0]

    # Determine target dimensions
    if needs_convert:
        target_w, target_h = compute_10_9_dimensions(width, height)
    else:
        target_w, target_h = width, height

    # --- Generate subtitles if needed ---
    segments = None
    position = margin = existing_region = None

    if sub_action:
        if tracker:
            tracker.begin("Analyzing subtitle placement")
        position, margin, existing_region, _, _ = find_subtitle_placement(
            video_path, ffmpeg, ffprobe,
            log=lambda m: _log(m),
        )
        if tracker:
            tracker.finish("Analyzing subtitle placement")

        segments = _get_subtitle_segments(
            sub_action["type"], video_path, target_lang, audio_lang,
            whisper_model, existing_region, ffmpeg, ffprobe,
            on_progress=on_progress, tracker=tracker,
        )

        if not segments:
            _log("  No speech/subtitle segments found — skipping subtitles.")
            sub_action = None
            if tracker:
                tracker.skip("Generating subtitle files")
                tracker.skip("Synthesizing audio")
                if not needs_convert:
                    tracker.skip("Encoding video")

    # --- Write subtitle files + encode ---
    result_paths = {}

    if segments:
        out_dir = os.path.join(base_dir, "subbed")
        os.makedirs(out_dir, exist_ok=True)

        ass_path = os.path.join(out_dir, f"{basename}.ass")
        srt_path = os.path.join(out_dir, f"{basename}.srt")
        output_path = os.path.join(out_dir, f"{basename}_subtitled.mp4")

        if tracker:
            tracker.begin("Generating subtitle files")
        _log(f"  Writing subtitle files (position: {position})...")
        save_srt(segments, srt_path)
        _log(f"  SRT: {srt_path}")
        generate_ass(segments, position, margin, target_w, target_h, ass_path)
        _log(f"  ASS: {ass_path}")
        if tracker:
            tracker.finish("Generating subtitle files")

        result_paths["output_srt"] = srt_path
        result_paths["output_ass"] = ass_path

        # --- Synthesize dubbed audio if requested ---
        wav_path = None
        if dub_audio:
            if tracker:
                tracker.begin("Synthesizing audio")
            _log("  Synthesizing TTS audio...")
            wav_path = os.path.join(out_dir, f"{basename}_dub.wav")
            synthesize_dubbed_audio(
                segments, target_lang, duration, wav_path, ffmpeg,
                voice_gender=voice_gender, log=lambda m: _log(m),
            )
            if tracker:
                tracker.finish("Synthesizing audio")
        else:
            if tracker:
                tracker.skip("Synthesizing audio")

        if tracker:
            tracker.begin("Encoding video")
        if dub_audio and wav_path:
            scale_w = target_w if needs_convert else None
            scale_h = target_h if needs_convert else None
            _log("  Encoding video with dubbed audio...")
            _ffmpeg_encode_with_dub(
                video_path, wav_path, ass_path, output_path,
                scale_w=scale_w, scale_h=scale_h, ffmpeg_path=ffmpeg,
            )
        elif needs_convert:
            _log(f"  Encoding: scale {width}x{height} -> "
                 f"{target_w}x{target_h} + burn subtitles...")
            _ffmpeg_convert_and_burn(
                video_path, ass_path, output_path,
                target_w, target_h, ffmpeg,
            )
        else:
            _log("  Burning subtitles into video...")
            burn_subtitles(video_path, ass_path, output_path, ffmpeg)
        if tracker:
            tracker.finish("Encoding video")

        result_paths["output_video"] = output_path
        return output_path, result_paths

    elif needs_convert:
        if tracker:
            tracker.begin("Encoding video")
        _log("  Converting to 10:9...")
        convert_script = os.path.join(base_dir, "convert109.sh")
        subprocess.run([convert_script, video_path], check=True,
                       capture_output=True)
        subbed_dir = os.path.join(base_dir, "subbed")
        latest = _latest_mp4(subbed_dir)
        if tracker:
            tracker.finish("Encoding video")
        if latest:
            result_paths["output_video"] = latest
            return latest, result_paths
        raise RuntimeError("No converted file found after 10:9 conversion")

    else:
        _log("  Nothing to do — video is already processed.")
        result_paths["output_video"] = video_path
        return video_path, result_paths


def _latest_mp4(directory):
    """Find the most recently modified .mp4 in a directory."""
    files = glob.glob(os.path.join(directory, "*.mp4"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------

def download_url(url, base_dir=None, on_progress=print_progress):
    """Download a URL using yt-dlp directly, return the downloaded file path."""
    if base_dir is None:
        base_dir = BASE_DIR

    on_progress(ProgressUpdate("download", f"\n--- Downloading: {url} ---", 0.0))
    downloads_dir = os.path.join(base_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    # Snapshot files and their mtimes before download
    before = {}
    if os.path.isdir(downloads_dir):
        for f in glob.glob(os.path.join(downloads_dir, "*.mp4")):
            before[f] = os.path.getmtime(f)

    # Find ffmpeg location — prefer bin/ dir, fall back to system
    ffmpeg_dir = os.path.join(base_dir, "bin")
    if not os.path.isfile(os.path.join(ffmpeg_dir, "ffmpeg")):
        ffmpeg_dir = None  # let yt-dlp find system ffmpeg

    cmd = [
        "yt-dlp",
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "-o", os.path.join(downloads_dir, "%(title).80s [%(id)s].%(ext)s"),
        "--no-playlist",
        "--embed-thumbnail",
        "--embed-metadata",
    ]
    if ffmpeg_dir:
        cmd += ["--ffmpeg-location", ffmpeg_dir]
    cmd.append(url)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        on_progress(ProgressUpdate("download", f"yt-dlp failed:\n{output}", -1))
        raise RuntimeError(f"yt-dlp download failed (exit {proc.returncode}): {output[-500:]}")
    on_progress(ProgressUpdate("download", output, 0.5))

    # Strategy 1: Parse yt-dlp output for the actual filename
    latest = _parse_downloaded_file(output, downloads_dir)

    # Strategy 2: Find files that are new or whose mtime changed
    if not latest:
        after = {}
        for f in glob.glob(os.path.join(downloads_dir, "*.mp4")):
            after[f] = os.path.getmtime(f)

        changed = [
            f for f, mtime in after.items()
            if f not in before or mtime != before[f]
        ]
        if changed:
            latest = max(changed, key=os.path.getmtime)

    # Strategy 3: Match by URL video ID
    if not latest:
        latest = _match_by_url(url, downloads_dir)

    # Strategy 4: Last resort — most recent file
    if not latest:
        latest = _latest_mp4(downloads_dir)

    if not latest:
        raise RuntimeError("No .mp4 found in downloads/ after download.")

    on_progress(ProgressUpdate("download", f"  -> Downloaded: {latest}", 1.0))
    return latest


def _parse_downloaded_file(output, downloads_dir):
    """Parse yt-dlp output to find the actual downloaded filepath."""
    for line in output.splitlines():
        # "has already been downloaded"
        m = re.search(r'\[download\]\s+(.+\.mp4)\s+has already been downloaded', line)
        if m:
            path = m.group(1).strip()
            if os.path.isfile(path):
                return path

        # "Destination:"
        m = re.search(r'\[download\]\s+Destination:\s+(.+\.mp4)', line)
        if m:
            path = m.group(1).strip()
            if os.path.isfile(path):
                return path

        # [Metadata] Adding metadata to "path"
        m = re.search(r'\[Metadata\]\s+Adding metadata to\s+"(.+\.mp4)"', line)
        if m:
            path = m.group(1).strip()
            if os.path.isfile(path):
                return path

    return None


def _match_by_url(url, downloads_dir):
    """Try to find a downloaded file by matching the video ID from the URL."""
    m = re.search(r'/status/(\d+)', url)
    if not m:
        return None

    tweet_id = m.group(1)
    for f in glob.glob(os.path.join(downloads_dir, "*.mp4")):
        basename = os.path.basename(f)
        id_match = re.search(r'\[(\d+)\]\.mp4$', basename)
        if id_match and id_match.group(1) == tweet_id:
            return f

    return None


def is_url(s):
    """Check if the input looks like a URL."""
    return bool(re.match(r'^https?://', s))


# ---------------------------------------------------------------------------
# Public API — process_video()
# ---------------------------------------------------------------------------

def process_video(input_source, target_lang="es", model_size="small",
                  dry_run=False, base_dir=None,
                  on_progress=print_progress, convert_portrait=True,
                  dub_audio=False, voice_gender="male"):
    """
    Main entry point for the auto-process pipeline.

    Args:
        input_source: Video file path or URL.
        target_lang: Target subtitle language (default: "es").
        model_size: Whisper model size (default: "small").
        dry_run: If True, assess only — show plan without executing.
        base_dir: Base directory for downloads/subbed (default: media-tools/).
        on_progress: Callback receiving ProgressUpdate objects.

    Returns:
        dict with keys:
            status: "completed" | "dry_run" | "error"
            output_video: path to final video (if applicable)
            output_srt: path to SRT file (if applicable)
            output_ass: path to ASS file (if applicable)
            assessment: dict with video info from Phase 1
            actions: list of action dicts from Phase 2
    """
    if base_dir is None:
        base_dir = BASE_DIR

    ffmpeg = os.path.join(base_dir, "bin", "ffmpeg")
    ffprobe = os.path.join(base_dir, "bin", "ffprobe")

    # If bin/ ffmpeg doesn't exist, fall back to system
    if not os.path.isfile(ffmpeg):
        ffmpeg = "ffmpeg"
    if not os.path.isfile(ffprobe):
        ffprobe = "ffprobe"

    # --- Check for existing subtitles before expensive work ---
    existing_srt_lang = None
    if target_lang is not None and not is_url(input_source):
        video_path_resolved = os.path.abspath(input_source)
        srt_path, detected_lang = _check_existing_subs(video_path_resolved, base_dir)
        if detected_lang == target_lang:
            existing_srt_lang = detected_lang
            on_progress(ProgressUpdate("assessment",
                f"  Existing subtitles found: {srt_path} (detected: {detected_lang})"
                f" — skipping subtitle generation", -1))

    # --- Build step tracker ---
    skip_whisper = target_lang is None or existing_srt_lang is not None
    tracker = StepTracker(on_progress)
    if is_url(input_source):
        tracker.add_step("Downloading video", weight=15)
    tracker.add_step("Analyzing video", weight=2)
    if not skip_whisper:
        tracker.add_step("Loading Whisper model", weight=8)
        tracker.add_step("Detecting language & scanning subtitles", weight=10)
    tracker.add_step("Planning actions", weight=1)
    tracker.add_step("Analyzing subtitle placement", weight=5)
    tracker.add_step("Transcribing audio", weight=30)
    tracker.add_step("Translating subtitles", weight=8)
    tracker.add_step("Generating subtitle files", weight=1)
    tracker.add_step("Synthesizing audio", weight=20)
    tracker.add_step("Encoding video", weight=25)

    result = {
        "status": "error",
        "output_video": None,
        "output_srt": None,
        "output_ass": None,
        "assessment": {},
        "actions": [],
    }

    # --- Resolve input ---
    if is_url(input_source):
        tracker.begin("Downloading video")
        video_path = download_url(input_source, base_dir=base_dir,
                                  on_progress=on_progress)
        tracker.finish("Downloading video")
    else:
        video_path = os.path.abspath(input_source)
        if not os.path.isfile(video_path):
            on_progress(ProgressUpdate("assessment", f"ERROR: File not found: {video_path}", -1))
            result["status"] = "error"
            return result

    # === Phase 1: Assessment ===
    tracker.begin("Analyzing video")
    on_progress(ProgressUpdate("assessment", f"  File: {video_path}", -1))

    width, height, duration, subtitle_streams = get_video_info(video_path, ffprobe)
    is_10_9, actual_ratio = assess_aspect_ratio(width, height)

    on_progress(ProgressUpdate("assessment",
        f"  Dimensions: {width}x{height} (ratio: {actual_ratio:.4f})", -1))
    on_progress(ProgressUpdate("assessment",
        f"  Duration: {duration:.1f}s ({duration/60:.1f}min)", -1))
    if subtitle_streams:
        on_progress(ProgressUpdate("assessment",
            f"  Embedded subtitle streams: {len(subtitle_streams)}", -1))
    tracker.finish("Analyzing video")

    # Load Whisper model + detect language/subs (skip when subtitles disabled or existing SRT matches)
    if skip_whisper:
        whisper_model = None
        audio_lang = None
        has_burned_subs = False
        sub_lang = None
        if target_lang is None:
            on_progress(ProgressUpdate("assessment", "  Subtitles disabled — skipping Whisper/OCR", -1))
        else:
            on_progress(ProgressUpdate("assessment", "  Existing SRT matches target — skipping Whisper/OCR", -1))

        result["assessment"] = {
            "width": width, "height": height, "duration": duration,
            "is_10_9": is_10_9, "actual_ratio": actual_ratio,
            "audio_lang": None, "lang_prob": 0.0,
            "has_burned_subs": False, "sub_lang": None,
            "readable_frames": 0, "total_frames": 0,
            "subtitle_streams": len(subtitle_streams),
        }
    else:
        tracker.begin("Loading Whisper model")
        whisper_model = load_whisper_model(model_size,
            log=lambda m: on_progress(ProgressUpdate("assessment", m, -1)))
        tracker.finish("Loading Whisper model")

        # Parallel: language detection + OCR sampling
        tracker.begin("Detecting language & scanning subtitles")
        with ThreadPoolExecutor(max_workers=2) as executor:
            lang_future = executor.submit(
                _detect_language, video_path, whisper_model,
            )
            ocr_future = executor.submit(
                _quick_ocr_sample, video_path, ffmpeg, width, height, duration,
            )

            audio_lang, lang_prob = lang_future.result()
            has_burned_subs, readable, total, sub_lang = ocr_future.result()

        on_progress(ProgressUpdate("assessment",
            f"  Audio language: {audio_lang} (confidence: {lang_prob:.2f})", -1))
        on_progress(ProgressUpdate("assessment",
            f"  Burned-in subs: {'YES' if has_burned_subs else 'NO'} "
            f"({readable}/{total} frames with readable text)", -1))
        if sub_lang:
            on_progress(ProgressUpdate("assessment",
                f"  Burned-in subs language: {sub_lang}", -1))
        tracker.finish("Detecting language & scanning subtitles")

        result["assessment"] = {
            "width": width, "height": height, "duration": duration,
            "is_10_9": is_10_9, "actual_ratio": actual_ratio,
            "audio_lang": audio_lang, "lang_prob": lang_prob,
            "has_burned_subs": has_burned_subs, "sub_lang": sub_lang,
            "readable_frames": readable, "total_frames": total,
            "subtitle_streams": len(subtitle_streams),
        }

    # === Phase 2: Decision ===
    tracker.begin("Planning actions")

    actions = decide_actions(is_10_9, actual_ratio, audio_lang, has_burned_subs,
                             target_lang, width, height, sub_lang,
                             convert_portrait=convert_portrait,
                             existing_srt_lang=existing_srt_lang)
    result["actions"] = actions

    plan_text = _format_plan(actions, video_path, dry_run=dry_run)
    on_progress(ProgressUpdate("decision", plan_text, -1))

    # Determine which execution steps are needed and skip the rest
    sub_action = next(
        (a for a in actions
         if a["type"] in ("transcribe_only", "ocr_sync_translate", "whisper_translate")),
        None,
    )
    needs_convert = any(a["type"] == "convert" for a in actions)
    needs_subs = sub_action is not None
    needs_translate = sub_action and sub_action["type"] == "whisper_translate"

    if not needs_subs:
        tracker.skip("Analyzing subtitle placement")
        tracker.skip("Transcribing audio")
        tracker.skip("Translating subtitles")
        tracker.skip("Generating subtitle files")
        tracker.skip("Synthesizing audio")
        if not needs_convert:
            tracker.skip("Encoding video")
    else:
        if not needs_translate:
            tracker.skip("Translating subtitles")
        if not dub_audio:
            tracker.skip("Synthesizing audio")

    tracker.finish("Planning actions")

    if dry_run:
        tracker.done()
        result["status"] = "dry_run"
        return result

    # === Phase 3: Execution ===
    final_file, paths = execute_pipeline(
        video_path, actions, target_lang, audio_lang,
        whisper_model, width, height, ffmpeg, ffprobe,
        base_dir=base_dir, on_progress=on_progress, tracker=tracker,
        duration=duration, dub_audio=dub_audio, voice_gender=voice_gender,
    )

    result.update(paths)
    result["status"] = "completed"

    tracker.done()
    on_progress(ProgressUpdate("done", f"  Final file: {final_file}", 1.0))

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Smart video assessment & auto-processing.",
    )
    ap.add_argument("input", help="Video file path or URL")
    ap.add_argument(
        "--target-lang", default="es",
        help="Target subtitle language (default: es/Spanish)",
    )
    ap.add_argument(
        "--model", default="small",
        help="Whisper model size: tiny/base/small/medium/large (default: small)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Assess only — show plan without executing",
    )
    ap.add_argument(
        "--no-convert", action="store_true",
        help="Skip portrait-to-10:9 conversion",
    )
    ap.add_argument(
        "--no-subs", action="store_true",
        help="Skip subtitles entirely (no Whisper/OCR)",
    )
    ap.add_argument(
        "--dub", action="store_true",
        help="Replace audio with TTS-synthesized translated speech",
    )
    ap.add_argument(
        "--voice", choices=["male", "female"], default="male",
        help="TTS voice gender for dubbing (default: male)",
    )
    args = ap.parse_args()

    result = process_video(
        args.input,
        target_lang=None if args.no_subs else args.target_lang,
        model_size=args.model,
        dry_run=args.dry_run,
        convert_portrait=not args.no_convert,
        dub_audio=args.dub,
        voice_gender=args.voice,
    )

    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
