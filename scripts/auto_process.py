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


def _quick_ocr_sample(video_path, ffmpeg_path, width, height, duration):
    """
    Quick OCR sample: extract ~10 frames from the bottom subtitle region,
    OCR each, check if >=30% have readable text.
    Returns (has_subs, readable_count, total_count). No stdout output.
    """
    num_samples = 10
    tmpdir = tempfile.mkdtemp(prefix="auto_ocr_")
    crop_h = int(height * 0.22)
    crop_y = height - crop_h

    readable = 0
    total = 0

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

        os.unlink(out)

    os.rmdir(tmpdir)

    has_subs = total > 0 and (readable / total) >= 0.30
    return has_subs, readable, total


# ---------------------------------------------------------------------------
# Phase 2 — Decision
# ---------------------------------------------------------------------------

def decide_actions(is_10_9, actual_ratio, audio_lang, has_burned_subs, target_lang,
                   width, height):
    """
    Decision matrix:
      Audio Language | Has Burned-in Subs | Action
      target         | Yes                | Skip subtitles
      target         | No                 | Transcribe only (no translation)
      other          | Yes                | OCR sync + translate
      other          | No                 | Whisper transcribe + translate

    Additionally: convert to 10:9 if portrait (height > width) and not already 10:9.
    Landscape videos are left at their original aspect ratio.
    """
    actions = []

    is_portrait = height > width
    if not is_10_9 and is_portrait:
        actions.append({
            "type": "convert",
            "reason": f"Portrait video ({width}x{height}, ratio {actual_ratio:.4f}) — stretch to 10:9",
        })
    elif not is_10_9 and not is_portrait:
        actions.append({
            "type": "skip_convert",
            "reason": f"Landscape video ({width}x{height}, ratio {actual_ratio:.4f}) — skipping 10:9 conversion",
        })

    audio_is_target = (audio_lang == target_lang)

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
    elif not audio_is_target and has_burned_subs:
        actions.append({
            "type": "ocr_sync_translate",
            "reason": f"Audio is '{audio_lang}', has burned-in subs — OCR sync + translate to {target_lang}",
        })
    else:
        actions.append({
            "type": "whisper_translate",
            "reason": f"Audio is '{audio_lang}', no burned-in subs — Whisper + translate to {target_lang}",
        })

    return actions


def print_plan(actions, video_path, dry_run=False):
    """Display the execution plan."""
    label = "DRY RUN — " if dry_run else ""
    print(f"\n{'=' * 60}")
    print(f"  {label}EXECUTION PLAN")
    print(f"{'=' * 60}")
    print(f"  Input: {video_path}")

    step = 1
    for action in actions:
        atype = action["type"]
        reason = action["reason"]
        if atype == "skip_subs":
            print(f"  Step {step}: SKIP subtitles — {reason}")
        elif atype == "skip_convert":
            print(f"  Step {step}: SKIP 10:9 conversion — {reason}")
        elif atype == "convert":
            print(f"  Step {step}: Convert to 10:9 — {reason}")
        elif atype == "transcribe_only":
            print(f"  Step {step}: Generate subtitles (transcribe, no translation) — {reason}")
        elif atype == "ocr_sync_translate":
            print(f"  Step {step}: Generate subtitles (OCR sync + translate) — {reason}")
        elif atype == "whisper_translate":
            print(f"  Step {step}: Generate subtitles (Whisper + translate) — {reason}")
        step += 1

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Phase 3 — Execution helpers
# ---------------------------------------------------------------------------

def compute_10_9_dimensions(src_w, src_h):
    """Compute target dimensions for 10:9 pixel ratio (horizontal stretch only)."""
    new_w = round(src_h * 10 / 9)
    # Ensure even width for H.264
    if new_w % 2 != 0:
        new_w += 1
    return new_w, src_h


def _get_subtitle_segments(sub_action_type, video_path, target_lang, audio_lang,
                           whisper_model, existing_region, ffmpeg, ffprobe):
    """
    Produce subtitle segments based on the decided strategy.
    Reuses the already-loaded whisper_model. Falls back to Whisper if OCR
    quality is too low.
    """
    if sub_action_type == "transcribe_only":
        print("  Transcribing audio (same language, no translation)...")
        segments, _ = transcribe_audio(
            video_path, language=audio_lang, model_size=None,
            model=whisper_model,
        )
        return segments

    elif sub_action_type == "ocr_sync_translate":
        print(f"  Running full OCR scan on '{existing_region}' region...")
        raw_ocr = detect_burned_in_subs(
            video_path, existing_region, ffmpeg, ffprobe,
        )
        is_usable, confidence, good_segments = assess_ocr_quality(raw_ocr)
        print(f"  OCR quality: {len(good_segments)}/{len(raw_ocr)} "
              f"readable ({confidence:.0%})")

        if is_usable:
            print(f"  Translating {len(good_segments)} OCR segments "
                  f"to {target_lang}...")
            return translate_segments(good_segments, audio_lang, target_lang)
        else:
            print("  OCR quality too low — falling back to Whisper...")
            segments, detected = transcribe_audio(
                video_path, language=audio_lang, model_size=None,
                model=whisper_model,
            )
            if detected != target_lang:
                segments = translate_segments(segments, detected, target_lang)
            return segments

    elif sub_action_type == "whisper_translate":
        print("  Transcribing audio with Whisper...")
        segments, detected = transcribe_audio(
            video_path, language=audio_lang, model_size=None,
            model=whisper_model,
        )
        if detected != target_lang:
            print(f"  Translating: {detected} -> {target_lang}...")
            segments = translate_segments(segments, detected, target_lang)
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
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-y", output_path,
    ]

    try:
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(tmp_ass.name)


def execute_pipeline(video_path, actions, target_lang, audio_lang,
                     whisper_model, width, height, ffmpeg, ffprobe):
    """
    Execute the decided actions using direct Python calls.
    Combines convert + burn into a single ffmpeg pass when both are needed.
    """
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
        print("\n  Analyzing subtitle placement...")
        position, margin, existing_region, _, _ = find_subtitle_placement(
            video_path, ffmpeg, ffprobe,
        )

        segments = _get_subtitle_segments(
            sub_action["type"], video_path, target_lang, audio_lang,
            whisper_model, existing_region, ffmpeg, ffprobe,
        )

        if not segments:
            print("  No speech/subtitle segments found — skipping subtitles.")
            sub_action = None

    # --- Write subtitle files + encode ---
    if segments:
        out_dir = os.path.join(BASE_DIR, "subbed")
        os.makedirs(out_dir, exist_ok=True)

        ass_path = os.path.join(out_dir, f"{basename}.ass")
        srt_path = os.path.join(out_dir, f"{basename}.srt")
        output_path = os.path.join(out_dir, f"{basename}_subtitled.mp4")

        print(f"\n  Writing subtitle files (position: {position})...")
        save_srt(segments, srt_path)
        print(f"  SRT: {srt_path}")
        generate_ass(segments, position, margin, target_w, target_h, ass_path)
        print(f"  ASS: {ass_path}")

        if needs_convert:
            print(f"\n  Single-pass encode: scale {width}x{height} -> "
                  f"{target_w}x{target_h} + burn subtitles...")
            _ffmpeg_convert_and_burn(
                video_path, ass_path, output_path,
                target_w, target_h, ffmpeg,
            )
        else:
            print("\n  Burning subtitles into video...")
            burn_subtitles(video_path, ass_path, output_path, ffmpeg)

        return output_path

    elif needs_convert:
        print("\n  Converting to 10:9 only...")
        convert_script = os.path.join(BASE_DIR, "convert109.sh")
        subprocess.run([convert_script, video_path], check=True)
        converted_dir = os.path.join(BASE_DIR, "converted")
        latest = _latest_mp4(converted_dir)
        if latest:
            return latest
        print("  ERROR: No converted file found!")
        sys.exit(1)

    else:
        print("\n  Nothing to do — video is already processed.")
        return video_path


def _latest_mp4(directory):
    """Find the most recently modified .mp4 in a directory."""
    files = glob.glob(os.path.join(directory, "*.mp4"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------

def download_url(url):
    """Download a URL using download.sh, return the downloaded file path."""
    print(f"\n--- Downloading: {url} ---")
    downloads_dir = os.path.join(BASE_DIR, "downloads")

    # Snapshot files and their mtimes before download
    before = {}
    if os.path.isdir(downloads_dir):
        for f in glob.glob(os.path.join(downloads_dir, "*.mp4")):
            before[f] = os.path.getmtime(f)

    download_script = os.path.join(BASE_DIR, "download.sh")
    result = subprocess.run(
        [download_script, url],
        check=True, capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    print(output, end="")

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
        print("ERROR: No .mp4 found in downloads/ after download.")
        sys.exit(1)

    print(f"  -> Downloaded: {latest}")
    return latest


def _parse_downloaded_file(output, downloads_dir):
    """Parse yt-dlp output to find the actual downloaded filepath."""
    # yt-dlp prints lines like:
    #   [download] /path/file.mp4 has already been downloaded
    #   [download] Destination: /path/file.mp4
    #   [Metadata] Adding metadata to "/path/file.mp4"
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
    # Extract numeric ID from Twitter/X URLs like .../status/1234567890
    m = re.search(r'/status/(\d+)', url)
    if not m:
        return None

    tweet_id = m.group(1)
    # yt-dlp may use a different video ID (e.g. for quote tweets),
    # so also capture the output from download.sh via yt-dlp's naming pattern
    for f in glob.glob(os.path.join(downloads_dir, "*.mp4")):
        # Files are named: "title [VIDEO_ID].mp4"
        basename = os.path.basename(f)
        id_match = re.search(r'\[(\d+)\]\.mp4$', basename)
        if id_match and id_match.group(1) == tweet_id:
            return f

    return None


def is_url(s):
    """Check if the input looks like a URL."""
    return bool(re.match(r'^https?://', s))


# ---------------------------------------------------------------------------
# Main
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
    args = ap.parse_args()

    ffmpeg = os.path.join(BASE_DIR, "bin", "ffmpeg")
    ffprobe = os.path.join(BASE_DIR, "bin", "ffprobe")

    # --- Resolve input ---
    if is_url(args.input):
        video_path = download_url(args.input)
    else:
        video_path = os.path.abspath(args.input)
        if not os.path.isfile(video_path):
            print(f"ERROR: File not found: {video_path}")
            sys.exit(1)

    # === Phase 1: Assessment ===
    print(f"\n{'=' * 60}")
    print("  PHASE 1: ASSESSMENT")
    print(f"{'=' * 60}")
    print(f"  File: {video_path}\n")

    # Dimensions (single ffprobe call, results reused everywhere)
    print("  Checking dimensions...")
    width, height, duration, subtitle_streams = get_video_info(video_path, ffprobe)
    is_10_9, actual_ratio = assess_aspect_ratio(width, height)
    print(f"  -> Dimensions: {width}x{height} (ratio: {actual_ratio:.4f})")
    print(f"  -> Is 10:9: {'YES' if is_10_9 else 'NO'}")
    print(f"  -> Duration: {duration:.1f}s ({duration/60:.1f}min)")

    if subtitle_streams:
        print(f"  -> Embedded subtitle streams: {len(subtitle_streams)}")

    # Load Whisper model once (with GPU auto-detection)
    print()
    whisper_model = load_whisper_model(args.model)

    # Parallel: language detection + OCR sampling
    print("\n  Running parallel assessment...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        lang_future = executor.submit(
            _detect_language, video_path, whisper_model,
        )
        ocr_future = executor.submit(
            _quick_ocr_sample, video_path, ffmpeg, width, height, duration,
        )

        audio_lang, lang_prob = lang_future.result()
        has_burned_subs, readable, total = ocr_future.result()

    print(f"  -> Audio language: {audio_lang} (confidence: {lang_prob:.2f})")
    print(f"  -> Burned-in subs: {'YES' if has_burned_subs else 'NO'} "
          f"({readable}/{total} frames with readable text)")

    # === Phase 2: Decision ===
    print(f"\n{'=' * 60}")
    print("  PHASE 2: DECISION")
    print(f"{'=' * 60}")

    actions = decide_actions(is_10_9, actual_ratio, audio_lang, has_burned_subs,
                             args.target_lang, width, height)
    print_plan(actions, video_path, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run complete — no changes made.")
        return

    # === Phase 3: Execution ===
    print(f"{'=' * 60}")
    print("  PHASE 3: EXECUTION")
    print(f"{'=' * 60}")

    final_file = execute_pipeline(
        video_path, actions, args.target_lang, audio_lang,
        whisper_model, width, height, ffmpeg, ffprobe,
    )

    print(f"\n{'=' * 60}")
    print(f"  DONE!")
    print(f"  Final file: {final_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
