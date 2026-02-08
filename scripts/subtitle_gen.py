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
import difflib
import glob
import json
import os
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


def find_subtitle_placement(video_path, ffmpeg_path, ffprobe_path):
    """
    Analyze frame regions and decide where to place new subtitles.
    Returns (new_sub_position, margin, existing_sub_region, width, height).
    """
    frames, w, h, tmpdir = extract_frames(
        video_path, num_frames=10,
        ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path,
    )

    if not frames:
        print("  (frame extraction failed — defaulting to top)")
        return "top", max(10, int(h * 0.03)), "bottom", w, h

    combined, means, stds = analyze_text_regions(frames, w, h)

    labels = ["top", "upper-mid", "center", "lower-mid", "bottom"]
    print("  Region scores (lower = more free):")
    for i, (label, score) in enumerate(zip(labels, combined)):
        tag = " <- candidate" if i in (0, 4) else ""
        print(f"    {label:>10}: {score:6.1f}{tag}")

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

    print(f"  -> Best placement for new subs: {new_position} (margin {margin}px)")
    print(f"  -> Existing content detected at: {existing_region}")
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


def ocr_frame(image_path):
    """Run Tesseract OCR on a single frame. Returns extracted text."""
    import pytesseract

    img = Image.open(image_path)
    # Preprocess: convert to grayscale, increase contrast, scale up
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img, cutoff=1)
    # Scale up 2x for better OCR on small text
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)

    text = pytesseract.image_to_string(
        img, lang="eng",
        config="--psm 6",  # Assume a single uniform block of text
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


def detect_burned_in_subs(video_path, region, ffmpeg_path, ffprobe_path):
    """
    OCR-scan the given region for burned-in subtitles.
    Returns list of {start, end, text} segments, or empty list.
    """
    print(f"  Extracting frames from '{region}' region at 2 fps...")
    frames, tmpdir, interval = extract_frames_dense(
        video_path, fps=2, region=region,
        ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path,
    )

    if not frames:
        return []

    print(f"  Running OCR on {len(frames)} frames...")
    results = []
    for i, fp in enumerate(frames):
        timestamp = i * interval
        text = ocr_frame(fp)
        results.append((timestamp, text))
        # Progress indicator every 20 frames
        if (i + 1) % 20 == 0:
            print(f"    ... {i + 1}/{len(frames)} frames processed")

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


def load_whisper_model(model_size="small"):
    """Load a Whisper model with GPU auto-detection."""
    from faster_whisper import WhisperModel

    device, compute_type = get_whisper_device()
    print(f"  Loading Whisper model '{model_size}' on {device}...")
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

def transcribe_audio(video_path, language, model_size, model=None):
    """Transcribe with faster-whisper. Returns (segments, detected_lang)."""
    if model is None:
        model = load_whisper_model(model_size)

    print("  Transcribing audio...")
    segments_iter, info = model.transcribe(
        video_path,
        language=language if language != "auto" else None,
        word_timestamps=True,
    )

    detected = info.language
    print(f"  Detected language: {detected} (p={info.language_probability:.2f})")

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

def translate_segments(segments, source_lang, target_lang="es"):
    """Translate subtitle text using deep-translator (Google free API)."""
    from deep_translator import GoogleTranslator

    print(f"  Translating {len(segments)} segments: {source_lang} -> {target_lang}")
    translator = GoogleTranslator(source=source_lang, target=target_lang)

    for seg in segments:
        try:
            translated = translator.translate(seg["text"])
            if translated:
                seg["text"] = translated
        except Exception as e:
            print(f"  Warning: Translation failed for segment: {e}")

    return segments


# ---------------------------------------------------------------------------
# ASS subtitle generation
# ---------------------------------------------------------------------------

def _ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_ass(segments, position, margin, width, height, output_path,
                 font_size=None):
    """Write an ASS subtitle file with styling and positioning."""
    if font_size is None:
        font_size = max(18, int(height * 0.05))

    alignment = 8 if position == "top" else 2

    header = (
        "[Script Info]\n"
        "Title: Auto-generated Spanish Subtitles\n"
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

def burn_subtitles(video_path, ass_path, output_path, ffmpeg_path):
    """Burn ASS subtitles into the video with libx264 + AAC."""
    tmp_ass = tempfile.NamedTemporaryFile(
        suffix=".ass", delete=False, prefix="subs_"
    )
    tmp_ass.close()
    with open(ass_path, "rb") as src, open(tmp_ass.name, "wb") as dst:
        dst.write(src.read())

    escaped = tmp_ass.name.replace("\\", "/").replace(":", "\\:")

    cmd = [
        ffmpeg_path, "-i", video_path,
        "-vf", f"ass={escaped}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-y", output_path,
    ]

    try:
        subprocess.run(cmd, check=True)
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
                 font_size=args.font_size)
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
