#!/usr/bin/env python3
"""
clip_extractor.py — Analyse video content, segment by topic, extract clips.

Workflow:
  1. Transcribe video with Whisper (reuses existing subtitle_gen functions)
  2. Send transcript to LLM for topic segmentation (or use fallback splitting)
  3. Let user review/edit sections
  4. Extract clips and run each through the processing pipeline
"""

import json
import os
import re
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from progress import ProgressUpdate, print_progress
from subtitle_gen import get_video_info

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Section data structure
# ---------------------------------------------------------------------------

def make_section(title, summary, start, end, selected=True):
    """Create a section dict."""
    return {
        "title": title,
        "summary": summary,
        "start": start,
        "end": end,
        "selected": selected,
    }


# ---------------------------------------------------------------------------
# Content analysis — LLM-based topic segmentation
# ---------------------------------------------------------------------------

_SEGMENTATION_PROMPT = """\
You are analysing a video transcript to identify thematic sections.

The transcript below has timestamped segments (start–end in seconds).
Group consecutive segments into coherent thematic sections based on topic
shifts, speaker changes, or narrative structure.

Rules:
- Each section must span at least 15 seconds.
- Sections must not overlap and must cover the full transcript.
- Return 2–30 sections depending on content length and variety.
- For each section provide: title (short), summary (1–2 sentences),
  start (seconds, float), end (seconds, float).
- Respond ONLY with a JSON array, no markdown fences, no explanation.

Example response:
[
  {"title": "Introduction", "summary": "The host introduces the topic.", "start": 0.0, "end": 45.2},
  {"title": "Main Discussion", "summary": "Analysis of economic data.", "start": 45.2, "end": 180.5}
]

TRANSCRIPT:
"""


def analyse_content_llm(segments, api_key, video_duration,
                        model="claude-sonnet-4-20250514", log=print):
    """Use Claude API to segment transcript into thematic sections."""
    import anthropic

    # Build transcript text
    transcript_lines = []
    for seg in segments:
        transcript_lines.append(
            f"[{seg['start']:.1f}–{seg['end']:.1f}] {seg['text']}"
        )
    transcript = "\n".join(transcript_lines)

    log("  Sending transcript to LLM for content analysis...")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": _SEGMENTATION_PROMPT + transcript,
        }],
    )

    raw = getattr(response.content[0], "text", "").strip()
    # Strip markdown code fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        sections_data = json.loads(raw)
    except json.JSONDecodeError as e:
        log(f"  Warning: LLM returned invalid JSON: {e}")
        log(f"  Falling back to fixed-duration split.")
        return analyse_content_fixed(segments, video_duration, log=log)

    # Validate and build Section list
    sections = []
    for item in sections_data:
        start = float(item.get("start", 0))
        end = float(item.get("end", video_duration))
        # Clamp to video bounds
        start = max(0.0, min(start, video_duration))
        end = max(start + 1.0, min(end, video_duration))
        sections.append(make_section(
            title=item.get("title", f"Section {len(sections) + 1}"),
            summary=item.get("summary", ""),
            start=start,
            end=end,
        ))

    if not sections:
        log("  Warning: LLM returned no sections, falling back.")
        return analyse_content_fixed(segments, video_duration, log=log)

    log(f"  Content analysis complete: {len(sections)} sections identified.")
    return sections


# ---------------------------------------------------------------------------
# Content analysis — Fallback modes (no LLM)
# ---------------------------------------------------------------------------

def analyse_content_fixed(segments, video_duration, chunk_minutes=5, log=print):
    """Split video into fixed-duration chunks."""
    chunk_secs = chunk_minutes * 60
    sections = []
    start = 0.0
    idx = 1

    while start < video_duration:
        end = min(start + chunk_secs, video_duration)
        # Find segment texts within this range
        texts = [s["text"] for s in segments
                 if s["start"] >= start and s["end"] <= end]
        summary = " ".join(texts[:3])
        if len(summary) > 150:
            summary = summary[:147] + "..."

        sections.append(make_section(
            title=f"Part {idx}",
            summary=summary or "(no speech detected)",
            start=start,
            end=end,
        ))
        start = end
        idx += 1

    log(f"  Fixed-duration split: {len(sections)} sections "
        f"({chunk_minutes} min each).")
    return sections


def analyse_content_silence(segments, video_duration, min_gap=3.0, log=print):
    """Split at silence gaps longer than min_gap seconds."""
    if not segments:
        return analyse_content_fixed(segments, video_duration, log=log)

    sections = []
    section_start = 0.0
    section_segments = []

    for i, seg in enumerate(segments):
        section_segments.append(seg)
        # Check gap to next segment
        if i + 1 < len(segments):
            gap = segments[i + 1]["start"] - seg["end"]
        else:
            gap = video_duration - seg["end"]

        if gap >= min_gap or i + 1 == len(segments):
            section_end = seg["end"]
            if i + 1 < len(segments) and gap >= min_gap:
                # Place cut in middle of gap
                section_end = seg["end"] + gap / 2

            texts = [s["text"] for s in section_segments[:3]]
            summary = " ".join(texts)
            if len(summary) > 150:
                summary = summary[:147] + "..."

            sections.append(make_section(
                title=f"Section {len(sections) + 1}",
                summary=summary or "(no speech detected)",
                start=section_start,
                end=min(section_end, video_duration),
            ))
            section_start = section_end
            section_segments = []

    if not sections:
        return analyse_content_fixed(segments, video_duration, log=log)

    log(f"  Silence-based split: {len(sections)} sections "
        f"(gap threshold: {min_gap}s).")
    return sections


# ---------------------------------------------------------------------------
# Clip extraction (ffmpeg)
# ---------------------------------------------------------------------------

def extract_clip(video_path, start, end, output_path, ffmpeg_path="ffmpeg",
                 codec="h264", log=print):
    """Cut a clip from the video with frame-accurate re-encoding."""
    duration = end - start
    log(f"  Cutting clip: {start:.1f}s – {end:.1f}s ({duration:.1f}s)")

    cmd = [ffmpeg_path, "-ss", str(start), "-i", video_path,
           "-t", str(duration),
           "-map", "0:v:0", "-map", "0:a:0?"]

    if codec == "copy":
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k"]

    cmd += ["-movflags", "+faststart", "-y", output_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

def process_clips(video_path, sections, target_lang, audio_lang,
                  whisper_model, width, height, ffmpeg_path, ffprobe_path,
                  base_dir=None, on_progress=print_progress,
                  dub_audio=False, voice_gender="male", burn_subs=True,
                  output_codec="h264", cancel_event=None, duration=0):
    """Extract and process all selected sections.

    Returns a list of dicts with keys: title, clip_path, output_path,
    srt_path, ass_path.
    """
    from auto_process import execute_pipeline, decide_actions, get_video_info

    if base_dir is None:
        base_dir = BASE_DIR

    selected = [s for s in sections if s.get("selected", True)]
    if not selected:
        return []

    clips_dir = os.path.join(base_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    results = []
    for i, section in enumerate(selected):
        if cancel_event and cancel_event.is_set():
            from auto_process import CancelledError
            raise CancelledError()

        title_safe = re.sub(r'[^\w\s-]', '', section["title"])[:60].strip()
        title_safe = re.sub(r'\s+', '_', title_safe)
        clip_filename = f"clip_{i + 1:03d}_{title_safe}.mp4"
        clip_path = os.path.join(clips_dir, clip_filename)

        on_progress(ProgressUpdate(
            "execution",
            f"  Extracting clip {i + 1}/{len(selected)}: {section['title']}",
            i / len(selected),
        ))

        # Cut the clip
        extract_clip(video_path, section["start"], section["end"],
                     clip_path, ffmpeg_path, codec="h264",
                     log=lambda m: on_progress(
                         ProgressUpdate("execution", m, -1)))

        # Get clip info for pipeline
        clip_w, clip_h, clip_dur, _ = get_video_info(clip_path, ffprobe_path)
        if clip_w == 0:
            clip_w, clip_h = width, height
        if clip_dur == 0:
            clip_dur = section["end"] - section["start"]

        # Decide actions for this clip
        is_10_9 = abs(clip_w / clip_h - 10 / 9) < 0.02 if clip_h > 0 else False
        actual_ratio = clip_w / clip_h if clip_h > 0 else 1.0
        actions = decide_actions(
            is_10_9, actual_ratio, audio_lang, False,
            target_lang, clip_w, clip_h,
            convert_portrait=False,
            dub_audio=dub_audio,
        )

        on_progress(ProgressUpdate(
            "execution",
            f"  Processing clip {i + 1}/{len(selected)}: {section['title']}",
            (i + 0.5) / len(selected),
        ))

        # Run the processing pipeline on the clip
        try:
            output_path, result_paths = execute_pipeline(
                clip_path, actions, target_lang, audio_lang,
                whisper_model, clip_w, clip_h, ffmpeg_path, ffprobe_path,
                base_dir=base_dir,
                on_progress=lambda u: on_progress(u),
                duration=clip_dur,
                dub_audio=dub_audio,
                voice_gender=voice_gender,
                burn_subs=burn_subs,
                output_codec=output_codec,
                cancel_event=cancel_event,
            )
        except Exception as e:
            on_progress(ProgressUpdate(
                "execution",
                f"  Warning: Processing failed for '{section['title']}': {e}",
                -1,
            ))
            output_path = clip_path
            result_paths = {}

        results.append({
            "title": section["title"],
            "summary": section["summary"],
            "start": section["start"],
            "end": section["end"],
            "clip_path": clip_path,
            "output_path": result_paths.get("output_video", output_path),
            "srt_path": result_paths.get("output_srt"),
            "ass_path": result_paths.get("output_ass"),
        })

    on_progress(ProgressUpdate(
        "execution",
        f"  All {len(results)} clips processed.",
        1.0,
    ))
    return results


# ---------------------------------------------------------------------------
# ZIP packaging
# ---------------------------------------------------------------------------

def create_clips_zip(results, output_path=None, base_dir=None):
    """Package all clip outputs into a ZIP file."""
    if base_dir is None:
        base_dir = BASE_DIR
    if output_path is None:
        output_path = os.path.join(base_dir, "clips", "all_clips.zip")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            video = r.get("output_path")
            if video and os.path.isfile(video):
                zf.write(video, os.path.basename(video))
            srt = r.get("srt_path")
            if srt and os.path.isfile(srt):
                zf.write(srt, os.path.basename(srt))

    return output_path
