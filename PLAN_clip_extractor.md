# Feature Plan: Video Content Analysis & Clip Extraction

## Overview

Analyse a video's full transcription, segment it into thematic sections using an
LLM, let the user review/edit the sections, then extract individual clips with
the user's chosen output format (aspect ratio, subtitles, dubbing).

---

## 1. Pipeline Architecture

### Phase 1 — Transcribe & Analyse (backend, automatic)

1. **Download / receive** the video (existing `download_url` or file upload).
2. **Transcribe** the full audio with Whisper (`transcribe_audio`), producing
   timestamped segments.  Already implemented — reuse as-is.
3. **Topic segmentation** — send the full transcription (with timestamps) to an
   LLM (Claude API) with a prompt like:

   > Given the following timestamped transcript, group consecutive segments into
   > coherent thematic sections.  For each section return: `title`, `summary`,
   > `start` (seconds), `end` (seconds).  Respond in JSON.

   This produces a list of **sections**, each spanning one or more Whisper
   segments.

   *Why an LLM?*  Keyword/embedding clustering struggles with natural speech.
   An LLM understands topic shifts, speaker changes, and narrative arcs
   natively.  The transcript of even a 4-hour video is well within a 200 k
   context window.

   *Fallback:* If no API key is configured, offer a simple split mode:
   fixed-duration chunks (e.g. every N minutes) or split on detected silence
   gaps > X seconds.

### Phase 2 — Review & Select (frontend, interactive)

The user sees the list of sections and can:
- **Toggle** individual sections on/off for extraction.
- **Edit** title, start and end times (fine-tune cuts).
- **Merge** adjacent sections or **split** a section at a given timestamp.
- **Preview** a 3-second thumbnail/clip for each section.
- **Choose output format** per clip (or globally):
  - Aspect ratio: Original / 10:9 / 9:16 (portrait)
  - Subtitles: None / Burned-in / SRT download only
  - Dubbing: None / Dub only / Subs + Dub
  - Target language (inherits from sidebar, overridable per clip)

### Phase 3 — Extract & Process (backend, batch)

For each selected section:
1. **ffmpeg cut** the source video from `start` to `end` (keyframe-accurate
   with re-encoding, or fast copy if codec allows).
2. **Run the existing processing pipeline** (`execute_pipeline`) on the cut
   clip, applying the chosen format (convert, subtitles, dub).
3. Collect all output files for download.

---

## 2. Backend Implementation

### New module: `scripts/clip_extractor.py`

```
analyse_content(video_path, segments, api_key=None) -> list[Section]
    Send transcript to LLM, return sections.
    Fallback: fixed-duration or silence-based splits.

extract_clip(video_path, start, end, output_path, ffmpeg_path) -> str
    ffmpeg cut with re-encode for frame-accurate cuts.

process_clips(video_path, sections, settings, ...) -> list[dict]
    Orchestrator: for each section, extract clip then run execute_pipeline.
    Supports cancel_event, on_progress for UI feedback.
```

### Data structures

```python
@dataclass
class Section:
    title: str
    summary: str
    start: float        # seconds
    end: float          # seconds
    selected: bool = True
    # Per-section output overrides (None = inherit global)
    target_lang: str | None = None
    burn_subs: bool | None = None
    dub_audio: bool | None = None
    convert_format: str | None = None   # "original", "10:9", "9:16"
```

### LLM integration

- Use the Claude API (`anthropic` SDK) or support an `OPENAI_API_KEY` env var
  as alternative.
- The prompt sends only the text + timestamps (not audio), so token cost is
  low even for long videos (~15 k tokens for 4 hours of speech).
- Response is parsed as JSON; validation ensures start/end stay within video
  bounds and sections don't overlap.

### Clip extraction (ffmpeg)

```bash
ffmpeg -ss {start} -to {end} -i input.mp4 \
       -c:v libx264 -c:a aac -movflags +faststart \
       -y clip_001.mp4
```

- `-ss` before `-i` for fast seek; re-encode to guarantee frame-accurate cuts.
- For `codec=copy` mode, use `-ss` after `-i` with `-avoid_negative_ts make_zero`.

---

## 3. Streamlit Frontend

### UI Flow

The feature is accessed via a **new tab** alongside the existing "URL" and
"Upload" tabs:

```
[URL]  [Upload]  [Clip Extractor]
```

Or alternatively, as a **post-processing action** after a video is loaded:
once the video is downloaded and assessed, show a button
**"Analyse & Extract Clips"** in the results area.

**Recommended approach: post-processing action.**  This avoids duplicating the
input UI and naturally follows the existing workflow.

### Detailed UI Layout

#### Step 1 — Analysis (automatic after button click)

```
┌─────────────────────────────────────────────────┐
│  [Analyse & Extract Clips]   (button)           │
│                                                 │
│  Progress: Transcribing... 45%                  │
│  Progress: Analysing content... (LLM)           │
└─────────────────────────────────────────────────┘
```

#### Step 2 — Section Review (interactive)

```
┌─────────────────────────────────────────────────┐
│  Sections found: 8                              │
│                                                 │
│  Global output settings:                        │
│  Format: [Original ▾]  Subs: [Burned-in ▾]     │
│  Dub: [None ▾]  Language: [Spanish ▾]           │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │ ☑  1. Introduction          00:00–02:35  │   │
│  │    "Speaker introduces the topic of..."  │   │
│  │    [Edit] [Split] [Preview]              │   │
│  ├──────────────────────────────────────────┤   │
│  │ ☑  2. Economic Analysis     02:35–08:12  │   │
│  │    "Discussion of GDP growth figures..." │   │
│  │    [Edit] [Split] [Preview]              │   │
│  ├──────────────────────────────────────────┤   │
│  │ ☐  3. Q&A Session           08:12–12:40  │   │
│  │    "Audience questions about..."         │   │
│  │    [Edit] [Split] [Preview]              │   │
│  └──────────────────────────────────────────┘   │
│                                                 │
│  [Select All] [Deselect All]                    │
│  [Extract Selected Clips]  (button)             │
└─────────────────────────────────────────────────┘
```

Each section row is rendered with `st.expander` or a card-like layout using
`st.container`.  The Edit mode (on expand) shows:

```
┌──────────────────────────────────────────────┐
│  Title: [Economic Analysis          ]        │
│  Start: [00:02:35]   End: [00:08:12]         │
│  Summary: [Discussion of GDP growth...]      │
│  Override format: [Use global ▾]             │
│  [Save] [Cancel]                             │
└──────────────────────────────────────────────┘
```

#### Step 3 — Extraction Progress

```
┌─────────────────────────────────────────────────┐
│  Extracting clips: 2/5                          │
│  Current: "Economic Analysis" — Dubbing... 60%  │
│  ████████████░░░░░░░░  60%                      │
│  [Cancel]                                       │
└─────────────────────────────────────────────────┘
```

#### Step 4 — Results

```
┌─────────────────────────────────────────────────┐
│  Clips ready: 5                                 │
│                                                 │
│  1. Introduction.mp4          [▶ Play] [⬇ Download] │
│  2. Economic Analysis.mp4     [▶ Play] [⬇ Download] │
│  3. Key Findings.mp4          [▶ Play] [⬇ Download] │
│  ...                                            │
│                                                 │
│  [⬇ Download All (ZIP)]                         │
└─────────────────────────────────────────────────┘
```

### Session State

```python
st.session_state.clip_sections      # list[Section] from LLM analysis
st.session_state.clip_processing    # bool — extraction in progress
st.session_state.clip_results       # list[dict] — output paths per clip
st.session_state.clip_transcript    # list[dict] — raw Whisper segments
```

---

## 4. Implementation Order

| Phase | Task                                    | Effort |
|-------|-----------------------------------------|--------|
| 1     | `clip_extractor.py` — LLM segmentation  | Medium |
| 2     | `clip_extractor.py` — ffmpeg clip cut    | Small  |
| 3     | `clip_extractor.py` — batch orchestrator | Medium |
| 4     | `app.py` — "Analyse" button + progress  | Small  |
| 5     | `app.py` — section review UI            | Large  |
| 6     | `app.py` — extraction progress + results| Medium |
| 7     | `app.py` — ZIP download                 | Small  |
| 8     | Translations (ES/EN)                    | Small  |
| 9     | Docker: add `anthropic` to requirements | Tiny   |

---

## 5. Dependencies

- **`anthropic`** Python SDK (or alternative LLM provider) — for content
  analysis.  Add to `requirements.txt`.
- **API key** — stored as env var `ANTHROPIC_API_KEY` in `docker-compose.yml`
  or entered via sidebar input.
- All other dependencies (ffmpeg, Whisper, edge-tts) already present.

---

## 6. Configuration (Sidebar additions)

```
── Clip Extractor ──
API Key: [••••••••••]   (password input, stored in session)
Fallback mode: [Fixed duration ▾]  (if no API key)
Chunk duration: [5 min]            (for fallback mode)
```

---

## 7. Edge Cases & Considerations

- **Very long videos (4+ hours):** Whisper transcription may take 30+ min.
  Show progress.  The transcript text itself fits easily in LLM context
  (~15 k tokens for 4 hours).
- **No speech segments:** If Whisper returns no segments, show a warning
  and offer the fixed-duration fallback.
- **Overlapping sections:** Validate LLM output; merge or reject overlaps.
- **Concurrency:** Use the existing `_processing_lock` to prevent parallel
  extraction jobs.
- **Disk space:** Multiple clips from a long video can use significant space.
  Show estimated size before extraction; clean up temp files after ZIP
  download.
- **Cancellation:** Pass `cancel_event` through the batch orchestrator so
  the user can abort mid-extraction.
