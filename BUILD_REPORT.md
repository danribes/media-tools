# Build Report: Streamlit Web UI + Docker for media-tools

## Goal

Replace terminal-based interaction with the auto-process pipeline with a web UI. Users can paste a URL or upload a file, see progress, preview results, and download output files. The existing CLI continues to work unchanged.

## Approach: Refactor with Callback-Based Progress

Refactored `auto_process.py` to expose a callable `process_video()` function with a progress callback, instead of only working as a CLI script. This gives structured data for the UI and clean progress streaming while preserving full backward compatibility.

---

## Files Created

### 1. `scripts/progress.py`
A small module defining the progress contract between the pipeline and any consumer (CLI, web UI, etc.).

- `ProgressUpdate` dataclass with fields: `phase`, `message`, `percent`, `detail`
- `ProgressCallback` type alias
- `print_progress()` — default callback that simply prints messages (preserves CLI behavior)

### 2. `requirements.txt`
Pinned all project dependencies in one place:
- `faster-whisper`, `deep-translator`, `pytesseract`, `Pillow`, `numpy`, `yt-dlp`, `langdetect`
- Added `streamlit` for the web UI

### 3. `app.py` — Streamlit Web UI
~190 lines. Layout:

- **Sidebar**: target language selector (9 languages), Whisper model size, dry-run checkbox
- **Main area**: two tabs — URL input and file upload
- **Progress**: `st.status` container with live updates while processing runs
- **Results**: assessment metrics (dimensions, duration, audio language, burned-in subs), execution plan expander, full log expander, `st.video` player, and download buttons for video/SRT/ASS

Progress streaming works via threading:
- Processing runs in a daemon thread that pushes `ProgressUpdate` objects to `st.session_state.progress_log`
- Main thread polls every 0.5s inside an `st.status` container, writing new messages as they arrive
- On completion, the status container updates to "Complete" or "Error" state

File upload writes the `UploadedFile` to `downloads/`, then processes it as a local file path.

### 4. `Dockerfile`
Based on `python:3.12-slim`:
- Installs `ffmpeg` and `tesseract-ocr` via apt
- Installs Python deps from `requirements.txt`
- Copies application code (`scripts/`, `app.py`, `.streamlit/`, `convert109.sh`)
- Creates `bin/ffmpeg` and `bin/ffprobe` symlinks pointing to `/usr/bin/` (matches existing path resolution used by the Python scripts)
- Creates `downloads/` and `subbed/` directories
- Exposes port 8501, runs `streamlit run app.py`

### 5. `docker-compose.yml`
- Bind mounts `downloads/` and `subbed/` to the host for easy access to output files
- Named volume `whisper-cache` mapped to `~/.cache/huggingface` so the Whisper model persists across container restarts
- Sets `STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500` (MB)
- `restart: unless-stopped`

### 6. `.dockerignore`
Excludes `venv/`, `downloads/`, `subbed/`, `.git/`, `__pycache__/`, `*.pyc`, `.env` from the build context.

### 7. `.streamlit/config.toml`
- `maxUploadSize = 500` (MB)
- `headless = true` (required for Docker/server environments)
- `base = "dark"` theme

---

## Files Modified

### 8. `scripts/subtitle_gen.py`
Minimal changes — added an optional `log=print` parameter to 5 functions that print status messages:

| Function | Print calls changed |
|----------|-------------------|
| `load_whisper_model()` | 1 |
| `transcribe_audio()` | 2 |
| `translate_segments()` | 2 |
| `find_subtitle_placement()` | ~5 |
| `detect_burned_in_subs()` | 3 |

Each `print(...)` became `log(...)`. The default `log=print` means all existing callers (including the CLI `main()` in subtitle_gen.py itself) work identically without changes.

### 9. `scripts/auto_process.py`
The largest change — full refactor of the 627-line file:

**a) Extracted `process_video()` as the public API:**
```python
def process_video(input_source, target_lang="es", model_size="small",
                  dry_run=False, base_dir=None,
                  on_progress=print_progress) -> dict:
```
Returns a structured result dict with: `status`, `output_video`, `output_srt`, `output_ass`, `assessment`, `actions`.

**b) Replaced ~40 `print()` calls with `on_progress(ProgressUpdate(...))`**, passing phase, message, and percent through every stage of the pipeline.

**c) `main()` became a thin CLI wrapper** — just parses args and calls `process_video()` with defaults.

**d) `download_url()` calls yt-dlp directly** instead of shelling out to `download.sh` (which requires venv activation). This makes it work in Docker without bash/venv sourcing. The same yt-dlp flags are preserved.

**e) Error handling**: replaced `sys.exit(1)` calls with exceptions (`RuntimeError`) so the web UI can catch and display errors gracefully. The CLI wrapper in `main()` translates error status to `sys.exit(1)`.

**f) `print_plan()` renamed to `_format_plan()`** — returns a string instead of printing directly, so the caller can route it through the progress callback.

**g) `execute_pipeline()` now returns `(path, paths_dict)`** — the paths dict contains `output_video`, `output_srt`, `output_ass` keys for structured access.

**h) ffmpeg fallback**: `process_video()` checks if `bin/ffmpeg` exists and falls back to system `ffmpeg`/`ffprobe` if not (needed for Docker where ffmpeg is at `/usr/bin/`).

---

## Design Decisions

1. **Callback pattern over subprocess**: Rather than running `auto_process.py` as a subprocess and parsing stdout, we refactored it into a callable function with a typed callback. This gives the UI structured data (phase, percent, detail dict) instead of raw text.

2. **Thread-based progress in Streamlit**: Streamlit reruns the script on every interaction, so long-running work must happen in a thread. The thread writes to `st.session_state` (which persists across reruns), and the main thread polls for new updates.

3. **Backward compatibility**: Every change was additive — new parameters have defaults matching the old behavior (`log=print`, `on_progress=print_progress`, `base_dir=None` → `BASE_DIR`). The CLI `python scripts/auto_process.py <url>` produces identical output.

4. **Docker portability**: The Dockerfile symlinks system ffmpeg into `bin/` so the existing path resolution (`BASE_DIR/bin/ffmpeg`) works. yt-dlp is called directly (not via `download.sh`) to avoid venv activation issues in containers.

---

## Verification Checklist

| Check | How to verify |
|-------|--------------|
| CLI regression | `python scripts/auto_process.py <video_url>` — output identical to before |
| Syntax validity | All 4 Python files pass `python3 -m py_compile` |
| Local Streamlit | `streamlit run app.py` — test URL input + file upload |
| Docker build | `docker compose build` |
| Docker run | `docker compose up` — accessible at http://localhost:8501 |
| End-to-end | Process a video via the web UI, verify output in `subbed/` |
| Model persistence | Restart container, verify Whisper model cache persists (named volume) |
