#!/usr/bin/env python3
"""
Streamlit web UI for media-tools auto-processing pipeline.

Run: streamlit run app.py
"""

import os
import sys
import threading
import time

import streamlit as st

# Allow importing from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from progress import ProgressUpdate, format_time
from auto_process import process_video, is_url

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "processing": False,
        "result": None,
        "error": None,
        "progress_log": [],
        # Plain dict shared with background thread.  st.session_state is
        # thread-local in Streamlit so the worker writes here instead.
        "shared": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _make_shared():
    """Create the plain dict used for cross-thread communication."""
    return {
        "progress_log": [],
        "step_info": None,
        "processing": True,
        "result": None,
        "error": None,
        "start_time": time.time(),
    }


def _make_callback(shared):
    """Return a progress callback that writes to the shared dict."""
    def progress_callback(update: ProgressUpdate):
        if update.phase == "step":
            shared["step_info"] = {
                "name": update.message,
                "percent": update.percent,
                **update.detail,
            }
        shared["progress_log"].append(update)
    return progress_callback


def _run_processing(shared, input_source, target_lang, model_size, dry_run,
                    convert_portrait=True):
    """Run process_video in a thread, storing result/error in shared dict."""
    try:
        result = process_video(
            input_source,
            target_lang=target_lang,
            model_size=model_size,
            dry_run=dry_run,
            base_dir=BASE_DIR,
            on_progress=_make_callback(shared),
            convert_portrait=convert_portrait,
        )
        shared["result"] = result
    except Exception as e:
        shared["error"] = str(e)
    finally:
        shared["processing"] = False


def main():
    st.set_page_config(page_title="Media Tools", page_icon="🎬", layout="wide")
    init_session_state()

    st.title("Media Tools")
    st.caption("Auto-process videos: download, convert, subtitle, translate")

    # --- Sidebar settings ---
    with st.sidebar:
        st.header("Settings")
        target_lang = st.selectbox("Target language", [
            ("es", "Spanish"),
            ("en", "English"),
            ("pt", "Portuguese"),
            ("fr", "French"),
            ("de", "German"),
            ("it", "Italian"),
            ("ja", "Japanese"),
            ("ko", "Korean"),
            ("zh", "Chinese"),
        ], format_func=lambda x: f"{x[1]} ({x[0]})", index=0)[0]

        model_size = st.selectbox("Whisper model", [
            "tiny", "base", "small", "medium", "large",
        ], index=2)

        convert_portrait = st.checkbox("Convert portrait to 10:9", value=True)

        dry_run = st.checkbox("Dry run (assess only)", value=False)

    # --- Input tabs ---
    tab_url, tab_upload = st.tabs(["URL", "Upload file"])

    input_source = None

    with tab_url:
        url = st.text_input("Paste a video URL",
                            placeholder="https://x.com/user/status/123456...")
        if url and is_url(url.strip()):
            input_source = url.strip()
        elif url:
            st.warning("Enter a valid URL starting with http:// or https://")

    with tab_upload:
        uploaded = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv", "webm"])
        if uploaded is not None:
            # Save uploaded file to downloads/
            downloads_dir = os.path.join(BASE_DIR, "downloads")
            os.makedirs(downloads_dir, exist_ok=True)
            save_path = os.path.join(downloads_dir, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            input_source = save_path
            st.success(f"Saved to {save_path}")

    # --- Process button ---
    if st.button("Process", type="primary", disabled=st.session_state.processing):
        if input_source is None:
            st.error("Provide a URL or upload a file first.")
        else:
            # Reset state
            st.session_state.result = None
            st.session_state.error = None
            st.session_state.progress_log = []
            st.session_state.processing = True

            shared = _make_shared()
            st.session_state.shared = shared

            thread = threading.Thread(
                target=_run_processing,
                args=(shared, input_source, target_lang, model_size, dry_run,
                      convert_portrait),
                daemon=True,
            )
            thread.start()
            st.rerun()

    # --- Progress display (rerun-based polling) ---
    shared = st.session_state.shared
    if st.session_state.processing and shared:
        # Check if the background thread has finished
        if not shared["processing"]:
            # Transfer results to session state and do a final rerun
            st.session_state.result = shared["result"]
            st.session_state.error = shared["error"]
            st.session_state.progress_log = list(shared["progress_log"])
            st.session_state.processing = False
            st.session_state.shared = None
            st.rerun()

        # -- Render current progress --
        step = shared["step_info"]
        if step:
            real_elapsed = time.time() - shared["start_time"]
            elapsed_str = format_time(real_elapsed)
            if step["remaining"] >= 0:
                time_since_update = real_elapsed - step["elapsed"]
                adjusted_remaining = max(0, step["remaining"] - time_since_update)
                remaining_str = f"~{format_time(adjusted_remaining)} remaining"
            else:
                remaining_str = "estimating..."
            text = (f"Step {step['step']}/{step['total']}: "
                    f"{step['name']}  \u2014  "
                    f"{elapsed_str} elapsed, {remaining_str}")
            st.progress(min(step.get("percent", 0), 0.99), text=text)
        else:
            st.progress(0, text="Starting...")

        # Show processing log so far
        log = shared["progress_log"]
        detail_msgs = [u.message.strip() for u in log
                       if u.phase != "step" and u.message.strip()]
        if detail_msgs:
            with st.expander("Processing log", expanded=True):
                for msg in detail_msgs:
                    st.text(msg)

        # Poll again after a short delay
        time.sleep(0.5)
        st.rerun()

    # --- Error display ---
    if st.session_state.error:
        st.error(f"Processing failed: {st.session_state.error}")

        with st.expander("Processing log", expanded=False):
            for update in st.session_state.progress_log:
                if update.phase == "step":
                    continue
                msg = update.message.strip()
                if msg:
                    st.text(msg)

    # --- Results display ---
    result = st.session_state.result
    if result and result["status"] in ("completed", "dry_run"):
        # Show assessment summary
        assessment = result.get("assessment", {})
        if assessment:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Dimensions", f"{assessment.get('width', '?')}x{assessment.get('height', '?')}")
            col2.metric("Duration", f"{assessment.get('duration', 0):.0f}s")
            col3.metric("Audio", assessment.get("audio_lang", "?"))
            col4.metric("Burned-in subs", "Yes" if assessment.get("has_burned_subs") else "No")

        # Show actions
        actions = result.get("actions", [])
        if actions:
            with st.expander("Execution plan", expanded=False):
                for action in actions:
                    st.write(f"**{action['type']}**: {action['reason']}")

        # Show full log
        with st.expander("Processing log", expanded=False):
            for update in st.session_state.progress_log:
                if update.phase == "step":
                    continue
                msg = update.message.strip()
                if msg:
                    st.text(msg)

        if result["status"] == "dry_run":
            st.info("Dry run complete — no files were produced.")
            return

        # --- Video preview + downloads ---
        video_path = result.get("output_video")
        srt_path = result.get("output_srt")
        ass_path = result.get("output_ass")

        if video_path and os.path.isfile(video_path):
            st.subheader("Result")
            st.video(video_path)

            # Download buttons
            cols = st.columns(3)
            with cols[0]:
                with open(video_path, "rb") as f:
                    st.download_button(
                        "Download video",
                        f.read(),
                        file_name=os.path.basename(video_path),
                        mime="video/mp4",
                    )

            if srt_path and os.path.isfile(srt_path):
                with cols[1]:
                    with open(srt_path, "r") as f:
                        st.download_button(
                            "Download SRT",
                            f.read(),
                            file_name=os.path.basename(srt_path),
                            mime="text/plain",
                        )

            if ass_path and os.path.isfile(ass_path):
                with cols[2]:
                    with open(ass_path, "r") as f:
                        st.download_button(
                            "Download ASS",
                            f.read(),
                            file_name=os.path.basename(ass_path),
                            mime="text/plain",
                        )
        elif not video_path:
            st.info("No output video — the video may have needed no processing.")


if __name__ == "__main__":
    main()
