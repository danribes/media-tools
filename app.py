#!/usr/bin/env python3
"""
Streamlit web UI for media-tools auto-processing pipeline.

Run: streamlit run app.py
"""

import os
import re
import sys
import threading
import time

import streamlit as st

# Allow importing from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from progress import ProgressUpdate, format_time
from auto_process import process_video, is_url, CancelledError
from subtitle_gen import UnsupportedLanguageError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Module-level lock to prevent multiple simultaneous processing sessions.
# Protects against concurrent users corrupting shared output files.
_processing_lock = threading.Lock()

TRANSLATIONS = {
    "en": {
        "caption": "Auto-process videos: download, convert, subtitle, translate",
        "settings": "Settings",
        "target_language": "Target language",
        "lang_spanish": "Spanish (es)",
        "lang_english": "English (en)",
        "lang_french": "French (fr)",
        "lang_german": "German (de)",
        "lang_portuguese": "Portuguese (pt)",
        "lang_italian": "Italian (it)",
        "lang_japanese": "Japanese (ja)",
        "lang_korean": "Korean (ko)",
        "lang_chinese": "Chinese (zh)",
        "lang_russian": "Russian (ru)",
        "lang_arabic": "Arabic (ar)",
        "lang_hindi": "Hindi (hi)",
        "output_mode": "Output mode",
        "output_subs_only": "Subtitles only",
        "output_dub_only": "Audio dub only",
        "output_subs_dub": "Subtitles + Audio dub",
        "output_no_subs_dub": "No subtitles / No dub",
        "voice": "Voice",
        "voice_male": "Male",
        "voice_female": "Female",
        "whisper_model": "Whisper model",
        "convert_portrait": "Convert portrait to 10:9",
        "dry_run": "Dry run (assess only)",
        "tab_url": "URL",
        "tab_upload": "Upload file",
        "paste_url": "Paste a video URL",
        "invalid_url": "Enter a valid URL starting with http:// or https://",
        "upload_video": "Upload a video file",
        "saved_to": "Saved to {path}",
        "process": "Process",
        "provide_input": "Provide a URL or upload a file first.",
        "remaining": "~{time} remaining",
        "estimating": "estimating...",
        "step_progress": "Step {step}/{total}: {name}  \u2014  {elapsed} elapsed, {remaining}",
        "starting": "Starting...",
        "processing_log": "Processing log",
        "unsupported_lang": "The detected audio language **'{lang}'** is not supported by the translation service. This video cannot be subtitled.",
        "continue_no_subs": "Continue without subtitles",
        "stop": "Stop",
        "processing_failed": "Processing failed: {error}",
        "dimensions": "Dimensions",
        "duration": "Duration",
        "audio": "Audio",
        "burned_in_subs": "Burned-in subs",
        "yes": "Yes",
        "no": "No",
        "execution_plan": "Execution plan",
        "dry_run_complete": "Dry run complete \u2014 no files were produced.",
        "result": "Result",
        "download_video": "Download video",
        "download_srt": "Download SRT",
        "download_ass": "Download ASS",
        "no_output": "No output video \u2014 the video may have needed no processing.",
        "browser_cookies": "Browser cookies",
        "browser_none": "None",
        "cookies_file": "Cookies file (.txt)",
        "output_codec": "Output codec",
        "codec_h264": "H.264",
        "codec_copy": "Original (copy)",
        "cancel": "Cancel",
        "cancelled": "Processing was cancelled.",
        "already_processing": "Another video is already being processed. Please wait for it to finish.",
    },
    "es": {
        "caption": "Procesar videos: descargar, convertir, subtitular, traducir",
        "settings": "Configuraci\u00f3n",
        "target_language": "Idioma destino",
        "lang_spanish": "Espa\u00f1ol (es)",
        "lang_english": "Ingl\u00e9s (en)",
        "lang_french": "Franc\u00e9s (fr)",
        "lang_german": "Alem\u00e1n (de)",
        "lang_portuguese": "Portugu\u00e9s (pt)",
        "lang_italian": "Italiano (it)",
        "lang_japanese": "Japon\u00e9s (ja)",
        "lang_korean": "Coreano (ko)",
        "lang_chinese": "Chino (zh)",
        "lang_russian": "Ruso (ru)",
        "lang_arabic": "\u00c1rabe (ar)",
        "lang_hindi": "Hindi (hi)",
        "output_mode": "Modo de salida",
        "output_subs_only": "Solo subt\u00edtulos",
        "output_dub_only": "Solo doblaje",
        "output_subs_dub": "Subt\u00edtulos + Doblaje",
        "output_no_subs_dub": "Sin subt\u00edtulos / Sin doblaje",
        "voice": "Voz",
        "voice_male": "Masculina",
        "voice_female": "Femenina",
        "whisper_model": "Modelo Whisper",
        "convert_portrait": "Convertir vertical a 10:9",
        "dry_run": "Simulaci\u00f3n (solo evaluar)",
        "tab_url": "URL",
        "tab_upload": "Subir archivo",
        "paste_url": "Pegar una URL de video",
        "invalid_url": "Ingrese una URL v\u00e1lida que comience con http:// o https://",
        "upload_video": "Subir un archivo de video",
        "saved_to": "Guardado en {path}",
        "process": "Procesar",
        "provide_input": "Proporcione una URL o suba un archivo primero.",
        "remaining": "~{time} restante",
        "estimating": "estimando...",
        "step_progress": "Paso {step}/{total}: {name}  \u2014  {elapsed} transcurrido, {remaining}",
        "starting": "Iniciando...",
        "processing_log": "Registro de procesamiento",
        "unsupported_lang": "El idioma de audio detectado **'{lang}'** no es compatible con el servicio de traducci\u00f3n. Este video no se puede subtitular.",
        "continue_no_subs": "Continuar sin subt\u00edtulos",
        "stop": "Detener",
        "processing_failed": "Error en el procesamiento: {error}",
        "dimensions": "Dimensiones",
        "duration": "Duraci\u00f3n",
        "audio": "Audio",
        "burned_in_subs": "Subt\u00edtulos incrustados",
        "yes": "S\u00ed",
        "no": "No",
        "execution_plan": "Plan de ejecuci\u00f3n",
        "dry_run_complete": "Simulaci\u00f3n completa \u2014 no se generaron archivos.",
        "result": "Resultado",
        "download_video": "Descargar video",
        "download_srt": "Descargar SRT",
        "download_ass": "Descargar ASS",
        "no_output": "Sin video de salida \u2014 el video puede no haber necesitado procesamiento.",
        "browser_cookies": "Cookies del navegador",
        "browser_none": "Ninguno",
        "cookies_file": "Archivo de cookies (.txt)",
        "output_codec": "Codec de salida",
        "codec_h264": "H.264",
        "codec_copy": "Original (copiar)",
        "cancel": "Cancelar",
        "cancelled": "El procesamiento fue cancelado.",
        "already_processing": "Otro video ya se est\u00e1 procesando. Espere a que termine.",
    },
}


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "processing": False,
        "result": None,
        "error": None,
        "cancelled": False,
        "unsupported_lang": None,  # set when translator doesn't support detected lang
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
        "unsupported_lang": None,
        "cancelled": False,
        "cancel_event": threading.Event(),
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
                    convert_portrait=True, dub_audio=False, voice_gender="male",
                    burn_subs=True, cookies_browser=None, cookies_file=None,
                    output_codec="h264"):
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
            cookies_browser=cookies_browser,
            cookies_file=cookies_file,
            dub_audio=dub_audio,
            voice_gender=voice_gender,
            burn_subs=burn_subs,
            cancel_event=shared["cancel_event"],
            output_codec=output_codec,
        )
        shared["result"] = result
    except CancelledError:
        shared["cancelled"] = True
    except UnsupportedLanguageError as e:
        shared["unsupported_lang"] = e.lang
    except Exception as e:
        shared["error"] = str(e)
    finally:
        shared["processing"] = False
        _processing_lock.release()


def main():
    st.set_page_config(page_title="Media Tools", page_icon="\U0001f3ac", layout="wide")
    init_session_state()

    # --- Language selector (before any translated content) ---
    ui_lang = st.sidebar.radio("Language / Idioma", ["Espa\u00f1ol", "English"],
                               index=0, horizontal=True)
    lang = "es" if ui_lang == "Espa\u00f1ol" else "en"
    t = TRANSLATIONS[lang]

    st.title("Media Tools")
    st.caption(t["caption"])

    # --- Sidebar settings ---
    with st.sidebar:
        st.header(t["settings"])

        _lang_keys = [
            ("es", "lang_spanish"), ("en", "lang_english"),
            ("fr", "lang_french"), ("de", "lang_german"),
            ("pt", "lang_portuguese"), ("it", "lang_italian"),
            ("ja", "lang_japanese"), ("ko", "lang_korean"),
            ("zh", "lang_chinese"), ("ru", "lang_russian"),
            ("ar", "lang_arabic"), ("hi", "lang_hindi"),
        ]
        _lang_codes = [code for code, _ in _lang_keys]
        target_lang_options = [t[key] for _, key in _lang_keys]
        target_lang_mode = st.selectbox(t["target_language"], target_lang_options, index=0)
        target_lang = _lang_codes[target_lang_options.index(target_lang_mode)]

        output_mode_options = [
            t["output_subs_only"],
            t["output_dub_only"],
            t["output_subs_dub"],
            t["output_no_subs_dub"],
        ]
        output_mode = st.radio(t["output_mode"], output_mode_options)
        output_mode_idx = output_mode_options.index(output_mode)
        dub_audio = output_mode_idx in (1, 2)
        burn_subs = output_mode_idx in (0, 2)

        if output_mode_idx == 3:
            target_lang = None

        if dub_audio:
            voice_options = [t["voice_male"], t["voice_female"]]
            voice_selection = st.radio(t["voice"], voice_options)
            voice_gender = "male" if voice_options.index(voice_selection) == 0 else "female"
        else:
            voice_gender = "male"

        if target_lang is not None:
            model_size = st.selectbox(t["whisper_model"], [
                "tiny", "base", "small", "medium", "large",
            ], index=2)
        else:
            model_size = "small"

        convert_portrait = st.checkbox(t["convert_portrait"], value=True)

        codec_options = [t["codec_copy"], t["codec_h264"]]
        codec_selection = st.radio(t["output_codec"], codec_options, index=0)
        output_codec = "copy" if codec_options.index(codec_selection) == 0 else "h264"

        dry_run = st.checkbox(t["dry_run"], value=False)

        browser_options = [t["browser_none"], "chrome", "firefox", "edge", "brave", "chromium"]
        cookies_browser = st.selectbox(t["browser_cookies"], browser_options, index=0)
        if cookies_browser == t["browser_none"]:
            cookies_browser = None

        cookies_file = None
        uploaded_cookies = st.file_uploader(t["cookies_file"], type=["txt"])
        if uploaded_cookies is not None:
            cookies_path = os.path.join(BASE_DIR, "cookies.txt")
            with open(cookies_path, "wb") as f:
                f.write(uploaded_cookies.getbuffer())
            cookies_file = cookies_path

    # --- Input tabs ---
    tab_url, tab_upload = st.tabs([t["tab_url"], t["tab_upload"]])

    input_source = None

    with tab_url:
        url = st.text_input(t["paste_url"],
                            placeholder="https://x.com/user/status/123456...")
        if url:
            # Strip whitespace and trailing non-URL junk (e.g. " ---")
            cleaned = re.sub(r'[\s\-]+$', '', url.strip())
            if is_url(cleaned):
                input_source = cleaned
            else:
                st.warning(t["invalid_url"])

    with tab_upload:
        uploaded = st.file_uploader(t["upload_video"], type=["mp4", "mov", "mkv", "webm"])
        if uploaded is not None:
            # Save uploaded file to downloads/
            downloads_dir = os.path.join(BASE_DIR, "downloads")
            os.makedirs(downloads_dir, exist_ok=True)
            save_path = os.path.join(downloads_dir, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            input_source = save_path
            st.success(t["saved_to"].format(path=save_path))

    # --- Process button ---
    if st.button(t["process"], type="primary", disabled=st.session_state.processing):
        if input_source is None:
            st.error(t["provide_input"])
        elif not _processing_lock.acquire(blocking=False):
            st.warning(t["already_processing"])
        else:
            # Reset state
            st.session_state.result = None
            st.session_state.error = None
            st.session_state.cancelled = False
            st.session_state.unsupported_lang = None
            st.session_state.progress_log = []
            st.session_state.processing = True

            shared = _make_shared()
            st.session_state.shared = shared

            thread = threading.Thread(
                target=_run_processing,
                args=(shared, input_source, target_lang, model_size, dry_run,
                      convert_portrait, dub_audio, voice_gender, burn_subs,
                      cookies_browser, cookies_file, output_codec),
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
            st.session_state.cancelled = shared.get("cancelled", False)
            st.session_state.unsupported_lang = shared.get("unsupported_lang")
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
                remaining_str = t["remaining"].format(time=format_time(adjusted_remaining))
            else:
                remaining_str = t["estimating"]
            text = t["step_progress"].format(
                step=step["step"], total=step["total"], name=step["name"],
                elapsed=elapsed_str, remaining=remaining_str,
            )
            st.progress(min(step.get("percent", 0), 0.99), text=text)
        else:
            st.progress(0, text=t["starting"])

        # Cancel button
        if st.button(t["cancel"], type="secondary"):
            shared["cancel_event"].set()

        # Show processing log so far
        log = shared["progress_log"]
        detail_msgs = [u.message.strip() for u in log
                       if u.phase != "step" and u.message.strip()]
        if detail_msgs:
            with st.expander(t["processing_log"], expanded=True):
                for msg in detail_msgs:
                    st.text(msg)

        # Poll again after a short delay
        time.sleep(0.5)
        st.rerun()

    # --- Cancelled display ---
    if st.session_state.cancelled:
        st.warning(t["cancelled"])
        st.session_state.cancelled = False

    # --- Unsupported language prompt ---
    if st.session_state.unsupported_lang:
        detected_lang = st.session_state.unsupported_lang
        st.warning(t["unsupported_lang"].format(lang=detected_lang))
        col_continue, col_stop, _ = st.columns([1, 1, 3])
        with col_continue:
            if st.button(t["continue_no_subs"]):
                if not _processing_lock.acquire(blocking=False):
                    st.warning(t["already_processing"])
                else:
                    st.session_state.unsupported_lang = None
                    st.session_state.error = None
                    st.session_state.progress_log = []
                    st.session_state.processing = True

                    shared = _make_shared()
                    st.session_state.shared = shared

                    thread = threading.Thread(
                        target=_run_processing,
                        args=(shared, input_source, None, model_size, dry_run,
                              convert_portrait, False, "male", False),
                        daemon=True,
                    )
                    thread.start()
                    st.rerun()
        with col_stop:
            if st.button(t["stop"]):
                st.session_state.unsupported_lang = None

        with st.expander(t["processing_log"], expanded=False):
            for update in st.session_state.progress_log:
                if update.phase == "step":
                    continue
                msg = update.message.strip()
                if msg:
                    st.text(msg)

    # --- Error display ---
    elif st.session_state.error:
        st.error(t["processing_failed"].format(error=st.session_state.error))

        with st.expander(t["processing_log"], expanded=False):
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
            col1.metric(t["dimensions"], f"{assessment.get('width', '?')}x{assessment.get('height', '?')}")
            col2.metric(t["duration"], f"{assessment.get('duration', 0):.0f}s")
            col3.metric(t["audio"], assessment.get("audio_lang", "?"))
            col4.metric(t["burned_in_subs"], t["yes"] if assessment.get("has_burned_subs") else t["no"])

        # Show actions
        actions = result.get("actions", [])
        if actions:
            with st.expander(t["execution_plan"], expanded=False):
                for action in actions:
                    st.write(f"**{action['type']}**: {action['reason']}")

        # Show full log
        with st.expander(t["processing_log"], expanded=False):
            for update in st.session_state.progress_log:
                if update.phase == "step":
                    continue
                msg = update.message.strip()
                if msg:
                    st.text(msg)

        if result["status"] == "dry_run":
            st.info(t["dry_run_complete"])
            return

        # --- Video preview + downloads ---
        video_path = result.get("output_video")
        srt_path = result.get("output_srt")
        ass_path = result.get("output_ass")

        if video_path and os.path.isfile(video_path):
            st.subheader(t["result"])
            st.video(video_path)

            # Download buttons
            cols = st.columns(3)
            with cols[0]:
                with open(video_path, "rb") as f:
                    st.download_button(
                        t["download_video"],
                        f.read(),
                        file_name=os.path.basename(video_path),
                        mime="video/mp4",
                    )

            if srt_path and os.path.isfile(srt_path):
                with cols[1]:
                    with open(srt_path, "r") as f:
                        st.download_button(
                            t["download_srt"],
                            f.read(),
                            file_name=os.path.basename(srt_path),
                            mime="text/plain",
                        )

            if ass_path and os.path.isfile(ass_path):
                with cols[2]:
                    with open(ass_path, "r") as f:
                        st.download_button(
                            t["download_ass"],
                            f.read(),
                            file_name=os.path.basename(ass_path),
                            mime="text/plain",
                        )
        elif not video_path:
            st.info(t["no_output"])


if __name__ == "__main__":
    main()
