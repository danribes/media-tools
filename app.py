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
from subtitle_gen import UnsupportedLanguageError, transcribe_audio, load_whisper_model
from clip_extractor import (
    analyse_content_llm, analyse_content_fixed, analyse_content_silence,
    process_clips, create_clips_zip,
)

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
        # Clip extractor
        "clip_analyse": "Analyse & Extract Clips",
        "clip_api_key": "Anthropic API Key",
        "clip_split_mode": "Split mode",
        "clip_split_llm": "AI content analysis",
        "clip_split_fixed": "Fixed duration",
        "clip_split_silence": "Silence gaps",
        "clip_chunk_minutes": "Chunk duration (min)",
        "clip_silence_gap": "Min silence gap (sec)",
        "clip_analysing": "Analysing content...",
        "clip_sections_found": "Sections found: {count}",
        "clip_global_settings": "Clip output settings",
        "clip_select_all": "Select all",
        "clip_deselect_all": "Deselect all",
        "clip_extract": "Extract selected clips",
        "clip_extracting": "Extracting clip {current}/{total}: {title}",
        "clip_done": "Clips ready: {count}",
        "clip_download_all": "Download all (ZIP)",
        "clip_no_sections": "No sections found.",
        "clip_no_selected": "No sections selected.",
        "clip_preview": "Preview",
        "clip_edit": "Edit",
        "clip_title": "Title",
        "clip_start": "Start (sec)",
        "clip_end": "End (sec)",
        "clip_summary": "Summary",
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
        # Clip extractor
        "clip_analyse": "Analizar y extraer clips",
        "clip_api_key": "API Key de Anthropic",
        "clip_split_mode": "Modo de segmentaci\u00f3n",
        "clip_split_llm": "An\u00e1lisis de contenido con IA",
        "clip_split_fixed": "Duraci\u00f3n fija",
        "clip_split_silence": "Pausas de silencio",
        "clip_chunk_minutes": "Duraci\u00f3n del segmento (min)",
        "clip_silence_gap": "Pausa m\u00ednima de silencio (seg)",
        "clip_analysing": "Analizando contenido...",
        "clip_sections_found": "Secciones encontradas: {count}",
        "clip_global_settings": "Configuraci\u00f3n de clips",
        "clip_select_all": "Seleccionar todo",
        "clip_deselect_all": "Deseleccionar todo",
        "clip_extract": "Extraer clips seleccionados",
        "clip_extracting": "Extrayendo clip {current}/{total}: {title}",
        "clip_done": "Clips listos: {count}",
        "clip_download_all": "Descargar todo (ZIP)",
        "clip_no_sections": "No se encontraron secciones.",
        "clip_no_selected": "Ninguna secci\u00f3n seleccionada.",
        "clip_preview": "Vista previa",
        "clip_edit": "Editar",
        "clip_title": "T\u00edtulo",
        "clip_start": "Inicio (seg)",
        "clip_end": "Fin (seg)",
        "clip_summary": "Resumen",
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
        # Clip extractor state
        "clip_sections": None,       # list[dict] from analysis
        "clip_transcript": None,     # list[dict] raw Whisper segments
        "clip_results": None,        # list[dict] extraction results
        "clip_processing": False,
        "clip_error": None,
        "clip_video_path": None,
        "clip_audio_lang": None,
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
            # Only write if the file doesn't already match (avoids overwriting
            # during Streamlit reruns while ffmpeg is reading the file).
            need_write = (
                not os.path.isfile(save_path)
                or os.path.getsize(save_path) != uploaded.size
            )
            if need_write:
                tmp_path = save_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                os.replace(tmp_path, save_path)  # atomic on same filesystem
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

        # === Clip Extractor Section ===
        _render_clip_extractor(t, result, target_lang, model_size,
                               dub_audio, voice_gender, burn_subs,
                               output_codec)


# ---------------------------------------------------------------------------
# Clip extractor UI
# ---------------------------------------------------------------------------

def _run_clip_analysis(shared, video_path, audio_lang, model_size, api_key,
                       split_mode, chunk_minutes, silence_gap):
    """Background thread: transcribe + analyse content."""
    try:
        log_fn = _make_callback(shared)
        log_fn(ProgressUpdate("execution", "Transcribing audio for analysis...", 0.1))

        model = load_whisper_model(model_size)
        segments, detected = transcribe_audio(
            video_path, audio_lang, model_size, model=model,
            log=lambda m: log_fn(ProgressUpdate("execution", m, -1)),
        )
        shared["transcript"] = segments
        shared["audio_lang"] = detected

        # Get video duration
        from subtitle_gen import get_video_info
        ffprobe = os.path.join(BASE_DIR, "bin", "ffprobe")
        if not os.path.isfile(ffprobe):
            ffprobe = "ffprobe"
        _, _, duration, _ = get_video_info(video_path, ffprobe)

        log_fn(ProgressUpdate("execution", "Segmenting content...", 0.7))

        if split_mode == "llm" and api_key:
            sections = analyse_content_llm(
                segments, api_key, duration,
                log=lambda m: log_fn(ProgressUpdate("execution", m, -1)),
            )
        elif split_mode == "silence":
            sections = analyse_content_silence(
                segments, duration, min_gap=silence_gap,
                log=lambda m: log_fn(ProgressUpdate("execution", m, -1)),
            )
        else:
            sections = analyse_content_fixed(
                segments, duration, chunk_minutes=chunk_minutes,
                log=lambda m: log_fn(ProgressUpdate("execution", m, -1)),
            )

        shared["sections"] = sections
        log_fn(ProgressUpdate("execution", f"Analysis complete: {len(sections)} sections.", 1.0))
    except CancelledError:
        shared["cancelled"] = True
    except Exception as e:
        shared["error"] = str(e)
    finally:
        shared["processing"] = False


def _run_clip_extraction(shared, video_path, sections, target_lang, audio_lang,
                         model_size, dub_audio, voice_gender, burn_subs,
                         output_codec):
    """Background thread: extract and process clips."""
    try:
        log_fn = _make_callback(shared)
        ffmpeg = os.path.join(BASE_DIR, "bin", "ffmpeg")
        ffprobe = os.path.join(BASE_DIR, "bin", "ffprobe")
        if not os.path.isfile(ffmpeg):
            ffmpeg = "ffmpeg"
        if not os.path.isfile(ffprobe):
            ffprobe = "ffprobe"

        from subtitle_gen import get_video_info
        _, _, duration, _ = get_video_info(video_path, ffprobe)
        w, h = 0, 0
        try:
            w, h, _, _ = get_video_info(video_path, ffprobe)
        except Exception:
            pass

        results = process_clips(
            video_path, sections,
            target_lang=target_lang,
            audio_lang=audio_lang,
            whisper_model=model_size,
            width=w, height=h,
            ffmpeg_path=ffmpeg, ffprobe_path=ffprobe,
            base_dir=BASE_DIR,
            on_progress=log_fn,
            dub_audio=dub_audio,
            voice_gender=voice_gender,
            burn_subs=burn_subs,
            output_codec=output_codec,
            cancel_event=shared["cancel_event"],
            duration=duration,
        )
        shared["clip_results"] = results
    except CancelledError:
        shared["cancelled"] = True
    except Exception as e:
        shared["error"] = str(e)
    finally:
        shared["processing"] = False


def _render_clip_extractor(t, result, target_lang, model_size,
                           dub_audio, voice_gender, burn_subs,
                           output_codec):
    """Render the clip extractor UI within the results area."""

    st.divider()
    st.subheader(t["clip_analyse"])

    # --- Settings row ---
    col_mode, col_key = st.columns([1, 2])
    with col_mode:
        split_options = [t["clip_split_llm"], t["clip_split_fixed"],
                         t["clip_split_silence"]]
        split_mode_label = st.selectbox(t["clip_split_mode"], split_options,
                                        index=0)
        split_mode_map = {
            t["clip_split_llm"]: "llm",
            t["clip_split_fixed"]: "fixed",
            t["clip_split_silence"]: "silence",
        }
        split_mode = split_mode_map[split_mode_label]

    with col_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if split_mode == "llm":
            api_key_input = st.text_input(
                t["clip_api_key"], value=api_key, type="password",
            )
            if api_key_input:
                api_key = api_key_input

    # Extra settings for non-LLM modes
    chunk_minutes = 5
    silence_gap = 3.0
    if split_mode == "fixed":
        chunk_minutes = st.slider(t["clip_chunk_minutes"], 1, 30, 5)
    elif split_mode == "silence":
        silence_gap = st.slider(t["clip_silence_gap"], 1.0, 15.0, 3.0, 0.5)

    # Source video path from the processed result
    video_path = result.get("output_video") or st.session_state.get("clip_video_path")
    # Use original video if available (better quality for clip extraction)
    assessment = result.get("assessment", {})
    audio_lang = assessment.get("audio_lang", "en")

    # If the result has a source video path, prefer it
    source_video = None
    for update in st.session_state.progress_log:
        if update.phase == "download" and "Downloaded:" in update.message:
            m = re.search(r'Downloaded:\s+(.+\.mp4)', update.message)
            if m and os.path.isfile(m.group(1)):
                source_video = m.group(1)
    if not source_video:
        source_video = video_path

    if not source_video or not os.path.isfile(source_video):
        return

    # --- Analyse button ---
    if st.session_state.clip_sections is None and not st.session_state.clip_processing:
        if st.button(t["clip_analyse"], key="clip_analyse_btn", type="primary"):
            if split_mode == "llm" and not api_key:
                st.error("API key required for AI content analysis.")
                return
            if not _processing_lock.acquire(blocking=False):
                st.warning(t["already_processing"])
                return

            st.session_state.clip_processing = True
            shared = _make_shared()
            shared["transcript"] = None
            shared["sections"] = None
            st.session_state.shared = shared
            st.session_state.clip_video_path = source_video

            thread = threading.Thread(
                target=_run_clip_analysis,
                args=(shared, source_video, audio_lang, model_size,
                      api_key, split_mode, chunk_minutes, silence_gap),
                daemon=True,
            )
            thread.start()
            st.rerun()

    # --- Analysis in progress ---
    if st.session_state.clip_processing:
        shared = st.session_state.shared
        if shared and not shared["processing"]:
            # Analysis finished
            if shared.get("error"):
                st.session_state.clip_error = shared["error"]
            elif shared.get("sections"):
                st.session_state.clip_sections = shared["sections"]
                st.session_state.clip_transcript = shared.get("transcript")
                st.session_state.clip_audio_lang = shared.get("audio_lang", audio_lang)
            st.session_state.clip_processing = False
            st.session_state.shared = None
            _processing_lock.release()
            st.rerun()
        else:
            st.progress(0.5, text=t["clip_analysing"])
            if st.button(t["cancel"], key="clip_cancel_analyse"):
                if shared:
                    shared["cancel_event"].set()
            time.sleep(1)
            st.rerun()
        return

    # --- Error display ---
    if st.session_state.clip_error:
        st.error(t["processing_failed"].format(error=st.session_state.clip_error))
        st.session_state.clip_error = None

    # --- Section review ---
    sections = st.session_state.clip_sections
    if sections is not None and st.session_state.clip_results is None:
        st.success(t["clip_sections_found"].format(count=len(sections)))

        # Select/deselect all
        col_sel, col_desel, _ = st.columns([1, 1, 4])
        with col_sel:
            if st.button(t["clip_select_all"]):
                for s in sections:
                    s["selected"] = True
                st.rerun()
        with col_desel:
            if st.button(t["clip_deselect_all"]):
                for s in sections:
                    s["selected"] = False
                st.rerun()

        # Render each section
        for idx, section in enumerate(sections):
            duration_str = f"{section['start']:.0f}s – {section['end']:.0f}s"
            dur_secs = section['end'] - section['start']
            label = f"{section['title']}  ({duration_str}, {dur_secs:.0f}s)"

            col_check, col_info = st.columns([0.05, 0.95])
            with col_check:
                checked = st.checkbox(
                    "", value=section.get("selected", True),
                    key=f"clip_sel_{idx}", label_visibility="collapsed",
                )
                section["selected"] = checked
            with col_info:
                with st.expander(label, expanded=False):
                    st.write(section.get("summary", ""))
                    # Inline editing
                    new_title = st.text_input(
                        t["clip_title"], value=section["title"],
                        key=f"clip_title_{idx}",
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        new_start = st.number_input(
                            t["clip_start"], value=section["start"],
                            min_value=0.0, step=1.0, key=f"clip_start_{idx}",
                        )
                    with c2:
                        new_end = st.number_input(
                            t["clip_end"], value=section["end"],
                            min_value=0.0, step=1.0, key=f"clip_end_{idx}",
                        )
                    section["title"] = new_title
                    section["start"] = new_start
                    section["end"] = new_end

        # --- Extract button ---
        selected_count = sum(1 for s in sections if s.get("selected"))
        if selected_count == 0:
            st.info(t["clip_no_selected"])
        else:
            if st.button(t["clip_extract"], type="primary", key="clip_extract_btn"):
                if not _processing_lock.acquire(blocking=False):
                    st.warning(t["already_processing"])
                    return

                st.session_state.clip_processing = True
                shared = _make_shared()
                shared["clip_results"] = None
                st.session_state.shared = shared

                clip_video = st.session_state.clip_video_path or source_video
                clip_audio_lang = st.session_state.clip_audio_lang or audio_lang

                thread = threading.Thread(
                    target=_run_clip_extraction,
                    args=(shared, clip_video, sections, target_lang,
                          clip_audio_lang, model_size, dub_audio,
                          voice_gender, burn_subs, output_codec),
                    daemon=True,
                )
                thread.start()
                st.rerun()

    # --- Extraction in progress ---
    if st.session_state.clip_processing and st.session_state.clip_sections is not None:
        shared = st.session_state.shared
        if shared and not shared["processing"]:
            if shared.get("error"):
                st.session_state.clip_error = shared["error"]
            elif shared.get("clip_results"):
                st.session_state.clip_results = shared["clip_results"]
            st.session_state.clip_processing = False
            st.session_state.shared = None
            _processing_lock.release()
            st.rerun()
        else:
            log = shared["progress_log"] if shared else []
            # Show latest execution message
            exec_msgs = [u.message.strip() for u in log
                         if u.phase == "execution" and u.message.strip()]
            if exec_msgs:
                st.progress(0.5, text=exec_msgs[-1])
            else:
                st.progress(0.3, text=t["clip_extracting"].format(
                    current="?", total="?", title="..."))
            if st.button(t["cancel"], key="clip_cancel_extract"):
                if shared:
                    shared["cancel_event"].set()
            time.sleep(1)
            st.rerun()
        return

    # --- Clip results ---
    clip_results = st.session_state.clip_results
    if clip_results:
        st.success(t["clip_done"].format(count=len(clip_results)))

        for i, cr in enumerate(clip_results):
            video_file = cr.get("output_path")
            if video_file and os.path.isfile(video_file):
                with st.expander(f"{i + 1}. {cr['title']}", expanded=False):
                    st.video(video_file)
                    dl_cols = st.columns(3)
                    with dl_cols[0]:
                        with open(video_file, "rb") as f:
                            st.download_button(
                                t["download_video"],
                                f.read(),
                                file_name=os.path.basename(video_file),
                                mime="video/mp4",
                                key=f"clip_dl_video_{i}",
                            )
                    srt_file = cr.get("srt_path")
                    if srt_file and os.path.isfile(srt_file):
                        with dl_cols[1]:
                            with open(srt_file, "r") as f:
                                st.download_button(
                                    t["download_srt"],
                                    f.read(),
                                    file_name=os.path.basename(srt_file),
                                    mime="text/plain",
                                    key=f"clip_dl_srt_{i}",
                                )

        # Download all as ZIP
        if len(clip_results) > 1:
            zip_path = create_clips_zip(clip_results, base_dir=BASE_DIR)
            if os.path.isfile(zip_path):
                with open(zip_path, "rb") as f:
                    st.download_button(
                        t["clip_download_all"],
                        f.read(),
                        file_name="clips.zip",
                        mime="application/zip",
                        key="clip_dl_zip",
                    )

        # Reset button to analyse again
        if st.button(t["clip_analyse"], key="clip_reanalyse_btn"):
            st.session_state.clip_sections = None
            st.session_state.clip_results = None
            st.session_state.clip_transcript = None
            st.session_state.clip_error = None
            st.rerun()


if __name__ == "__main__":
    main()
