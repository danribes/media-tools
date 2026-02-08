#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

show_menu() {
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║           MEDIA TOOLS                ║"
    echo "╠══════════════════════════════════════╣"
    echo "║  1) Download video                   ║"
    echo "║  2) Convert to 10:9                  ║"
    echo "║  3) Generate subtitles               ║"
    echo "║  4) Download + Convert to 10:9       ║"
    echo "║  5) Download + Subtitle              ║"
    echo "║  6) Download + Convert + Subtitle    ║"
    echo "║  7) Auto-process (smart)             ║"
    echo "║  8) Quit                             ║"
    echo "╚══════════════════════════════════════╝"
    echo ""
}

prompt_url() {
    read -rp "Enter URL: " URL
    if [[ -z "$URL" ]]; then
        echo "No URL provided."
        exit 1
    fi
}

prompt_file() {
    local label="${1:-video}"
    read -rp "Enter path to $label file (or drag & drop): " FILE
    # Strip quotes that some terminals add on drag & drop
    FILE="${FILE//\'/}"
    FILE="${FILE//\"/}"
    FILE="${FILE## }"
    FILE="${FILE%% }"
    if [[ ! -f "$FILE" ]]; then
        echo "File not found: $FILE"
        exit 1
    fi
}

prompt_sub_lang() {
    echo ""
    echo "Subtitle language:"
    echo "  1) Spanish"
    echo "  2) English"
    read -rp "Choose [1-2]: " lang_choice
    case "$lang_choice" in
        2)  TGT_LANG="en" ;;
        *)  TGT_LANG="es" ;;
    esac
    SRC_LANG="auto"
    echo "→ Subtitles in: $TGT_LANG (source language: auto-detect)"
}

# Find the latest .mp4 in a directory
latest_mp4() {
    ls -t "$1"/*.mp4 2>/dev/null | head -1
}

do_download() {
    prompt_url
    "$SCRIPT_DIR/download.sh" "$URL"
}

do_convert() {
    prompt_file "input video"
    "$SCRIPT_DIR/convert109.sh" "$FILE"
}

do_subtitle() {
    prompt_file "input video"
    prompt_sub_lang
    local args=()
    [[ "$SRC_LANG" != "auto" ]] && args+=(--lang "$SRC_LANG")
    [[ "$TGT_LANG" != "es" ]] && args+=(--target-lang "$TGT_LANG")
    "$SCRIPT_DIR/subtitles.sh" "$FILE" "${args[@]}"
}

do_download_convert() {
    prompt_url
    "$SCRIPT_DIR/download.sh" "$URL"

    LATEST=$(latest_mp4 "$SCRIPT_DIR/downloads")
    if [[ -z "$LATEST" ]]; then
        echo "Error: No .mp4 found in downloads/ after download."
        exit 1
    fi

    echo ""
    echo "=== Converting to 10:9 ==="
    "$SCRIPT_DIR/convert109.sh" "$LATEST"
}

do_download_subtitle() {
    prompt_url
    prompt_sub_lang
    "$SCRIPT_DIR/download.sh" "$URL"

    LATEST=$(latest_mp4 "$SCRIPT_DIR/downloads")
    if [[ -z "$LATEST" ]]; then
        echo "Error: No .mp4 found in downloads/ after download."
        exit 1
    fi

    echo ""
    echo "=== Generating subtitles ==="
    local args=()
    [[ "$SRC_LANG" != "auto" ]] && args+=(--lang "$SRC_LANG")
    [[ "$TGT_LANG" != "es" ]] && args+=(--target-lang "$TGT_LANG")
    "$SCRIPT_DIR/subtitles.sh" "$LATEST" "${args[@]}"
}

do_auto_process() {
    read -rp "Enter file path (or URL): " INPUT
    # Strip quotes that some terminals add on drag & drop
    INPUT="${INPUT//\'/}"
    INPUT="${INPUT//\"/}"
    INPUT="${INPUT## }"
    INPUT="${INPUT%% }"

    if [[ -z "$INPUT" ]]; then
        echo "No input provided."
        return
    fi

    source "$SCRIPT_DIR/venv/bin/activate"

    local args=("$INPUT")
    read -rp "Dry run only? (y/N): " dry
    if [[ "$dry" =~ ^[Yy]$ ]]; then
        args+=(--dry-run)
    fi

    python3 "$SCRIPT_DIR/scripts/auto_process.py" "${args[@]}"
}

do_download_convert_subtitle() {
    prompt_url
    prompt_sub_lang
    "$SCRIPT_DIR/download.sh" "$URL"

    LATEST=$(latest_mp4 "$SCRIPT_DIR/downloads")
    if [[ -z "$LATEST" ]]; then
        echo "Error: No .mp4 found in downloads/ after download."
        exit 1
    fi

    echo ""
    echo "=== Converting to 10:9 ==="
    "$SCRIPT_DIR/convert109.sh" "$LATEST"

    CONVERTED=$(latest_mp4 "$SCRIPT_DIR/converted")
    if [[ -z "$CONVERTED" ]]; then
        echo "Error: No .mp4 found in converted/ after conversion."
        exit 1
    fi

    echo ""
    echo "=== Generating subtitles ==="
    local args=()
    [[ "$SRC_LANG" != "auto" ]] && args+=(--lang "$SRC_LANG")
    [[ "$TGT_LANG" != "es" ]] && args+=(--target-lang "$TGT_LANG")
    "$SCRIPT_DIR/subtitles.sh" "$CONVERTED" "${args[@]}"
}

# --- Main loop ---
while true; do
    show_menu
    read -rp "Choose an option [1-8]: " choice

    case "$choice" in
        1) do_download ;;
        2) do_convert ;;
        3) do_subtitle ;;
        4) do_download_convert ;;
        5) do_download_subtitle ;;
        6) do_download_convert_subtitle ;;
        7) do_auto_process ;;
        8) echo "Bye!"; exit 0 ;;
        *) echo "Invalid option." ;;
    esac

    echo ""
    echo "--- Done ---"
done
