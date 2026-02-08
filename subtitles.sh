#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input_video> [options]"
    echo ""
    echo "Auto-generates Spanish subtitles and burns them into the video."
    echo "Detects existing subtitle regions and places new subs in free space."
    echo ""
    echo "Options:"
    echo "  -o, --output FILE     Output video file (default: subbed/<name>_subtitled.mp4)"
    echo "  --lang LANG           Force source language (default: auto-detect)"
    echo "  --target-lang LANG    Target language (default: es)"
    echo "  --model SIZE          Whisper model: tiny/base/small/medium/large (default: small)"
    echo "  --font-size N         Override subtitle font size"
    echo "  --position top|bottom Force subtitle position (default: auto-detect)"
    echo "  --no-ocr              Skip OCR detection, use Whisper directly"
    echo "  --ocr-sync            Auto-accept OCR sync (non-interactive)"
    echo "  --srt-only            Only generate SRT/ASS files, don't burn in"
    exit 1
fi

python3 "$SCRIPT_DIR/scripts/subtitle_gen.py" "$@"
