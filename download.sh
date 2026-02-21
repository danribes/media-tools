#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOWNLOADS_DIR="$SCRIPT_DIR/downloads"
FFMPEG="$SCRIPT_DIR/bin/ffmpeg"
VENV="$SCRIPT_DIR/venv/bin/activate"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <URL> [extra yt-dlp args...]"
    exit 1
fi

URL="$1"
shift

source "$VENV"

echo "Downloading: $URL"
echo "Output dir:  $DOWNLOADS_DIR"
echo ""

yt-dlp \
    --ffmpeg-location "$SCRIPT_DIR/bin" \
    -f "bv*+ba/b" \
    --merge-output-format mp4 \
    -o "$DOWNLOADS_DIR/%(title).80s [%(id)s].%(ext)s" \
    --no-playlist \
    "$@" \
    "$URL"

echo ""
echo "Done. Files saved to $DOWNLOADS_DIR"
