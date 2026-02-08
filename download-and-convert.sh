#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <URL> [extra yt-dlp args...]"
    echo ""
    echo "Downloads the video, then converts it to H.264 with 10:9 aspect ratio."
    exit 1
fi

URL="$1"
shift

# --- Download ---
echo "=== Step 1: Downloading ==="
"$SCRIPT_DIR/download.sh" "$URL" "$@"

# --- Find the most recently downloaded file ---
LATEST=$(ls -t "$SCRIPT_DIR/downloads/"*.mp4 2>/dev/null | head -1)

if [[ -z "$LATEST" ]]; then
    echo "Error: No .mp4 file found in downloads/ after download."
    exit 1
fi

echo ""
echo "=== Step 2: Converting to H.264 10:9 ==="
echo "Input: $LATEST"
"$SCRIPT_DIR/convert109.sh" "$LATEST"
