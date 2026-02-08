#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONVERTED_DIR="$SCRIPT_DIR/converted"
FFMPEG="$SCRIPT_DIR/bin/ffmpeg"
FFPROBE="$SCRIPT_DIR/bin/ffprobe"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input_video> [output_file]"
    echo ""
    echo "Pixel-stretches the video to a 10:9 aspect ratio and re-encodes"
    echo "with H.264 (libx264) + AAC. Output goes to converted/ by default."
    exit 1
fi

INPUT="$1"
BASENAME="$(basename "${INPUT%.*}")"

if [[ $# -ge 2 ]]; then
    OUTPUT="$2"
else
    OUTPUT="$CONVERTED_DIR/${BASENAME}_10x9.mp4"
fi

# Get source dimensions
SRC_W=$("$FFPROBE" -v error -select_streams v:0 \
    -show_entries stream=width -of csv=p=0 "$INPUT")
SRC_H=$("$FFPROBE" -v error -select_streams v:0 \
    -show_entries stream=height -of csv=p=0 "$INPUT")

echo "Source: ${SRC_W}x${SRC_H} (pixel ratio $(echo "scale=4; $SRC_W / $SRC_H" | bc))"

# Target: 10:9 pixel aspect ratio, horizontal stretch only (keep original height).
# New width = height * 10 / 9, rounded to nearest even number for H.264.
NEW_H=$SRC_H
NEW_W=$(( (SRC_H * 10 + 4) / 9 ))
# Ensure even width
NEW_W=$(( (NEW_W + 1) / 2 * 2 ))

if [[ $((SRC_W * 9)) -eq $((SRC_H * 10)) ]]; then
    echo "Source is already 10:9 in pixels"
else
    echo "Stretching horizontally: ${SRC_W} → ${NEW_W} pixels (height unchanged)"
fi

echo "Output:  ${NEW_W}x${NEW_H} (pixel ratio $(echo "scale=4; $NEW_W / $NEW_H" | bc))"
echo "File:    $OUTPUT"
echo ""

"$FFMPEG" -i "$INPUT" \
    -vf "scale=${NEW_W}:${NEW_H},setsar=1" \
    -c:v libx264 -preset medium -crf 18 \
    -c:a aac -b:a 192k \
    -movflags +faststart \
    -y \
    "$OUTPUT"

echo ""
echo "Done. Output: $OUTPUT"
echo "Verify: $FFPROBE -v error -show_entries stream=width,height,codec_name \"$OUTPUT\""
