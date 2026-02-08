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

echo "Source: ${SRC_W}x${SRC_H} (aspect $(echo "scale=4; $SRC_W / $SRC_H" | bc))"

# Target aspect ratio: 10:9
# Compare source AR to 10/9 ≈ 1.1111
# If source is wider  (AR > 10/9): keep width, increase height → new_h = w * 9/10
# If source is narrower (AR < 10/9): keep height, increase width → new_w = h * 10/9

# Use integer math: compare SRC_W * 9 vs SRC_H * 10
WIDE_TEST=$((SRC_W * 9))
NARROW_TEST=$((SRC_H * 10))

# For exact 10:9 with H.264 (even dimensions), both W and H must be multiples
# of 20 and 18 respectively: W=20k, H=18k guarantees W/H = 10/9 exactly.
if [[ $WIDE_TEST -gt $NARROW_TEST ]]; then
    # Source is wider than 10:9 → find smallest k where 20k >= SRC_W
    K=$(( (SRC_W + 19) / 20 ))
    echo "Source is wider than 10:9 → stretching height"
elif [[ $WIDE_TEST -lt $NARROW_TEST ]]; then
    # Source is narrower than 10:9 → find smallest k where 18k >= SRC_H
    K=$(( (SRC_H + 17) / 18 ))
    echo "Source is narrower than 10:9 → stretching width"
else
    # Already 10:9 — find nearest valid pair
    K=$(( (SRC_W + 19) / 20 ))
    echo "Source is already 10:9"
fi

NEW_W=$(( K * 20 ))
NEW_H=$(( K * 18 ))

echo "Output:  ${NEW_W}x${NEW_H} (aspect $(echo "scale=4; $NEW_W / $NEW_H" | bc))"
echo "File:    $OUTPUT"
echo ""

"$FFMPEG" -i "$INPUT" \
    -vf "scale=${NEW_W}:${NEW_H}" \
    -c:v libx264 -preset medium -crf 18 \
    -c:a aac -b:a 192k \
    -movflags +faststart \
    -y \
    "$OUTPUT"

echo ""
echo "Done. Output: $OUTPUT"
echo "Verify: $FFPROBE -v error -show_entries stream=width,height,codec_name \"$OUTPUT\""
