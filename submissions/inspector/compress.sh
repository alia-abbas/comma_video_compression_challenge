#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="$HERE/../../videos"
OUTPUT_DIR="$HERE/output"
FILE_LIST="$HERE/../../public_test_video_names.txt"

mkdir -p "$OUTPUT_DIR"

echo "Starting compression from: $HERE"

python "$HERE/compress.py" \
    --video-dir   "$DATA_DIR" \
    --video-names "$FILE_LIST" \
    --output-dir  "$OUTPUT_DIR"