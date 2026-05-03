#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Path to the folder containing 0.mkv
DATA_DIR="videos"

# 2. Path to the output folder you created
OUTPUT_DIR="submissions/root/output"

# 3. Path to the txt file that says "0.mkv"
FILE_LIST="public_test_video_names.txt"

mkdir -p "$OUTPUT_DIR"

# Use "python" for Windows/Git Bash compatibility
PYTHON_BIN="python"

"$PYTHON_BIN" "$HERE/compress.py" \
    --video-dir "$DATA_DIR" \
    --video-names "$FILE_LIST" \
    --output-dir "$OUTPUT_DIR"