#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Evaluator passes three positional args:
ARCHIVE_DIR="$1"   # directory containing archive.br
OUTPUT_DIR="$2"    # where to write 0.raw
FILE_LIST="$3"     # e.g. public_test_video_names.txt

mkdir -p "$OUTPUT_DIR"

python "$HERE/inflate.py" "$ARCHIVE_DIR" "$OUTPUT_DIR" "$FILE_LIST"