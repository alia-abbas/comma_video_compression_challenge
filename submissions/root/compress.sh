# Environment setup (3090-only). Installs k-quants or custom ops.
#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="$1"    # Path to the videos (e.g., videos/)
OUTPUT_DIR="$2"  # Path to save the compressed .br files
FILE_LIST="$3"   # The .txt file containing names like 0.mkv

mkdir -p "$OUTPUT_DIR"

# Standardize python call
PYTHON_BIN="python3"

# Run your new compression engine
"$PYTHON_BIN" "$HERE/compress.py" \
    --video-dir "$DATA_DIR" \
    --video-names "$FILE_LIST" \
    --output-dir "$OUTPUT_DIR"