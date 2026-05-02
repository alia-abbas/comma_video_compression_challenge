# Bash script. The evaluator runs this. It unzips archive.zip and triggers inflate.py.
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# These arguments are provided by the evaluator
DATA_DIR="$1"    # Where your model.pt.br etc. are stored
OUTPUT_DIR="$2"  # Where the .raw frames should be written
FILE_LIST="$3"   # The list of videos to reconstruct

mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="python3"

# Trigger the reconstruction logic
"$PYTHON_BIN" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"