# Bash script. The evaluator runs this. It unzips archive.zip and triggers inflate.py.
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="python" 

# Trigger the reconstruction logic
"$PYTHON_BIN" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"