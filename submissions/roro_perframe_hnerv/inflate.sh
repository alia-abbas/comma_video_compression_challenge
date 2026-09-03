#!/usr/bin/env bash
# Convert extracted archive/ -> raw video frames at <output_dir>/<base>.raw
# A .raw is a flat uint8 RGB dump of shape (N, H, W, 3), no header.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/0.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  cd "$ROOT"
  python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
done < "$FILE_LIST"
