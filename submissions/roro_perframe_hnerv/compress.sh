#!/usr/bin/env bash
# Train the QLR decoder + per-pair latents from scratch, then build archive.zip.
#
# Designed for a single NVIDIA GPU (e.g. RTX 4090). Typical run: a few hours.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

TAG="${TAG:-prod}"
EPOCHS="${EPOCHS:-3000}"
RANK="${RANK:-16}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
LATENT_DIM="${LATENT_DIM:-20}"
QAT_BITS="${QAT_BITS:-6}"
BATCH="${BATCH:-32}"
W_PIX="${W_PIX:-0.02}"
W_POSE="${W_POSE:-2000}"
W_SEG="${W_SEG:-1.0}"
LR="${LR:-2e-3}"
SEG_LOSS="${SEG_LOSS:-ce}"
OPTIMIZER="${OPTIMIZER:-adamw}"
POSE_LOSS="${POSE_LOSS:-mse}"
EMA="${EMA:-0.0}"
PRUNE="${PRUNE:-0.0}"
QAT_START="${QAT_START:-400}"
RESUME_FLAG=""
[ "${RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"
INIT_FLAG=""
[ -n "${INIT_FROM:-}" ] && INIT_FLAG="--init-from ${INIT_FROM}"

cd "$ROOT"
export PYTHONIOENCODING=utf-8
python -u -m submissions.roro_perframe_hnerv.src.train \
  --tag "$TAG" --epochs "$EPOCHS" --rank "$RANK" \
  --base-channels "$BASE_CHANNELS" --latent-dim "$LATENT_DIM" \
  --qat-bits "$QAT_BITS" --batch "$BATCH" --lr "$LR" \
  --w-pix "$W_PIX" --w-pose "$W_POSE" --w-seg "$W_SEG" --qat-start-epoch "$QAT_START" \
  --seg-loss "$SEG_LOSS" --optimizer "$OPTIMIZER" --pose-loss "$POSE_LOSS" --ema "$EMA" --prune "$PRUNE" \
  --use-cache $RESUME_FLAG $INIT_FLAG

ARCHIVE_BIN="${HERE}/src/ckpts/run_${TAG}/submission_archive/0.bin"
if [ ! -f "$ARCHIVE_BIN" ]; then
  echo "ERROR: $ARCHIVE_BIN not found" >&2
  exit 1
fi

cd "$(dirname "$ARCHIVE_BIN")"
rm -f "$HERE/archive.zip"
zip -j "$HERE/archive.zip" "0.bin"
echo "Wrote $HERE/archive.zip ($(stat -c%s "$HERE/archive.zip" 2>/dev/null || stat -f%z "$HERE/archive.zip") bytes)"
