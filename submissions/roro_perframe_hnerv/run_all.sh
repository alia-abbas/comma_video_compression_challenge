#!/usr/bin/env bash
# Fire-and-forget: warm cache -> train -> zip archive -> evaluate -> mark DONE.
# Designed to be launched detached (tmux or nohup) so you can disconnect/sleep.
#
#   tmux new -s run                                  # recommended
#   bash submissions/roro_perframe_hnerv/run_all.sh            # then Ctrl-b d to detach
#   # ...or...
#   nohup bash submissions/roro_perframe_hnerv/run_all.sh >/dev/null 2>&1 &
#
# Knobs (env): TAG, EPOCHS, RANK, BASE_CHANNELS, LATENT_DIM, QAT_BITS, BATCH, FAST, RESUME
#   FAST=1  -> batch 96 (≈2x faster wall-clock on a 4090; tiny model fits easily)
#
# When it finishes, these exist in submissions/roro_perframe_hnerv/:
#   archive.zip            <- your submission
#   report.txt             <- the score
#   DONE                   <- empty marker file (run is complete)
#   run_all.log            <- full log
set -uo pipefail   # NOT -e: a late-stage failure must not hide an already-built archive

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LOG="$HERE/run_all.log"

TAG="${TAG:-prod}"
EPOCHS="${EPOCHS:-3000}"
BATCH="${BATCH:-32}"
[ "${FAST:-0}" = "1" ] && BATCH=96

rm -f "$HERE/DONE"
# tee all output to the log AND the terminal
exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "=== run_all START  tag=$TAG epochs=$EPOCHS batch=$BATCH ==="
echo "================================================================"
cd "$ROOT"
export PYTHONIOENCODING=utf-8

# ---- 1. warm the frame cache (idempotent; skipped if already cached) --------
echo "[1/3] warming frame cache ..."
PYTHONPATH="$ROOT" python -u -c "
from submissions.roro_perframe_hnerv.src.data import PairData, ROOT
PairData(str(ROOT/'videos'/'0.mkv'), cache='submissions/roro_perframe_hnerv/cache/eval_frames.pt')
print('cache ready')
" || { echo 'FATAL: cache warm failed'; exit 1; }

# ---- 2. train + package archive.zip -----------------------------------------
echo "[2/3] training ($EPOCHS epochs, batch $BATCH) ..."
TAG="$TAG" EPOCHS="$EPOCHS" RANK="${RANK:-16}" BASE_CHANNELS="${BASE_CHANNELS:-32}" \
  LATENT_DIM="${LATENT_DIM:-20}" QAT_BITS="${QAT_BITS:-6}" RESUME="${RESUME:-0}" \
  BATCH="$BATCH" bash "$HERE/compress.sh"
if [ ! -f "$HERE/archive.zip" ]; then
  echo "FATAL: archive.zip was not produced — check the log above."
  exit 1
fi
echo "archive.zip ready: $(stat -c%s "$HERE/archive.zip") bytes"

# ---- 3. evaluate (non-fatal: archive is already safe if this stumbles) ------
echo "[3/3] evaluating on GPU ..."
bash "$ROOT/evaluate.sh" --submission-dir "$HERE" --device cuda \
  || echo "WARNING: evaluation failed, but archive.zip is built and valid."

echo "================================================================"
echo "=== run_all DONE  $( [ -f "$HERE/report.txt" ] && grep -h 'Final score' "$HERE/report.txt" )"
echo "================================================================"
touch "$HERE/DONE"

# ---- optional: stop the pod to halt GPU billing while you sleep -------------
# AUTOSTOP=1 stops (NOT terminates) the pod. /workspace + archive.zip survive;
# restart the pod when you wake to download. Needs RunPod's preinstalled
# runpodctl + the injected RUNPOD_POD_ID (present on RunPod pods by default).
if [ "${AUTOSTOP:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "AUTOSTOP=1 -> stopping pod $RUNPOD_POD_ID in 20s (Ctrl-C to cancel) ..."
  sleep 20
  runpodctl stop pod "$RUNPOD_POD_ID" || echo "runpodctl stop failed; stop the pod manually."
fi
