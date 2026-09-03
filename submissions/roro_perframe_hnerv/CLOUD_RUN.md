# Running QLR-HNeRV on a rented GPU (RunPod)

End-to-end recipe to produce `archive.zip` on a single RTX 4090. Total wall-clock
~2–4 h; cost ~$2–5.

## 0. Provision
- GPU: **1× RTX 4090 (24 GB)**, Community Cloud, **On-Demand**.
- Image: official **RunPod PyTorch 2.x** (CUDA ≥ 12.6).
- Container disk: **40 GB**. Expose SSH + Jupyter.

## 1. One-time setup (paste in the pod terminal)
```bash
apt-get update && apt-get install -y git git-lfs ffmpeg zip unzip curl
git lfs install
git clone https://github.com/commaai/comma_video_compression_challenge.git
cd comma_video_compression_challenge
git lfs pull                                  # pulls videos/0.mkv + models/*

# our submission files travel with this repo; if you cloned the upstream repo,
# copy the submissions/roro_perframe_hnerv/ folder in (scp or git) before continuing.

curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync --group cu128                          # if driver too old: --group cu126
source .venv/bin/activate
python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Warm the frame cache (decodes 0.mkv once, ~30 s on GPU box)
```bash
PYTHONPATH=. python -c "
from submissions.roro_perframe_hnerv.src.data import PairData, ROOT
PairData(str(ROOT/'videos'/'0.mkv'), cache='submissions/roro_perframe_hnerv/cache/eval_frames.pt')
print('cache ready')"
```

## 3. Train + package (the actual run)
```bash
# defaults: rank=16, base_channels=32, latent_dim=20, INT6 QAT, 3000 epochs
TAG=prod EPOCHS=3000 bash submissions/roro_perframe_hnerv/compress.sh
```
This writes `submissions/roro_perframe_hnerv/archive.zip`. The script prints the archive
size and the `25*rate` contribution as it finishes.

## 4. Evaluate on the box (sanity-check the score before submitting)
```bash
bash evaluate.sh --submission-dir ./submissions/roro_perframe_hnerv --device cuda
cat submissions/roro_perframe_hnerv/report.txt
```

## 5. Download `archive.zip`, then **Stop the pod**.

---

## Sweeping the rate–distortion frontier (recommended)
The single most informative experiment: vary decoder capacity and watch score.
Run a few tags in parallel-ish and compare `report.txt`:
```bash
TAG=r12 RANK=12 BASE_CHANNELS=24 EPOCHS=3000 bash submissions/roro_perframe_hnerv/compress.sh
TAG=r16 RANK=16 BASE_CHANNELS=32 EPOCHS=3000 bash submissions/roro_perframe_hnerv/compress.sh
TAG=r24 RANK=24 BASE_CHANNELS=40 EPOCHS=3000 bash submissions/roro_perframe_hnerv/compress.sh
# also sweep QAT_BITS=5 vs 6 on the best capacity.
```
Pick the (capacity, bits) with the lowest **Final score**. That plot is the
core of the write-up.

## Resuming after a stop / interruption
Checkpoints land in `submissions/roro_perframe_hnerv/src/ckpts/run_<TAG>/ckpt.pt` every
`--log-every` epochs. To continue an interrupted run, re-run the same command
with `RESUME=1`:
```bash
TAG=prod EPOCHS=3000 RESUME=1 bash submissions/roro_perframe_hnerv/compress.sh
```
It reloads decoder + latents + LR schedule position and continues to `EPOCHS`.
