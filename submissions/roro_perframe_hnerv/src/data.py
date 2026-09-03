"""Frame loading for single-video overfitting.

Decodes videos/0.mkv into the exact RGB frames the evaluator sees (same
`yuv420_to_rgb` as frame_utils -> AVVideoDataset), then exposes them grouped
into consecutive pairs (frame 2k, 2k+1).

Pairing semantics that drive the whole design (verified against evaluate.py):
  - frame 2k+1 (the *second* / odd-global frame of each pair) is fed to BOTH
    SegNet (which only looks at x[:, -1]) and PoseNet.
  - frame 2k (the *first* / even-global frame) is fed to PoseNet ONLY.

The evaluator downsamples whatever it gets to segnet_model_input_size before
running the nets, so we train directly at that resolution (EVAL_H, EVAL_W) and
let inflate.py bicubic-upsample back to camera resolution for the .raw.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frame_utils import yuv420_to_rgb, camera_size, segnet_model_input_size  # noqa: E402

CAMERA_W, CAMERA_H = camera_size                # 1164, 874
EVAL_W, EVAL_H = segnet_model_input_size        # 512, 384


def decode_frames(video_path: str, max_frames: int | None = None) -> torch.Tensor:
    """Decode frames as (N, H, W, 3) uint8, matching AVVideoDataset exactly."""
    import av
    fmt = 'hevc' if str(video_path).endswith('.hevc') else None
    container = av.open(str(video_path), format=fmt)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(yuv420_to_rgb(frame))  # (H, W, 3) uint8
        if max_frames is not None and len(frames) >= max_frames:
            break
    container.close()
    return torch.stack(frames)


def to_eval_res(frames_nhwc_u8: torch.Tensor) -> torch.Tensor:
    """(N,H,W,3) uint8 camera-res -> (N,3,EVAL_H,EVAL_W) float32 in [0,255].

    Uses bilinear to match the evaluator's preprocess interpolate. This is the
    resolution at which both nets actually consume the frames, so it's the
    resolution we train and store latents for.
    """
    x = frames_nhwc_u8.permute(0, 3, 1, 2).float()  # (N,3,H,W)
    x = F.interpolate(x, size=(EVAL_H, EVAL_W), mode='bilinear', align_corners=False)
    return x


class PairData:
    """Holds the video as consecutive pairs at eval resolution.

    Attributes
    ----------
    pairs : (n_pairs, 2, 3, EVAL_H, EVAL_W) float32 in [0,255]
        pairs[k, 0] = frame 2k   (even / PoseNet-only)
        pairs[k, 1] = frame 2k+1 (odd  / SegNet + PoseNet)
    """

    def __init__(self, video_path: str, device: torch.device = torch.device('cpu'),
                 max_frames: int | None = None, cache: str | None = None):
        if cache is not None and Path(cache).exists():
            eval_frames = torch.load(cache)         # (N,3,EVAL_H,EVAL_W) float32
        else:
            full = decode_frames(video_path, max_frames=max_frames)  # (N,H,W,3) u8
            eval_frames = to_eval_res(full)         # (N,3,EVAL_H,EVAL_W)
            if cache is not None:
                torch.save(eval_frames, cache)
        n = eval_frames.shape[0]
        n -= n % 2                                  # drop a trailing unpaired frame if any
        self.n_frames = n
        self.n_pairs = n // 2
        self.pairs = eval_frames[:n].view(self.n_pairs, 2, 3, EVAL_H, EVAL_W).to(device)
        self.device = device

    def batch(self, idx):
        """idx: LongTensor of pair indices -> (B,2,3,EVAL_H,EVAL_W)."""
        return self.pairs[idx]


if __name__ == "__main__":
    # smoke test: decode, check shapes + that our decode matches frame_utils path
    import time
    mf = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.time()
    pd = PairData(str(ROOT / 'videos' / '0.mkv'), max_frames=mf)
    print(f"decoded {pd.n_frames} frames -> {pd.n_pairs} pairs in {time.time()-t0:.1f}s")
    print("pairs shape:", tuple(pd.pairs.shape), pd.pairs.dtype,
          "range", pd.pairs.min().item(), pd.pairs.max().item())
