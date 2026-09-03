#!/usr/bin/env python
"""Inflate archive (0.bin) -> raw uint8 RGB frames (N, H, W, 3) at camera res.

Parses the QLR decoder + per-pair latents, runs the decoder at eval resolution
(384x512), bicubic-upsamples each frame to the camera resolution required by the
eval harness (874x1164), and writes contiguous uint8 NHWC bytes.

Invoked by inflate.sh:
    python -m submissions.roro_perframe_hnerv.inflate <data_dir>/0.bin <output_dir>/0.raw
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src'))

from model import QLRDecoder          # noqa: E402
from codec import parse_archive       # noqa: E402

CAMERA_H, CAMERA_W = 874, 1164


def inflate(src_bin: str, dst_raw: str):
    archive_bytes = Path(src_bin).read_bytes()
    decoder_sd, latents, meta = parse_archive(archive_bytes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decoder = QLRDecoder(latent_dim=meta['latent_dim'],
                         base_channels=meta['base_channels'],
                         rank=meta['rank'],
                         eval_size=tuple(meta['eval_size'])).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    # per-frame latents: latent i -> frame i, already in playback order
    n_frames = meta.get('n_frames', 2 * meta['n_pairs'])
    eval_h, eval_w = meta['eval_size']

    n = 0
    with torch.inference_mode(), open(dst_raw, 'wb') as fout:
        for i in range(0, n_frames, 32):
            j = min(i + 32, n_frames)
            frames_eval = decoder(latents[i:j])        # (B,3,eval_h,eval_w)
            up = F.interpolate(frames_eval, size=(CAMERA_H, CAMERA_W),
                               mode='bicubic', align_corners=False)
            frames = (up.clamp(0, 255).permute(0, 2, 3, 1)
                        .round().to(torch.uint8).cpu().numpy())
            fout.write(frames.tobytes())
            n += frames.shape[0]
    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m submissions.roro_perframe_hnerv.inflate <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
