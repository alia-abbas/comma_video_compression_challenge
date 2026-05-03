#!/usr/bin/env python
import io, os, sys, av
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import brotli
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Generator — must exactly match compress.py's MaskPairGenerator
# ─────────────────────────────────────────────────────────────────────────────

class DWSConv(nn.Module):
    def __init__(self, inc, outc, k=3, p=1):
        super().__init__()
        self.dw = nn.Conv2d(inc, inc, k, padding=p, groups=inc, bias=False)
        self.pw = nn.Conv2d(inc, outc, 1, bias=False)
    def forward(self, x):
        return self.pw(self.dw(x))

class MaskPairGenerator(nn.Module):
    def __init__(self, nc=5, ch=48):
        super().__init__()
        self.emb  = nn.Embedding(nc, ch)
        self.enc  = nn.Sequential(
            DWSConv(ch * 2, 96), nn.GELU(),
            DWSConv(96,     96), nn.GELU(),
            DWSConv(96,     96), nn.GELU(),
            DWSConv(96,     64), nn.GELU(),
        )
        self.dec1 = nn.Sequential(DWSConv(64, ch), nn.GELU(),
                                  nn.Conv2d(ch, 3, 1), nn.Sigmoid())
        self.dec2 = nn.Sequential(DWSConv(64, ch), nn.GELU(),
                                  nn.Conv2d(ch, 3, 1), nn.Sigmoid())

    def forward(self, m1, m2):
        f1 = self.emb(m1).permute(0, 3, 1, 2)
        f2 = self.emb(m2).permute(0, 3, 1, 2)
        h  = self.enc(torch.cat([f1, f2], dim=1))
        return self.dec1(h), self.dec2(h)

# ─────────────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_generator(model_br_path: Path, device: torch.device,
                   nc: int = 5, ch: int = 48) -> MaskPairGenerator:
    raw = brotli.decompress(model_br_path.read_bytes())
    sd  = torch.load(io.BytesIO(raw), map_location=device)

    # FP16 → FP32
    sd = {k: (v.float() if torch.is_floating_point(v) else v)
          for k, v in sd.items()}

    gen = MaskPairGenerator(nc=nc, ch=ch).to(device)
    gen.load_state_dict(sd, strict=True)
    gen.eval()
    return gen


def load_masks(masks_br_path: Path) -> torch.Tensor:
    """
    Returns (N, H, W) uint8 tensor of class indices 0-4.
    Pixel values in the grayscale video are multiples of 63.
    """
    import tempfile
    raw_mp4 = brotli.decompress(masks_br_path.read_bytes())

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(raw_mp4)
        tmp_path = tmp.name

    container = av.open(tmp_path)
    frames = []
    for frame in container.decode(video=0):
        gray = frame.to_ndarray(format='gray')               # H, W  uint8
        cls  = np.round(gray / 63.0).astype(np.uint8)
        cls  = np.clip(cls, 0, 4)
        frames.append(cls)
    container.close()
    os.unlink(tmp_path)

    return torch.from_numpy(np.stack(frames))                # N, H, W

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <data_dir> <output_dir> <file_list_txt>")
        sys.exit(1)

    data_dir   = Path(sys.argv[1])
    out_dir    = Path(sys.argv[2])
    file_names = [l.strip() for l in
                  Path(sys.argv[3]).read_text().splitlines() if l.strip()]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load artifacts from archive ───────────────────────────────────────────
    print("Loading generator weights…")
    generator = load_generator(data_dir / "model.pt.br", device)

    print("Loading mask video…")
    all_masks = load_masks(data_dir / "masks.mp4.br")   # (N_total, H, W)

    out_h, out_w = 874, 1164
    cursor       = 0

    with torch.inference_mode():
        for file_name in file_names:
            base      = os.path.splitext(file_name)[0]
            raw_path  = out_dir / f"{base}.raw"
            raw_path.parent.mkdir(parents=True, exist_ok=True)

            # Each video uses up to 1200 mask frames
            n_frames    = min(1200, all_masks.shape[0] - cursor)
            video_masks = all_masks[cursor : cursor + n_frames]   # (N, H, W)
            cursor     += n_frames

            # Process in consecutive pairs
            usable = (video_masks.shape[0] // 2) * 2
            pairs  = video_masks[:usable].view(-1, 2,
                                               video_masks.shape[-2],
                                               video_masks.shape[-1])

            print(f"Decoding {file_name}  ({pairs.shape[0]} pairs)…")

            with open(raw_path, 'wb') as f_out:
                batch_size = 8
                for i in tqdm(range(0, pairs.shape[0], batch_size),
                              desc=f"  {file_name}"):
                    bp = pairs[i : i + batch_size].to(device)

                    m1 = bp[:, 0].long()   # B, H, W
                    m2 = bp[:, 1].long()

                    rgb1, rgb2 = generator(m1, m2)

                    # Upsample to evaluation resolution
                    rgb1 = F.interpolate(rgb1, size=(out_h, out_w),
                                         mode='bilinear', align_corners=False)
                    rgb2 = F.interpolate(rgb2, size=(out_h, out_w),
                                         mode='bilinear', align_corners=False)

                    # Write interleaved frames as raw YUV444p
                    for rgb in (rgb1, rgb2):
                        # rgb: (B, 3, H, W) float [0,1]
                        imgs = (rgb.clamp(0, 1) * 255).round().to(torch.uint8)
                        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # B,H,W,3

                        for img in imgs:
                            frame     = av.VideoFrame.from_ndarray(img, format='rgb24')
                            yuv_frame = frame.reformat(out_w, out_h, 'yuv444p')
                            for plane_idx in range(3):
                                plane  = yuv_frame.planes[plane_idx]
                                stride = plane.line_size
                                data   = (np.frombuffer(plane, dtype=np.uint8)
                                            .reshape(out_h, stride))
                                f_out.write(data[:, :out_w].tobytes())

            print(f"  → {raw_path}")

    print("All videos inflated.")


if __name__ == "__main__":
    main()