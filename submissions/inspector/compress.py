#!/usr/bin/env python
"""
compress.py — Saliency-guided ROI video compression, v2.

Importance map pipeline (per frame):
  1. Gradient map      — from check.py (100*segnet_grad + √10*posenet_grad)
  2. SegNet object mask — non-background classes → guaranteed object protection
  3. Motion mask        — frame differencing → protects moving objects
  4. Combine + dilate   — safety zone around all edges
  5. Composite          — important regions: original pixels
                          unimportant regions: 2x downsample→upsample (smooth, cheap to encode)
  6. Encode with SVT-AV1 at CRF 43
"""

import sys, av, brotli, argparse, zipfile
import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage
from pathlib import Path

# ── Repo imports (free at compress time, never in archive) ────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from modules import SegNet
from frame_utils import segnet_model_input_size
from safetensors.torch import load_file

# ── Config ────────────────────────────────────────────────────────────────────
CRF                = 43      # sweet spot from sweep
DOWNSAMPLE_FACTOR  = 4       # unimportant regions: shrink by this, upsample back
DILATION_RADIUS    = 20      # pixels — safety zone around important edges
MOTION_THRESH      = 15      # pixel diff to count as motion (0-255)
GRADIENT_WEIGHT    = 1.0     # weight for gradient map
SEGNET_WEIGHT      = 2.0     # weight for segnet object mask (strong prior)
MOTION_WEIGHT      = 1.5     # weight for motion mask
IMPORTANCE_THRESH  = 0.35    # final combined threshold: above → protect, below → smooth

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video-dir',   required=True)
    p.add_argument('--video-names', required=True)
    p.add_argument('--output-dir',  required=True)
    return p.parse_args()


# ── SegNet loader ─────────────────────────────────────────────────────────────

def load_segnet(device):
    segnet = SegNet().to(device).eval()
    segnet.load_state_dict(load_file(str(REPO / 'models/segnet.safetensors'),
                                     device=str(device)))
    for p in segnet.parameters():
        p.requires_grad_(False)
    print(f"  SegNet loaded on {device}")
    return segnet


def get_segnet_mask(frame_rgb: np.ndarray, segnet, device) -> np.ndarray:
    """
    Returns (H, W) float32 binary mask: 1.0 where any non-background class,
    0.0 for background (class 0). Upsampled to original frame resolution.
    """
    H, W = frame_rgb.shape[:2]
    t = (torch.from_numpy(frame_rgb).permute(2, 0, 1)
             .float().unsqueeze(0).to(device))
    t_in = F.interpolate(t,
                         size=(segnet_model_input_size[1],
                               segnet_model_input_size[0]),
                         mode='bilinear', align_corners=False)
    with torch.inference_mode():
        logits = segnet(t_in)                       # (1, 5, H', W')
    classes = logits.argmax(1).squeeze(0)           # (H', W')
    obj_mask = (classes != 0).float().unsqueeze(0).unsqueeze(0)  # (1,1,H',W')
    obj_mask = F.interpolate(obj_mask, size=(H, W),
                             mode='nearest').squeeze().cpu().numpy()
    return obj_mask.astype(np.float32)


# ── Motion mask via frame differencing ───────────────────────────────────────

def get_motion_mask(curr_rgb: np.ndarray,
                    prev_rgb: np.ndarray) -> np.ndarray:
    """
    Returns (H, W) float32 binary mask: 1.0 where significant pixel change.
    """
    if prev_rgb is None:
        return np.zeros(curr_rgb.shape[:2], dtype=np.float32)
    diff = np.abs(curr_rgb.astype(np.float32) - prev_rgb.astype(np.float32))
    motion = (diff.max(axis=2) > MOTION_THRESH).astype(np.float32)
    return motion


# ── Importance combination + dilation ────────────────────────────────────────

def build_importance(gradient: np.ndarray,
                     segnet_mask: np.ndarray,
                     motion_mask: np.ndarray) -> np.ndarray:
    """
    Combine all importance signals, dilate for edge safety zones.
    Returns (H, W) float32 normalized to [0, 1].
    """
    combined = (GRADIENT_WEIGHT * gradient +
                SEGNET_WEIGHT   * segnet_mask +
                MOTION_WEIGHT   * motion_mask)

    # Normalize
    mx = combined.max()
    if mx > 1e-8:
        combined = combined / mx

    # Binary threshold then dilate — safety zone around all important regions
    binary   = (combined > IMPORTANCE_THRESH).astype(np.float32)
    dilated  = scipy.ndimage.binary_dilation(
        binary, iterations=DILATION_RADIUS
    ).astype(np.float32)

    return dilated


# ── Frame compositing ─────────────────────────────────────────────────────────

def process_frame(curr_rgb: np.ndarray,
                  importance: np.ndarray) -> np.ndarray:
    """
    Important regions  (importance=1): original pixels, full detail.
    Unimportant regions (importance=0): 2x downsample → upsample back.
      Low-frequency content → codec spends almost no bits here.
      No blur artifacts. No motion freezing.
    """
    H, W = curr_rgb.shape[:2]
    small_h, small_w = H // DOWNSAMPLE_FACTOR, W // DOWNSAMPLE_FACTOR

    # Downsample → upsample the whole frame (removes high-freq detail)
    t_small  = torch.from_numpy(curr_rgb[::DOWNSAMPLE_FACTOR,
                                          ::DOWNSAMPLE_FACTOR]).permute(2,0,1).float().unsqueeze(0)
    t_smooth = F.interpolate(t_small, size=(H, W), mode='bilinear', align_corners=False)
    smooth   = t_smooth.squeeze(0).permute(1,2,0).round().clamp(0,255).numpy().astype(np.uint8)

    mask  = importance[..., np.newaxis].astype(np.float32)  # (H, W, 1)
    out   = (curr_rgb.astype(np.float32) * mask +
             smooth.astype(np.float32)  * (1.0 - mask))
    return out.round().astype(np.uint8)


# ── Importance map loading ────────────────────────────────────────────────────

def load_importance(saliency_path: Path):
    imp = torch.load(str(saliency_path), map_location='cpu').float()
    v_min, v_max = imp.min(), imp.max()
    imp = (imp - v_min) / (v_max - v_min + 1e-8)
    print(f"  Gradient maps: {imp.shape}")
    return imp.numpy()   # (N, H, W) float32


# ── Main compression ──────────────────────────────────────────────────────────

def compress_video(video_path: Path, gradient_maps: np.ndarray,
                   segnet, device, out_ivf: Path):
    in_c  = av.open(str(video_path))
    out_c = av.open(str(out_ivf), mode='w')

    in_stream  = in_c.streams.video[0]
    out_stream = out_c.add_stream('libsvtav1', rate=in_stream.average_rate)
    out_stream.width   = in_stream.width
    out_stream.height  = in_stream.height
    out_stream.pix_fmt = 'yuv420p'
    out_stream.options = {'preset': '6', 'crf': str(CRF)}

    N        = gradient_maps.shape[0]
    prev_rgb = None
    idx      = 0

    for frame in in_c.decode(video=0):
        curr_rgb = frame.reformat(format='rgb24').to_ndarray()   # (H,W,3) uint8

        i          = min(idx, N - 1)
        seg_mask   = get_segnet_mask(curr_rgb, segnet, device)
        mot_mask   = get_motion_mask(curr_rgb, prev_rgb)
        importance = build_importance(gradient_maps[i], seg_mask, mot_mask)
        processed  = process_frame(curr_rgb, importance)

        out_frame           = av.VideoFrame.from_ndarray(processed, format='rgb24')
        out_frame.pts       = frame.pts
        out_frame.time_base = frame.time_base
        for pkt in out_stream.encode(out_frame):
            out_c.mux(pkt)

        prev_rgb = curr_rgb   # store original for motion diff
        idx     += 1
        if idx % 100 == 0:
            print(f"  {idx}/{N} frames encoded…")

    for pkt in out_stream.encode():
        out_c.mux(pkt)

    in_c.close()
    out_c.close()
    print(f"  IVF: {out_ivf.stat().st_size / 1e6:.1f} MB")


def package(ivf_path: Path, output_dir: Path, submission_dir: Path):
    print("Brotli compressing…")
    br_path = output_dir / 'archive.br'
    br_path.write_bytes(brotli.compress(ivf_path.read_bytes(), quality=11))

    zip_path = submission_dir / 'archive.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:
        z.write(br_path,                       arcname='archive.br')
        z.write(submission_dir / 'inflate.py', arcname='inflate.py')
        z.write(submission_dir / 'inflate.sh', arcname='inflate.sh')

    total = zip_path.stat().st_size
    rate  = total / 37_545_489
    print(f"  archive.zip: {total/1e6:.1f} MB | rate={rate:.5f} | "
          f"rate term={25*rate:.3f}")


def main():
    args       = parse_args()
    HERE       = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saliency_path = HERE / 'saliency' / 'importance.pt'
    if not saliency_path.exists():
        print(f"ERROR: {saliency_path} not found — run check.py first.")
        sys.exit(1)

    device = (torch.device('cuda') if torch.cuda.is_available() else
              torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cpu'))
    print(f"Device: {device}")

    print("Loading gradient maps…")
    gradient_maps = load_importance(saliency_path)

    print("Loading SegNet…")
    segnet = load_segnet(device)

    video_names = [v.strip() for v in
                   Path(args.video_names).read_text().splitlines() if v.strip()]

    for video_name in video_names:
        video_path = Path(args.video_dir) / video_name
        print(f"\nProcessing {video_name}…")

        out_ivf = output_dir / 'compressed.ivf'
        compress_video(video_path, gradient_maps, segnet, device, out_ivf)
        package(out_ivf, output_dir, HERE)
        out_ivf.unlink()

    print("\nDone.")


if __name__ == '__main__':
    main()