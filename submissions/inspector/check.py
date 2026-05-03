#!/usr/bin/env python
"""
check.py — Saliency profiling via SegNet + PoseNet gradients.

For each consecutive frame pair (t, t+1):
  importance_t = 100 * |d(SegNet)/d(frame_t)|
              + sqrt(10) * |d(PoseNet)/d(frame_t)|   ← weighted by contest scoring

Outputs:
  saliency/frame_0000_heat.png     — heatmap overlay (frame 0)
  saliency/frame_0600_heat.png     — heatmap overlay (frame 600)
  saliency/frame_1199_heat.png     — heatmap overlay (frame 1199)
  saliency/importance.pt           — (N, H, W) float32, all frames, original res
"""

import math, sys
import numpy as np
import torch
import torch.nn.functional as F
import einops
import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file

# ── Repo imports ──────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from modules import SegNet, PoseNet
from frame_utils import rgb_to_yuv6, segnet_model_input_size

# ── Paths ─────────────────────────────────────────────────────────────────────
VIDEO_PATH   = REPO / 'videos/0.mkv'
SEGNET_W     = REPO / 'models/segnet.safetensors'
POSENET_W    = REPO / 'models/posenet.safetensors'
SALIENCY_DIR = REPO / 'submissions/inspector/saliency'

# Frames to save as PNG heatmaps for visual inspection
VISUAL_FRAMES = {0, 600, 1199}

# Gradient computation always runs on CPU.
# MPS has incomplete autograd support for FastViT (PoseNet backbone).
# CUDA (3090) supports full autograd — change GRAD_DEVICE to 'cuda' there.
GRAD_DEVICE = torch.device('cuda') 

# ─────────────────────────────────────────────────────────────────────────────

def load_models(device):
    segnet = SegNet().to(device)
    segnet.load_state_dict(load_file(str(SEGNET_W), device=str(device)))
    segnet.eval()

    posenet = PoseNet().to(device)
    posenet.load_state_dict(load_file(str(POSENET_W), device=str(device)))
    posenet.eval()

    # Freeze — we only want gradients w.r.t. input frames, not model weights
    #for p in list(segnet.parameters()) + list(posenet.parameters()):
    #    p.requires_grad_(False)

    return segnet, posenet


def load_frames(video_path):
    """Returns list of (H, W, 3) uint8 numpy arrays."""
    container = av.open(str(video_path))
    frames = []
    for frame in tqdm(container.decode(video=0), desc="Loading frames"):
        frames.append(frame.to_ndarray(format='rgb24'))
    container.close()
    print(f"Loaded {len(frames)} frames — shape {frames[0].shape}")
    return frames


def to_tensor(img_np, device):
    """(H,W,3) uint8 → (1,3,H,W) float32 [0-255]."""
    return (torch.from_numpy(img_np)
                 .permute(2, 0, 1)
                 .float()
                 .unsqueeze(0)
                 .to(device))


def segnet_grad(frame_np, segnet, device):
    """
    Single-frame SegNet gradient.
    Returns (H, W) float32 importance map at original resolution.
    """
    x = to_tensor(frame_np, device).requires_grad_(True)

    # Resize to model input — bilinear is differentiable, grad flows back
    x_in = F.interpolate(
        x,
        size=(segnet_model_input_size[1], segnet_model_input_size[0]),
        mode='bilinear', align_corners=False
    )

    logits = segnet(x_in)       # (1, 5, H', W')
    logits.sum().backward()

    # x.grad: (1, 3, H, W) → sum over batch+channel → (H, W)
    return x.grad.abs().sum(dim=[0, 1])


def posenet_grad(frame_t_np, frame_t1_np, posenet, device):
    """
    Two-frame PoseNet gradient.
    Returns two (H, W) float32 importance maps (one per frame) at original res.

    PoseNet expects (B, T*6, H', W') — 2 frames × 6 YUV channels.
    We keep the whole preprocessing chain in autograd so gradients flow back.
    """
    x0 = to_tensor(frame_t_np,  device).detach().requires_grad_(True)
    x1 = to_tensor(frame_t1_np, device).detach().requires_grad_(True)

    # ── Replicate posenet.preprocess_input with autograd ─────────────────────
    B, T = 1, 2
    pair = torch.cat([x0, x1], dim=0)          # (2, 3, H, W)
    pair_resized = F.interpolate(
        pair,
        size=(segnet_model_input_size[1], segnet_model_input_size[0]),
        mode='bilinear', align_corners=False
    )                                            # (2, 3, H', W')
    pair_yuv = rgb_to_yuv6(pair_resized)        # (2, 6, H', W')
    # Stack into (B=1, T*6=12, H', W')
    x_in = einops.rearrange(
        pair_yuv, '(b t) c h w -> b (t c) h w', b=B, t=T
    )

    posenet.zero_grad()
    pose_out = posenet(x_in)                    # dict — 'pose': (1, 12)
    print(f"Check x_in grad_fn: {x_in.grad_fn}") 
    
    #pose_out['pose'].sum().backward()
    loss = pose_out['pose'].sum()
    loss.backward()

    if x0.grad is None or x1.grad is None:
        raise RuntimeError("Gradient did not propagate to input tensors. Check if rgb_to_yuv6 is differentiable.")

    grad0 = x0.grad.abs().squeeze().sum(dim=[0, 1])       # (H, W)
    grad1 = x1.grad.abs().squeeze().sum(dim=[0, 1])       # (H, W)
    return grad0, grad1


def normalize(t):
    mn, mx = t.min(), t.max()
    if (mx - mn).item() < 1e-8:
        return torch.zeros_like(t)
    return (t - mn) / (mx - mn)


def save_heatmap(frame_np, importance_norm, idx, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].imshow(frame_np)
    axes[0].set_title(f'Original — frame {idx}')
    axes[0].axis('off')

    axes[1].imshow(frame_np)
    axes[1].imshow(importance_norm.cpu().numpy(), cmap='hot', alpha=0.65,
                   vmin=0, vmax=1)
    axes[1].set_title('Importance  (bright = protect,  dark = destroy)')
    axes[1].axis('off')

    plt.tight_layout()
    out_path = out_dir / f'frame_{idx:04d}_heat.png'
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved heatmap → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = (torch.device('cuda')  if torch.cuda.is_available()  else
              torch.device('mps')   if torch.backends.mps.is_available() else
              torch.device('cpu'))
    print(f"Inference device : {device}")
    print(f"Gradient device  : {GRAD_DEVICE}  (CPU — MPS lacks full autograd)")

    SALIENCY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load models onto GRAD_DEVICE for autograd compatibility
    print("Loading SegNet and PoseNet…")
    segnet, posenet = load_models(GRAD_DEVICE)

    # 2. Load all frames
    frames = load_frames(VIDEO_PATH)
    N = len(frames)
    H, W = frames[0].shape[:2]

    # 3. Accumulate importance maps — one per frame
    #    For frame N-1 (no t+1), we reuse the posenet grad from the last pair.
    importance_all = torch.zeros(N, H, W, dtype=torch.float32)

    last_pose_grad1 = None  # carries forward for the final frame

    print(f"\nComputing importance maps for {N} frames…")
    for i in tqdm(range(N)):
        # ── SegNet gradient (single frame) ───────────────────────────────────
        seg_g = segnet_grad(frames[i], segnet, GRAD_DEVICE)     # (H, W)

        # ── PoseNet gradient (pair) ───────────────────────────────────────────
        if i < N - 1:
            p_g0, p_g1 = posenet_grad(frames[i], frames[i + 1], posenet, GRAD_DEVICE)
            last_pose_grad1 = p_g1
        else:
            # Last frame: no i+1 — reuse the grad1 from the previous pair
            p_g0 = last_pose_grad1 if last_pose_grad1 is not None else torch.zeros_like(seg_g)

        # ── Weighted combination matching contest scoring weights ─────────────
        importance = 100.0 * seg_g + math.sqrt(10.0) * p_g0    # (H, W)
        importance_all[i] = importance.cpu()

        # ── Visual heatmaps for 3 selected frames ────────────────────────────
        if i in VISUAL_FRAMES:
            save_heatmap(frames[i], normalize(importance), i, SALIENCY_DIR)

        # ── Numerical validation: print mean every 100 frames ────────────────
        if i % 100 == 0:
            print(f"  frame {i:4d} | mean_importance = {importance.mean().item():.4f}")

    # 4. Save full importance tensor
    out_pt = SALIENCY_DIR / 'importance.pt'
    torch.save(importance_all, out_pt)
    print(f"\nSaved all importance maps → {out_pt}")
    print(f"Shape: {importance_all.shape}  |  dtype: {importance_all.dtype}")
    print(f"Global mean: {importance_all.mean().item():.4f}")
    print(f"Global max:  {importance_all.max().item():.4f}")
    print("\nDone. Check submissions/root/saliency/ for heatmap PNGs.")


if __name__ == '__main__':
    main()