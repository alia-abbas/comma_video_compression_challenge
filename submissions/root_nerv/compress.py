#!/usr/bin/env python
"""
compress.py — Mask-conditioned video compression.

Pipeline:
  1. Run SegNet on every frame → store 5-class masks as H.265 grayscale video
  2. Train a tiny MaskPairGenerator (mask_t, mask_t+1) → (rgb_t, rgb_t+1)
     using the EXACT contest scoring function as the loss.
     SegNet and PoseNet are FREE at training time — never go in archive.
  3. Quantise generator to FP16, brotli-compress both artifacts.

Archive contents (both count toward rate):
  model.pt.br   — generator weights (~64–100 KB)
  masks.mp4.br  — segmentation mask video (~200–350 KB)
"""

import io, os, sys, av, argparse, tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import brotli
from pathlib import Path
from tqdm import tqdm

# ── Challenge repo imports (free at training time, NOT in archive) ────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from modules import load_segnet, load_posenet
except ImportError:
    from modules import SegNet, PoseNet
    def load_segnet():  return SegNet()
    def load_posenet(): return PoseNet()

# ─────────────────────────────────────────────────────────────────────────────
# Architecture — must exactly mirror inflate.py
# ─────────────────────────────────────────────────────────────────────────────

class DWSConv(nn.Module):
    def __init__(self, inc, outc, k=3, p=1):
        super().__init__()
        self.dw = nn.Conv2d(inc, inc, k, padding=p, groups=inc, bias=False)
        self.pw = nn.Conv2d(inc, outc, 1, bias=False)
    def forward(self, x):
        return self.pw(self.dw(x))

class MaskPairGenerator(nn.Module):
    """
    (mask_t, mask_t1) — (B,H,W) long, values 0-4
    →  (rgb_t, rgb_t1) — (B,3,H,W) float [0,1]

    Joint processing of pairs gives temporal context → better PoseNet score.
    """
    def __init__(self, nc=5, ch=48):
        super().__init__()
        self.emb = nn.Embedding(nc, ch)
        self.enc = nn.Sequential(
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
# Step 1 — Mask extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_masks(video_path: Path, segnet, device):
    """
    Returns:
      masks_np   : list[ndarray(H,W) uint8]  class indices 0-4
      frames_rgb : list[ndarray(H,W,3) uint8]
    """
    container = av.open(str(video_path))
    masks_np, frames_rgb = [], []

    with torch.inference_mode():
        for frame in tqdm(container.decode(video=0), desc="  SegNet pass"):
            img = frame.to_ndarray(format='rgb24')
            frames_rgb.append(img)

            t = (torch.from_numpy(img)
                      .permute(2, 0, 1).float().div(255.0)
                      .unsqueeze(0).to(device))
            pred = segnet(t).argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
            masks_np.append(pred)

    container.close()
    print(f"  Extracted {len(masks_np)} masks  shape={masks_np[0].shape}")
    return masks_np, frames_rgb


def encode_masks_to_mp4(masks: list, out_path: Path, crf: int = 45):
    """
    H.265 grayscale encoding of 5-class masks.
    Class → pixel: 0→0  1→63  2→126  3→189  4→252
    5-symbol alphabet compresses extremely well with H.265.
    """
    h, w = masks[0].shape
    out    = av.open(str(out_path), 'w')
    stream = out.add_stream('libx265', rate=20)
    stream.width, stream.height = w, h
    stream.pix_fmt = 'gray'
    stream.options = {'crf': str(crf), 'preset': 'slow',
                      'x265-params': 'log-level=none'}

    for m in tqdm(masks, desc="  Encoding mask video"):
        gray  = (m * 63).astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(gray, format='gray')
        out.mux(stream.encode(frame))
    out.mux(stream.encode(None))
    out.close()

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Precompute targets (run once, reuse every epoch)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_targets(frames_rgb, segnet, posenet, device):
    """
    Cache metric-model outputs so we don't recompute each epoch.
    Both models are frozen and FREE — they never go in the archive.

    Returns:
      seg_targets  : list[Tensor(H,W) long]   — argmax class per frame
      seg_logit_tgts: list[Tensor(C,H,W)]     — soft logits for cross-entropy
      pose_targets : list[Tensor(...)]         — posenet output per pair (i, i+1)
    """
    print("  Precomputing SegNet targets…")
    seg_targets, seg_logit_tgts = [], []
    with torch.inference_mode():
        for img in tqdm(frames_rgb):
            t = (torch.from_numpy(img).permute(2,0,1)
                      .float().div(255.0).unsqueeze(0).to(device))
            logits = segnet(t).squeeze(0).cpu()          # (C, H, W)
            seg_logit_tgts.append(logits)
            seg_targets.append(logits.argmax(0))         # (H, W)

    print("  Precomputing PoseNet targets…")
    pose_targets = []
    with torch.inference_mode():
        for i in range(len(frames_rgb) - 1):
            t0 = (torch.from_numpy(frames_rgb[i]).permute(2,0,1)
                       .float().div(255.0).unsqueeze(0).to(device))
            t1 = (torch.from_numpy(frames_rgb[i+1]).permute(2,0,1)
                       .float().div(255.0).unsqueeze(0).to(device))
            pose_targets.append(posenet(t0, t1).cpu())

    return seg_targets, seg_logit_tgts, pose_targets

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Training
# ─────────────────────────────────────────────────────────────────────────────

def train(generator, masks_np, seg_targets, seg_logit_tgts,
          pose_targets, segnet, posenet, device, cfg):
    """
    Loss = exact contest scoring function:
      100 * segnet_distortion + sqrt(10 * posenet_distortion)

    KEY FIX: segnet loss is cross-entropy against segnet logits, which IS
    differentiable (argmax-based comparison gives zero gradients).
    """
    opt   = torch.optim.AdamW(generator.parameters(),
                               lr=cfg['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])

    best_loss, best_state = float('inf'), None
    n = len(masks_np)

    for epoch in range(cfg['epochs']):
        generator.train()
        epoch_loss = 0.0

        # Train on all consecutive pairs, shuffled
        all_pairs = list(range(n - 1))
        np.random.shuffle(all_pairs)

        for i in all_pairs:
            j = i + 1

            m_t  = torch.from_numpy(masks_np[i]).long().unsqueeze(0).to(device)
            m_t1 = torch.from_numpy(masks_np[j]).long().unsqueeze(0).to(device)

            pred_t, pred_t1 = generator(m_t, m_t1)

            # ── SegNet loss: cross-entropy against precomputed logits ──────
            # cross_entropy expects (B, C, H, W) logits and (B, H, W) targets.
            # This IS differentiable — argmax comparison is NOT.
            seg_tgt_t  = seg_targets[i].unsqueeze(0).to(device)   # (1,H,W) long
            seg_tgt_t1 = seg_targets[j].unsqueeze(0).to(device)

            seg_logits_t  = segnet(pred_t)    # (1, C, H, W)
            seg_logits_t1 = segnet(pred_t1)

            seg_loss = (F.cross_entropy(seg_logits_t,  seg_tgt_t) +
                        F.cross_entropy(seg_logits_t1, seg_tgt_t1)) / 2.0

            # ── PoseNet loss ───────────────────────────────────────────────
            pose_pred = posenet(pred_t, pred_t1)
            pose_tgt  = pose_targets[i].to(device)
            pose_loss = F.mse_loss(pose_pred, pose_tgt)

            # ── Exact contest loss ─────────────────────────────────────────
            loss = 100.0 * seg_loss + (10.0 * pose_loss).sqrt()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()

        sched.step()
        avg = epoch_loss / max(len(all_pairs), 1)

        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone()
                          for k, v in generator.state_dict().items()}

        if epoch % 10 == 0 or epoch == cfg['epochs'] - 1:
            print(f"  Epoch {epoch:3d}/{cfg['epochs']} | "
                  f"loss={avg:.4f} | best={best_loss:.4f}")

    return best_state

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Package
# ─────────────────────────────────────────────────────────────────────────────

def package(state_dict: dict, mask_mp4_path: Path, output_dir: Path):
    # Model → FP16 → brotli
    fp16_sd = {k: v.to(torch.float16) for k, v in state_dict.items()}
    buf = io.BytesIO()
    torch.save(fp16_sd, buf)
    model_compressed = brotli.compress(buf.getvalue(), quality=11)
    (output_dir / "model.pt.br").write_bytes(model_compressed)

    # Masks → brotli
    mask_compressed = brotli.compress(mask_mp4_path.read_bytes(), quality=11)
    (output_dir / "masks.mp4.br").write_bytes(mask_compressed)

    total     = len(model_compressed) + len(mask_compressed)
    rate      = total / 37_545_489
    print(f"\n  model.pt.br : {len(model_compressed)/1024:7.1f} KB")
    print(f"  masks.mp4.br: {len(mask_compressed)/1024:7.1f} KB")
    print(f"  Total       : {total/1024:7.1f} KB")
    print(f"  rate        : {rate:.5f}   rate term (×25): {25*rate:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video-dir',   required=True)
    p.add_argument('--video-names', required=True)
    p.add_argument('--output-dir',  required=True)
    p.add_argument('--epochs', type=int,   default=100)
    p.add_argument('--lr',    type=float,  default=3e-4)
    p.add_argument('--ch',    type=int,    default=48,
                   help='Generator base channels (48 → ~88k params)')
    p.add_argument('--crf',   type=int,    default=45,
                   help='Mask video CRF (higher = smaller file, rougher masks)')
    args = p.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_names = [v.strip() for v in
                   Path(args.video_names).read_text().splitlines() if v.strip()]

    # Load metric models — FREE, not counted in archive
    print("Loading SegNet and PoseNet (free — not archived)…")
    segnet  = load_segnet().to(device).eval()
    posenet = load_posenet().to(device).eval()
    for param in list(segnet.parameters()) + list(posenet.parameters()):
        param.requires_grad = False

    cfg = {'epochs': args.epochs, 'lr': args.lr}

    for video_name in video_names:
        video_path = Path(args.video_dir) / video_name
        print(f"\n{'='*60}\nProcessing {video_name}\n{'='*60}")

        # 1. Extract masks + raw frames
        masks_np, frames_rgb = extract_masks(video_path, segnet, device)

        # 2. Encode masks to temporary MP4
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            mask_mp4 = Path(tmp.name)
        encode_masks_to_mp4(masks_np, mask_mp4, crf=args.crf)
        print(f"  Raw mask MP4: {mask_mp4.stat().st_size/1024:.1f} KB")

        # 3. Precompute metric targets (done once, reused every epoch)
        seg_targets, seg_logit_tgts, pose_targets = precompute_targets(
            frames_rgb, segnet, posenet, device)

        # 4. Train generator
        generator = MaskPairGenerator(nc=5, ch=args.ch).to(device)
        n_params  = sum(p.numel() for p in generator.parameters())
        print(f"  Generator: {n_params:,} parameters")

        best_state = train(generator, masks_np,
                           seg_targets, seg_logit_tgts, pose_targets,
                           segnet, posenet, device, cfg)

        # 5. Package
        package(best_state, mask_mp4, output_dir)
        mask_mp4.unlink()

    print("\nDone. Add model.pt.br, masks.mp4.br, and inflate.py to archive.zip.")


if __name__ == '__main__':
    main()