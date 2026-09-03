"""Overfit the QLR decoder + per-pair latents to videos/0.mkv, directly in the
evaluator's metric space, with quantization-aware training (QAT).

Objective (all computed at eval resolution 384x512, the resolution both nets
actually consume):

  loss = w_seg * CE( SegNet(recon_odd), argmax SegNet(gt_odd) )      # SegNet term
       + w_pose * MSE( PoseNet(recon_pair)[:6], PoseNet(gt_pair)[:6])# PoseNet term
       + w_pix  * L1( recon, gt )                                    # stabilizer

The SegNet/PoseNet weights are frozen (loaded from models/). Gradients flow
through them into the decoder + latents but never update them.

QAT: in the final phase, decoder weights are fake-quantized to `qat_bits`
(per-tensor symmetric, straight-through estimator) on every forward so the
trained weights survive the real INT quantization in codec.py with ~no drop.

Writes checkpoints (resumable) and, at the end, the compressed archive to
<run_dir>/submission_archive/0.bin via codec.build_archive.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import modules  # noqa: E402
import frame_utils  # noqa: E402
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from data import PairData, ROOT as DATA_ROOT  # noqa: E402


# ---------------------------------------------------------------------------
# CRITICAL FIX: the challenge's frame_utils.rgb_to_yuv6 (which PoseNet's
# preprocess_input calls every forward) is decorated @torch.no_grad() and uses
# in-place clamp_(), so the PoseNet loss gradient NEVER reaches the decoder ->
# pose distortion is frozen at its init value during training. Here is a
# differentiable, value-identical replacement (out-of-place clamp, no no_grad).
# We monkey-patch it into both modules so PoseNet.preprocess_input uses it.
# Forward values are bit-identical to the original; only gradients now flow.
# (Inference / the real evaluator still use the original; this affects training.)
# ---------------------------------------------------------------------------
def _rgb_to_yuv6_diff(rgb_chw):
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2 * H2, :2 * W2]
    R = rgb[..., 0, :, :]; G = rgb[..., 1, :, :]; B = rgb[..., 2, :, :]
    kYR, kYG, kYB = 0.299, 0.587, 0.114
    Y = (R * kYR + G * kYG + B * kYB).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
             U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
             V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    y00 = Y[..., 0::2, 0::2]; y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]; y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


modules.rgb_to_yuv6 = _rgb_to_yuv6_diff
frame_utils.rgb_to_yuv6 = _rgb_to_yuv6_diff
from model import QLRDecoder, count_params  # noqa: E402


# ---------------------------------------------------------------------------
# QAT: per-tensor symmetric fake-quant with straight-through estimator
# ---------------------------------------------------------------------------
class _FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, n_levels):
        m = w.abs().max()
        scale = torch.where(m > 0, m / n_levels, torch.ones_like(m))
        q = torch.clamp(torch.round(w / scale), -n_levels, n_levels)
        return q * scale

    @staticmethod
    def backward(ctx, g):
        return g, None  # straight-through


def fake_quant(w, bits):
    n_levels = 2 ** (bits - 1) - 1
    return _FakeQuant.apply(w, float(n_levels))


# ---------------------------------------------------------------------------
# Target precomputation (gt outputs of the frozen nets) -- done once
# ---------------------------------------------------------------------------
@torch.inference_mode()
def precompute_targets(pairs, segnet, posenet, device, batch=16):
    n = pairs.shape[0]
    seg_labels = torch.empty((n, *pairs.shape[-2:]), dtype=torch.long)
    pose_tgt = torch.empty((n, 6), dtype=torch.float32)
    for i in range(0, n, batch):
        b = pairs[i:i + batch].to(device)
        seg_in = segnet.preprocess_input(b)
        seg_labels[i:i + b.shape[0]] = segnet(seg_in).argmax(dim=1).cpu()
        pose_in = posenet.preprocess_input(b)
        pose_tgt[i:i + b.shape[0]] = posenet(pose_in)['pose'][:, :6].cpu()
    return seg_labels, pose_tgt


@torch.no_grad()
def eval_distortion(decoder, latents, pairs, seg_labels, pose_tgt, segnet, posenet,
                    device, batch=16, qat_bits=None):
    """Real (non-surrogate) distortion proxies on the 384x512 recon."""
    decoder.eval()
    n = pairs.shape[0]
    seg_dis, pose_mse, cnt = 0.0, 0.0, 0
    for i in range(0, n, batch):
        pidx = torch.arange(i, min(i + batch, n), device=latents.device)
        recon = decode_pairs(decoder, latents, pidx, qat_bits)  # (B,2,3,H,W)
        B = pidx.shape[0]
        seg_in = segnet.preprocess_input(recon)
        pred = segnet(seg_in).argmax(dim=1).cpu()
        seg_dis += (pred != seg_labels[i:i + B]).float().mean(dim=(1, 2)).sum().item()
        pose_in = posenet.preprocess_input(recon)
        po = posenet(pose_in)['pose'][:, :6].cpu()
        pose_mse += (po - pose_tgt[i:i + B]).pow(2).mean(dim=1).sum().item()
        cnt += B
    decoder.train()
    return seg_dis / cnt, pose_mse / cnt


def _decode(decoder, z, qat_bits):
    if qat_bits is None:
        return decoder(z)
    # temporarily swap in fake-quantized weights. Quantize EVERY param (incl.
    # biases, dim==1) so the QAT grid matches exactly what codec.build_archive
    # packs -> the inflated decoder is byte-exact with the trained one.
    saved = {}
    for name, p in decoder.named_parameters():
        saved[name] = p.data
        p.data = fake_quant(p.data, qat_bits)
    try:
        return decoder(z)
    finally:
        for name, p in decoder.named_parameters():
            if name in saved:
                p.data = saved[name]


def seg_smooth_disagreement(logits, labels, tau=0.3):
    """Smooth surrogate for SegNet argmax-disagreement (what the score measures).

    margin = (target-class logit) - (max other-class logit). Minimizing
    sigmoid(-margin/tau) puts the strongest gradient on pixels whose margin is
    near 0 -- the ones about to flip class -- pushing them back to the correct
    argmax. Directly targets the metric, unlike CE which over-weights easy pixels.
    """
    tgt = logits.gather(1, labels.unsqueeze(1)).squeeze(1)            # (B,H,W)
    other = logits.scatter(1, labels.unsqueeze(1), float('-inf'))
    other_max = other.max(dim=1).values                              # (B,H,W)
    margin = tgt - other_max
    return torch.sigmoid(-margin / tau).mean()


def seg_l7(logits, labels, tau=0.3, hard_thresh=1.0, hard_weight=5.0):
    """L7-weighted smooth disagreement: same smooth surrogate, but pixels whose
    margin < hard_thresh (the near-boundary, about-to-flip ones) get up-weighted
    (renormalized to mean 1). Concentrates gradient where the argmax metric is
    actually decided -> pushes segdis far lower than uniform CE/smooth."""
    tgt = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other = logits.scatter(1, labels.unsqueeze(1), float('-inf'))
    margin = tgt - other.max(dim=1).values
    base = torch.sigmoid(-margin / tau)
    w = torch.where(margin < hard_thresh,
                    torch.as_tensor(hard_weight, device=logits.device),
                    torch.as_tensor(1.0, device=logits.device))
    w = w / w.mean()
    return (base * w).mean()


# ---------------------------------------------------------------------------
# Muon optimizer (Newton-Schulz orthogonalized momentum) -- the lever the #1
# winner credited for the final gains. Applied to hidden conv weight matrices;
# stem/RGB-heads/biases/latents stay on AdamW.
# ---------------------------------------------------------------------------
def _zeropower_newtonschulz5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, momentum=0.95, weight_decay=5e-4, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            mom, lr, wd, ns = grp['momentum'], grp['lr'], grp['weight_decay'], grp['ns_steps']
            for p in grp['params']:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if 'buf' not in st:
                    st['buf'] = torch.zeros_like(g)
                buf = st['buf']
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom)                     # nesterov
                g2d = g.reshape(g.size(0), -1)
                u = _zeropower_newtonschulz5(g2d, ns).reshape_as(p)
                scale = max(1.0, p.size(0) / (p.numel() / p.size(0))) ** 0.5
                p.mul_(1 - lr * wd).add_(u, alpha=-lr * scale)


def decode_pairs(decoder, latents, pair_idx, qat_bits):
    """Decode both frames of each pair from their OWN per-frame latents.

    pair_idx: (B,) pair indices. Frame indices are 2k and 2k+1. Returns the
    reconstructed pair stacked as (B, 2, 3, H, W) for the SegNet/PoseNet path.
    """
    fidx = torch.stack([2 * pair_idx, 2 * pair_idx + 1], dim=1).reshape(-1)  # (2B,)
    flat = _decode(decoder, latents[fidx.to(latents.device)], qat_bits)       # (2B,3,H,W)
    return flat.view(pair_idx.shape[0], 2, *flat.shape[1:])


def train(cfg):
    device = torch.device(cfg.device if cfg.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(cfg.seed)
    print(f"device={device}  config={vars(cfg)}")

    # data
    cache = str(HERE.parent / 'cache' / 'eval_frames.pt')
    pd = PairData(str(DATA_ROOT / 'videos' / '0.mkv'), device=torch.device('cpu'),
                  max_frames=cfg.max_frames, cache=cache if cfg.use_cache else None)
    pairs = pd.pairs  # (n_pairs,2,3,384,512) on cpu
    n_pairs = pd.n_pairs
    print(f"n_pairs={n_pairs}")

    # frozen nets
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    for p in segnet.parameters(): p.requires_grad_(False)
    for p in posenet.parameters(): p.requires_grad_(False)

    print("precomputing gt targets...")
    seg_labels, pose_tgt = precompute_targets(pairs.to(device) if device.type == 'cpu' else pairs,
                                              segnet, posenet, device)

    # model + per-frame latents (one code per frame, n_frames = 2 * n_pairs)
    n_frames = 2 * n_pairs
    decoder = QLRDecoder(cfg.latent_dim, cfg.base_channels, cfg.rank).to(device)
    latents = nn.Parameter(0.01 * torch.randn(n_frames, cfg.latent_dim, device=device))
    print(f"decoder params={count_params(decoder):,}  n_frames={n_frames}  latents={latents.numel():,}")

    opt_muon = None
    if cfg.optimizer == 'muon':
        muon_p, adamw_p = [], []
        for name, p in decoder.named_parameters():
            if p.ndim >= 2 and ('blocks' in name or 'refine' in name):
                muon_p.append(p)
            else:
                adamw_p.append(p)          # stem, rgb heads, all biases
        opt_muon = Muon(muon_p, lr=cfg.lr, momentum=0.95, weight_decay=5e-4)
        opt = torch.optim.AdamW([
            {'params': adamw_p, 'lr': cfg.lr * 0.1},
            {'params': [latents], 'lr': cfg.lr * cfg.latent_lr_mult},
        ], weight_decay=0.0)
        print(f"Muon on {len(muon_p)} conv tensors, AdamW on {len(adamw_p)} + latents")
    else:
        opt = torch.optim.AdamW([
            {'params': decoder.parameters(), 'lr': cfg.lr},
            {'params': [latents], 'lr': cfg.lr * cfg.latent_lr_mult},
        ], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    sched_muon = torch.optim.lr_scheduler.CosineAnnealingLR(opt_muon, cfg.epochs) if opt_muon else None

    run_dir = HERE / 'ckpts' / f'run_{cfg.tag}'
    run_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    ckpt_path = run_dir / 'ckpt.pt'
    if cfg.init_from:  # Phase 2+: warm-start weights from a prior run, fresh schedule
        ck = torch.load(cfg.init_from, map_location=device, weights_only=False)
        decoder.load_state_dict(ck['decoder'])
        with torch.no_grad():
            latents.copy_(ck['latents'].to(device))
        print(f"init-from {cfg.init_from} (epoch {ck.get('epoch')}); fresh fine-tune schedule")
    elif cfg.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        decoder.load_state_dict(ck['decoder'])
        with torch.no_grad():
            latents.copy_(ck['latents'].to(device))
        start_epoch = ck['epoch'] + 1
        for _ in range(start_epoch):
            sched.step()
        print(f"resumed from epoch {start_epoch}")

    # Magnitude pruning: build per-tensor masks from the (warm-started) weights,
    # zero the pruned ones, and keep them zero throughout fine-tuning.
    prune_masks = {}
    if cfg.prune > 0:
        sd = decoder.state_dict()
        with torch.no_grad():
            for k, v in sd.items():
                if v.ndim >= 2 and ('blocks' in k or 'refine' in k):
                    t = v.abs().flatten()
                    thr = t.kthvalue(max(1, int(cfg.prune * t.numel()))).values
                    m = (v.abs() >= thr).float()
                    prune_masks[k] = m
                    v.mul_(m)
        kept = sum(int(m.sum()) for m in prune_masks.values())
        tot = sum(m.numel() for m in prune_masks.values())
        print(f"pruned to {cfg.prune:.0%}: {tot-kept}/{tot} conv weights zeroed")

    def _apply_masks(state):
        for k, m in prune_masks.items():
            state[k].mul_(m)

    # EMA of decoder + latents (smooths the noisy single-video overfit; we eval
    # and archive the EMA, not the raw weights). best.pt keeps the lowest INT-b
    # archive-score state seen -> that's what the final archive is built from.
    use_ema = cfg.ema > 0
    if use_ema:
        live_sd = decoder.state_dict()                          # live refs (share storage)
        ema_dec = {k: v.detach().clone() for k, v in live_sd.items()}
        ema_lat = latents.detach().clone()
    best_proxy = float('inf')
    best_path = run_dir / 'best.pt'

    idx_all = torch.arange(n_pairs)
    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        qat = cfg.qat_bits if epoch >= cfg.qat_start_epoch else None
        perm = idx_all[torch.randperm(n_pairs)]
        decoder.train()
        running = 0.0
        for i in range(0, n_pairs, cfg.batch):
            idx = perm[i:i + cfg.batch]                       # cpu indices for targets
            gt = pairs[idx].to(device)
            recon = decode_pairs(decoder, latents, idx, qat)  # (B,2,3,H,W)

            seg_in = segnet.preprocess_input(recon)
            seg_logits = segnet(seg_in)
            seg_lbl = seg_labels[idx].to(device)
            if cfg.seg_loss == 'l7':
                ce = seg_l7(seg_logits, seg_lbl) + 0.1 * F.cross_entropy(seg_logits, seg_lbl)
            elif cfg.seg_loss == 'smooth':
                ce = seg_smooth_disagreement(seg_logits, seg_lbl) + 0.1 * F.cross_entropy(seg_logits, seg_lbl)
            else:
                ce = F.cross_entropy(seg_logits, seg_lbl)

            pose_in = posenet.preprocess_input(recon)
            pose_out = posenet(pose_in)['pose'][:, :6]
            _mse = F.mse_loss(pose_out, pose_tgt[idx].to(device))
            # Concave sqrt form matches the score's pose term and keeps a large
            # gradient at small MSE (plain MSE's gradient -> 0 and plateaus).
            pmse = torch.sqrt(10 * _mse + 1e-12) if cfg.pose_loss == 'sqrt' else _mse

            # MSE pixel fidelity is the PRIMARY signal: faithful reconstruction is
            # what makes both SegNet and PoseNet outputs match (pose follows fidelity).
            # SegNet CE + PoseNet MSE are light refinements on top.
            pix = F.mse_loss(recon, gt)
            loss = cfg.w_seg * ce + cfg.w_pose * pmse + cfg.w_pix * pix

            opt.zero_grad(set_to_none=True)
            if opt_muon is not None:
                opt_muon.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if opt_muon is not None:
                opt_muon.step()
            if prune_masks:
                with torch.no_grad():
                    _apply_masks(decoder.state_dict())
            if use_ema:
                with torch.no_grad():
                    for k in ema_dec:
                        ema_dec[k].mul_(cfg.ema).add_(live_sd[k], alpha=1 - cfg.ema)
                    ema_lat.mul_(cfg.ema).add_(latents, alpha=1 - cfg.ema)
                    if prune_masks:
                        _apply_masks(ema_dec)
            running += loss.item() * idx.numel()
        sched.step()
        if sched_muon is not None:
            sched_muon.step()

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            # evaluate the EMA weights, ALWAYS at the target bit-depth so the
            # reported score reflects the real INT-b archive (not optimistic FP).
            if use_ema:
                raw_sd = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
                decoder.load_state_dict(ema_dec)
                eval_lat = ema_lat.detach()
            else:
                eval_lat = latents.detach()
            sd, pm = eval_distortion(decoder, eval_lat, pairs, seg_labels,
                                     pose_tgt, segnet, posenet, device,
                                     qat_bits=cfg.qat_bits)
            if use_ema:
                decoder.load_state_dict(raw_sd)             # restore training weights
            proxy = 100 * sd + (10 * pm) ** 0.5             # distortion part of the score
            dt = time.time() - t0
            tag = ''
            if proxy < best_proxy:
                best_proxy = proxy
                save_dec = ema_dec if use_ema else decoder.state_dict()
                torch.save({'decoder': {k: v.detach().cpu() for k, v in save_dec.items()},
                            'latents': eval_lat.cpu(), 'cfg': vars(cfg), 'epoch': epoch,
                            'proxy': proxy}, best_path)
                tag = ' *best*'
            print(f"ep {epoch:4d} loss {running/n_pairs:.4f}  segdis {sd:.5f}  "
                  f"posemse {pm:.6f}  proxy {proxy:.4f}  best {best_proxy:.4f}  qat={qat}  {dt:.0f}s{tag}")
            torch.save({'decoder': decoder.state_dict(), 'latents': latents.detach().cpu(),
                        'cfg': vars(cfg), 'epoch': epoch}, run_dir / 'ckpt.pt')

    # build final archive
    from codec import build_archive
    meta = {'latent_dim': cfg.latent_dim, 'base_channels': cfg.base_channels,
            'rank': cfg.rank, 'eval_size': [384, 512], 'n_pairs': n_pairs,
            'n_frames': n_frames, 'qat_bits': cfg.qat_bits}
    # build from the BEST (lowest archive-score) state captured during training
    if best_path.exists():
        bk = torch.load(best_path, weights_only=False, map_location='cpu')
        final_dec, final_lat = bk['decoder'], bk['latents']
        print(f"building archive from best.pt (epoch {bk['epoch']}, proxy {bk['proxy']:.4f})")
    else:
        final_dec, final_lat = decoder.state_dict(), latents.detach().cpu()
    blob = build_archive(final_dec, final_lat, meta, weight_bits=cfg.qat_bits)
    out_dir = run_dir / 'submission_archive'
    out_dir.mkdir(exist_ok=True)
    (out_dir / '0.bin').write_bytes(blob)
    print(f"wrote archive {len(blob):,} bytes -> {out_dir/'0.bin'}  "
          f"(rate={len(blob)/37545489:.5f}, 25*rate={25*len(blob)/37545489:.4f})")


def get_cfg():
    p = argparse.ArgumentParser()
    p.add_argument('--latent-dim', type=int, default=20)
    p.add_argument('--base-channels', type=int, default=32)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--latent-lr-mult', type=float, default=5.0)
    p.add_argument('--w-seg', type=float, default=1.0)
    p.add_argument('--w-pose', type=float, default=2000.0)
    p.add_argument('--w-pix', type=float, default=0.02)
    p.add_argument('--qat-bits', type=int, default=6)
    p.add_argument('--qat-start-epoch', type=int, default=400)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--tag', type=str, default='dev')
    p.add_argument('--max-frames', type=int, default=None)
    p.add_argument('--use-cache', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--init-from', type=str, default=None,
                   help='warm-start decoder+latents from this ckpt.pt, fresh schedule')
    p.add_argument('--seg-loss', type=str, default='ce', choices=['ce', 'smooth', 'l7'],
                   help="ce; smooth = smooth-disagreement; l7 = hard-pixel-weighted (lowest segdis)")
    p.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'muon'],
                   help="muon = Muon on conv weights + AdamW on the rest (winner's final-gain lever)")
    p.add_argument('--pose-loss', type=str, default='mse', choices=['mse', 'sqrt'],
                   help="sqrt = concave sqrt(10*MSE); keeps pushing pose down at small MSE")
    p.add_argument('--ema', type=float, default=0.0,
                   help="EMA decay for decoder+latents (e.g. 0.999); 0 disables. Smooths the "
                        "noisy single-video overfit; EMA state is what gets evaluated + archived.")
    p.add_argument('--prune', type=float, default=0.0,
                   help="fraction of conv weights to magnitude-prune to zero (e.g. 0.5). The "
                        "surviving weights are fine-tuned to recover; zeros compress far smaller.")
    return p.parse_args()


if __name__ == "__main__":
    train(get_cfg())
