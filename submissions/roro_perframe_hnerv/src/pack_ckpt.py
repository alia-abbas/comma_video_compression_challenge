"""Build archive.zip from a training checkpoint (for mid-run evaluation).

Usage: python -m submissions.roro_perframe_hnerv.src.pack_ckpt <ckpt.pt> [weight_bits]
Writes submissions/roro_perframe_hnerv/archive.zip ready for evaluate.sh.
"""
import sys, os, subprocess
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from codec import build_archive

SUB = HERE.parent  # submissions/roro_perframe_hnerv


def prune_decoder(sd, frac):
    """Magnitude-prune `frac` of each conv weight tensor to zero (smallest |w|).
    Zeros compress far better -> smaller archive."""
    out = {}
    for k, v in sd.items():
        v = v.clone()
        if frac > 0 and v.ndim >= 2 and ('blocks' in k or 'refine' in k):
            t = v.abs().flatten()
            thr = t.kthvalue(max(1, int(frac * t.numel()))).values
            v = torch.where(v.abs() < thr, torch.zeros_like(v), v)
        out[k] = v
    return out


def main():
    ckpt = sys.argv[1]
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    c = torch.load(ckpt, weights_only=False)
    cfg, lat = c['cfg'], c['latents']
    c['decoder'] = prune_decoder(c['decoder'], frac)
    nf = lat.shape[0]
    meta = {'latent_dim': cfg['latent_dim'], 'base_channels': cfg['base_channels'],
            'rank': cfg['rank'], 'eval_size': [384, 512], 'n_pairs': nf // 2,
            'n_frames': nf, 'qat_bits': bits}
    blob = build_archive(c['decoder'], lat, meta, weight_bits=bits)
    od = SUB / 'src' / 'ckpts' / 'run_pack' / 'submission_archive'
    od.mkdir(parents=True, exist_ok=True)
    (od / '0.bin').write_bytes(blob)
    zip_path = SUB / 'archive.zip'
    if zip_path.exists():
        zip_path.unlink()
    subprocess.run(['zip', '-j', str(zip_path), str(od / '0.bin')], check=True,
                   stdout=subprocess.DEVNULL)
    rate = len(blob) / 37545489
    print(f"ckpt epoch {c.get('epoch')}  bits {bits}  archive {len(blob)} bytes  "
          f"25*rate {25*rate:.4f}  -> {zip_path}")


if __name__ == "__main__":
    main()
