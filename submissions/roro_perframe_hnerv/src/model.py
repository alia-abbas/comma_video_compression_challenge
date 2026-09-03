"""Low-rank HNeRV-style decoder (the "QLR" decoder).

Standard HNeRV submissions use *dense* 3x3 convs in every upsample stage, then
quantize the whole state dict to INT8 + brotli. Decoder weights are ~56% of the
frontier score, so we attack them directly with two levers:

  1. PARAMETER EFFICIENCY (this file): every upsample stage factorizes its dense
     3x3 conv into  depthwise-3x3  ->  pointwise 1x1 (in->rank)  ->  pointwise
     1x1 (rank->out*4).  This is a low-rank + separable bottleneck: it cuts the
     channel-mixing cost from  in*out*9  down to  in*9 + in*r + r*out*4, which
     for our channel widths is ~10-15x fewer params per stage with little
     fidelity loss (the per-video signal is low-rank).

  2. LOW-BIT FRIENDLINESS (train.py / codec.py): fewer, smaller tensors quantize
     to INT4-6 and entropy-code far better than a dense INT8 decoder.

Everything else mirrors the proven recipe: 6x8 seed -> 6 PixelShuffle(2) stages
-> 384x512, sin() activations, bilinear skips, separate RGB heads for the two
frames of the pair (frame 2k and 2k+1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LRUpBlock(nn.Module):
    """Upsample block: x2 spatial, in_ch -> out_ch.

    rank > 0  -> low-rank separable (depthwise-3x3 -> 1x1 reduce -> 1x1 expand):
                 cheap params, but the bottleneck caps reconstruction fidelity.
    rank == 0 -> DENSE full 3x3 conv (in -> out*4): more params, much higher
                 fidelity. Faithful reconstruction is what drives PoseNet error
                 down, so the full runs use dense.
    """

    def __init__(self, in_ch: int, out_ch: int, rank: int):
        super().__init__()
        self.dense = (rank == 0)
        if self.dense:
            self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)
        else:
            rank = max(4, min(rank, in_ch, out_ch * 4))
            self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
            self.reduce = nn.Conv2d(in_ch, rank, 1)
            self.expand = nn.Conv2d(rank, out_ch * 4, 1)
        self.ps = nn.PixelShuffle(2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        identity = self.skip(identity)
        if self.dense:
            h = self.conv(x)
        else:
            h = self.expand(self.reduce(self.depthwise(x)))
        return torch.sin(self.ps(h) + identity)


class QLRDecoder(nn.Module):
    """Per-FRAME decoder: one latent -> one RGB frame at eval resolution.

    The earlier per-pair design (one latent + two heads -> both frames) made the
    two frames near-identical, so PoseNet saw no motion and its distortion never
    dropped. Here each frame has its own latent and is decoded independently, so
    consecutive frames differ correctly and inter-frame motion is preserved.
    """

    def __init__(self, latent_dim: int = 20, base_channels: int = 32,
                 rank: int = 16, eval_size=(384, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.rank = rank
        self.eval_size = tuple(eval_size)
        self.base_h, self.base_w = 6, 8
        C = base_channels
        # 6 stages: 6x8 -> 12x16 -> 24x32 -> 48x64 -> 96x128 -> 192x256 -> 384x512
        self.channels = [C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)]

        self.stem = nn.Linear(latent_dim, self.channels[0] * self.base_h * self.base_w)

        self.blocks = nn.ModuleList()
        for i in range(6):
            self.blocks.append(LRUpBlock(self.channels[i], self.channels[i + 1], rank))

        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, 3, padding=1, groups=final_ch),  # cheap depthwise refine
            nn.Conv2d(final_ch, final_ch, 1),
        )
        self.rgb = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z):
        B = z.shape[0]
        x = self.stem(z).view(B, self.channels[0], self.base_h, self.base_w)
        x = torch.sin(x)
        for block in self.blocks:
            x = block(x)
        x = x + 0.1 * torch.sin(self.refine(x))
        return torch.sigmoid(self.rgb(x)) * 255.0  # (B,3,EVAL_H,EVAL_W)


def count_params(m: nn.Module):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for (C, r, L) in [(32, 16, 20), (40, 24, 24), (24, 12, 16)]:
        dec = QLRDecoder(latent_dim=L, base_channels=C, rank=r)
        z = torch.randn(8, L)
        out = dec(z)
        n = count_params(dec)
        stem = dec.stem.weight.numel()
        print(f"C={C} rank={r} L={L}: params={n:,} (stem={stem:,}) out={tuple(out.shape)} "
              f"range=[{out.min():.0f},{out.max():.0f}]")
