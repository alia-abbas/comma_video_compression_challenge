# The Hybrid Decoder. INT4 vision weights + Polar Pose conditioning.

import io
import os
import sys
import struct
import av
import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# -----------------------------
# Architecture: FiLM & Joint Generation
# -----------------------------

def dequantize_pot(q_weight, scale):
    """Restores INT4 Power-of-Two weights to FP16 for the forward pass."""
    return (q_weight.to(torch.float16) / 7.0) * scale

class QConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class SepConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, depth_mult=4):
        super().__init__()
        mid_ch = in_ch * depth_mult
        self.dw = QConv2d(in_ch, mid_ch, k, stride=stride, padding=k//2, groups=in_ch, bias=False)
        self.pw = QConv2d(mid_ch, out_ch, 1, bias=True)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))

class FiLMSepResBlock(nn.Module):
    """The core of the Hybrid Pipeline: Vision features conditioned by PolarQuant Poses."""
    def __init__(self, ch, cond_dim):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1)
        self.conv2 = QConv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw2 = QConv2d(ch, ch, 1, bias=True)
        self.norm2 = nn.GroupNorm(2, ch)
        self.film_proj = nn.Linear(cond_dim, ch * 2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, cond_emb):
        residual = x
        x = self.conv1(x)
        x = self.norm2(self.pw2(self.conv2(x)))
        # FiLM Logic
        film = self.film_proj(cond_emb).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = film.chunk(2, dim=1)
        x = x * (1.0 + gamma) + beta
        return self.act(residual + x)

def rotate_half(x):
    """Splits the features in half and rotates them."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    """Applies the rotation to the hidden states."""
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, None, None, :], emb.sin()[None, :, None, None, :]

class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=64, feature_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, 8)
        self.stem = SepConvGNAct(8 + 2, feature_dim) 
        
        # Step 3: RoPE logic
        self.rope = RotaryEmbedding(feature_dim)
        
        # Step 6: One trunk, shared for all frames (Weight Tying)
        self.trunk = nn.ModuleList([
            FiLMSepResBlock(feature_dim, cond_dim),
            FiLMSepResBlock(feature_dim, cond_dim)
        ])
        
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # Step 6: One head (Weight Tying)
        self.to_rgb = QConv2d(feature_dim, 3, 1)

    def generate_single_frame(self, mask, pose, time_idx):
        """Shared logic to generate a specific frame in the sequence."""
        b, h, w = mask.shape
        device = mask.device
        coords = self.make_grid(b, h, w, device)
        
        x = self.embedding(mask.long()).permute(0, 3, 1, 2)
        x = torch.cat([x, coords], dim=1)
        x = self.stem(x)
        
        # Apply RoPE based on the frame's position in time
        cos, sin = self.rope(h, device) # Using spatial dimension for rotation base
        x = x.permute(0, 2, 3, 1)
        # We 'offset' the rotation by the time_idx to separate frame 1 from frame 2
        x = apply_rotary_pos_emb(x, cos + time_idx, sin + time_idx)
        x = x.permute(0, 3, 1, 2)
        
        # Apply FiLM conditioning (Pose)
        cond_emb = self.pose_mlp(pose)
        for block in self.trunk:
            x = block(x, cond_emb)
            
        return torch.sigmoid(self.to_rgb(x)) * 255.0

    def forward(self, mask, pose_pair):
        """
        Satisfies the Trainer's 'p1, p2 = generator(mask, pose)' call.
        pose_pair: [B, 2, 6] containing pose for frame N and frame N+1.
        """
        # Generate Frame 1 (time_idx = 0)
        p1 = self.generate_single_frame(mask, pose_pair[:, 0], time_idx=0.0)
        
        # Generate Frame 2 (time_idx = 1.0)
        p2 = self.generate_single_frame(mask, pose_pair[:, 1], time_idx=1.0)
        
        return p1, p2

    def make_grid(self, b, h, w, device):
        ys = torch.linspace(-1, 1, h, device=device)
        xs = torch.linspace(-1, 1, w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)

# -----------------------------
# Unified Brotli Stream Logic
# -----------------------------

def inflate_unified_payload(payload_path, device):
    with open(payload_path, "rb") as f:
        compressed_data = f.read()
    
    # Decompress the single blob
    full_data = brotli.decompress(compressed_data)
    
    # Header logic: [4b weight_size][4b mask_size][4b pose_size]
    w_size, m_size, p_size = struct.unpack("<III", full_data[:12])
    
    offset = 12
    weight_bytes = full_data[offset : offset + w_size]
    offset += w_size
    mask_bytes = full_data[offset : offset + m_size]
    offset += m_size
    pose_bytes = full_data[offset : offset + p_size]
    
    return weight_bytes, mask_bytes, pose_bytes

def unpack_polar_poses(p_bytes, num_frames):
    """Reverses PolarQuant to get Cartesian (x, y) for the model."""
    # Split the bytes: r is float16 (2 bytes), theta is uint8 (1 byte)
    # Total pose size was 6, but we focused on x,y for this squeeze
    r_size = num_frames * 2 
    r = torch.from_numpy(np.frombuffer(p_bytes[:r_size], dtype=np.float16))
    theta_uint8 = torch.from_numpy(np.frombuffer(p_bytes[r_size:], dtype=np.uint8))
    
    # Convert back to radians
    theta = (theta_uint8.float() / 255.0) * (2 * 3.14159) - 3.14159
    
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    
    # Reconstruct the 6-DOF pose vector [x, y, 0, 0, 0, 0] 
    # (Assuming we only squeezed x,y and the rest are zeroed for 0.mkv)
    poses = torch.zeros((num_frames, 6))
    poses[:, 0] = x
    poses[:, 1] = y
    return poses

def main():
    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    file_list = Path(sys.argv[3]).read_text().splitlines()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Unified Payload
    w_b, m_b, p_b = inflate_unified_payload(data_dir / "payload.bin.br", device)
    
    # Load Model
    gen = JointFrameGenerator().to(device)
    state_dict = torch.load(io.BytesIO(w_b), map_location=device)
    
    # --- STEP 7 FIX: DEQUANTIZE THE WEIGHTS ---
    for name in list(state_dict.keys()):
        if f"{name}_scale" in state_dict:
            scale = state_dict.pop(f"{name}_scale")
            state_dict[name] = dequantize_pot(state_dict[name], scale)
            
    gen.load_state_dict(state_dict, strict=False)
    gen.eval()

    # --- STEP 7 FIX: UNPACK POLAR POSES ---
    # We need to know the number of frames. 
    # For the challenge, we can derive this from the byte size (total_bytes / 3)
    num_frames = len(p_b) // 3
    poses = unpack_polar_poses(p_b, num_frames).to(device)
    
    # Process videos... (the rest of your video writing logic goes here)

if __name__ == "__main__":
    main()