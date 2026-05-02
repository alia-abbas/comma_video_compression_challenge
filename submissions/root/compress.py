# Target: 0.mkv. Implements Muon + Asymmetric Loss + QAT.

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import brotli
import io
import struct
from pathlib import Path
from tqdm import tqdm
from inflate import JointFrameGenerator
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Muon Optimizer
# -----------------------------
class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum Orthogonalized Newton.
    Essential for overfitting weights to specific video data by 
    maintaining the internal rank of the weight matrices.
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                # Spectral update for 2D weights
                if p.ndim == 2:
                    X = buf
                    for _ in range(ns_steps):
                        X = 1.5 * X - 0.5 * X @ X.T @ X
                    p.data.add_(X, alpha=-lr)
                else:
                    p.data.add_(buf, alpha=-lr)

# -----------------------------
# Asymmetric Loss Masking
# -----------------------------
def get_asymmetric_loss(pred, gt, mask_labels, model):
    """
    Step 7 Update: Injects QAT (Quantization Aware Training).
    Simulates INT4 precision during the loss calculation.
    """
    # Simulate INT4 quantization error on weights during forward pass
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.ndim >= 2:
                scale = param.abs().max() + 1e-8
                param.copy_(torch.round(param / scale * 7) / 7 * scale)

    # Base pixel-wise MSE
    mse_per_pixel = torch.mean((pred - gt) ** 2, dim=1)

    # Priority Mask (Road=1, Lanes=2, Cars=3)
    weight_mask = torch.full_like(mask_labels, 0.1, dtype=torch.float32)
    priority_classes = [1, 2, 3]
    for cls in priority_classes:
        weight_mask[mask_labels == cls] = 5.0

    return (mse_per_pixel * weight_mask).mean()

# -----------------------------
# The Overfitter Engine
# -----------------------------
def run_overfit(generator, loader, device, run_config):
    """
    REMOVED 'segnet' from arguments because get_asymmetric_loss 
    uses hardcoded priority classes (Road, Cars, etc.)
    """
    muon_params = []
    adamw_params = []
    
    for name, p in generator.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embedding" not in name and "bias" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    opt_muon = Muon(muon_params, lr=run_config['lr_muon'], momentum=0.95)
    opt_adam = torch.optim.AdamW(adamw_params, lr=run_config['lr_adam'], weight_decay=0.01)
    
    generator.to(device)

    for epoch in range(run_config['epochs']):
        generator.train()
        epoch_loss = 0.0
        
        for batch_rgb, gt_mask, gt_pose in loader:
            batch_rgb = batch_rgb.to(device).float() 
            gt_mask = gt_mask.to(device).long() 
            gt_pose = gt_pose.to(device).float()
            
            opt_muon.zero_grad()
            opt_adam.zero_grad()
            
            p1, p2 = generator(gt_mask, gt_pose)
            
            # Using 'generator' as the last argument to match your loss function
            loss1 = get_asymmetric_loss(p1, batch_rgb[:, 0], gt_mask, generator)
            loss2 = get_asymmetric_loss(p2, batch_rgb[:, 1], gt_mask, generator)
            total_loss = loss1 + loss2
            
            total_loss.backward()
            opt_muon.step()
            opt_adam.step()
            
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")
        if avg_loss < 0.0005:
            break
            
    return generator.state_dict()


def polar_quant_pose(pose_tensor):
    """
    Step 7: Converts [x, y] to [r, theta]. 
    Stores theta as int8 (0-255) for maximum compression.
    """
    x, y = pose_tensor[..., 0], pose_tensor[..., 1]
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    
    # Scale theta (-pi to pi) to (0 to 255)
    theta_int8 = (((theta + np.pi) / (2 * np.pi)) * 255).to(torch.uint8)
    return r.to(torch.float16), theta_int8

# -----------------------------
# Packaging Logic
# -----------------------------
def package_submission(state_dict, mask_path, pose_tensor, output_file):
    """
    Step 7: The Hybrid Squeeze Assembly.
    """
    # 1. Final Weight Quantization to INT4
    quantized_payload = {}
    for name, param in state_dict.items():
        if "weight" in name and param.ndim >= 2:
            scale = param.abs().max() + 1e-8
            q_weight = torch.round(param / scale * 7).to(torch.int8)
            quantized_payload[name] = q_weight
            quantized_payload[f"{name}_scale"] = scale.to(torch.float16)
        else:
            quantized_payload[name] = param.to(torch.float16)

    # Serialize Model
    model_buf = io.BytesIO()
    torch.save(quantized_payload, model_buf)
    w_bytes = model_buf.getvalue()

    # 2. PolarQuant the Poses
    r_half, theta_int8 = polar_quant_pose(pose_tensor)
    p_bytes = r_half.cpu().numpy().tobytes() + theta_int8.cpu().numpy().tobytes()

    # 3. Final Brotli Wrap
    m_bytes = open(mask_path, "rb").read()
    header = struct.pack("<III", len(w_bytes), len(m_bytes), len(p_bytes))
    
    full_blob = header + w_bytes + m_bytes + p_bytes
    compressed = brotli.compress(full_blob, quality=11)
    
    with open(output_file, "wb") as f:
        f.write(compressed)
    print(f"Compressed payload saved to {output_file} ({len(compressed)/1024:.2f} KB)")

class VideoDataset(Dataset):
    def __init__(self, video_path, mask_path):
        # Load video frames using PyAV
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            # Convert to RGB and then to a Torch tensor
            img = frame.to_image().convert("RGB")
            frames.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))
        
        self.all_frames = torch.stack(frames) # [Total_Frames, 3, 874, 1164]
        # In a real scenario, you'd load the mask.obu here. 
        # For a CPU test, we'll generate a dummy mask of the right shape.
        self.all_masks = torch.randint(0, 5, (len(self.all_frames), 874, 1164))
        self.pose_tensor = torch.randn(len(self.all_frames), 6) 

    def __len__(self):
        return len(self.all_frames) - 1

    def __getitem__(self, idx):
        return self.all_frames[idx : idx + 2], self.all_masks[idx], self.pose_tensor[idx : idx + 2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str, required=True)
    parser.add_argument('--video-names', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        'epochs': 1000,
        'lr_muon': 0.02,
        'lr_adam': 1e-4,
    }
    
    # Use the paths passed in from the .sh script
    video_path = Path(args.video_dir) / "0.mkv"
    mask_path = Path(args.video_dir) / "mask.obu"
    
    # 1. Initialize Data
    dataset = VideoDataset(str(video_path), str(mask_path))
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 2. Initialize Generator
    generator = JointFrameGenerator().to(device)
    
    # 3. Run Overfit
    print(f"Starting overfit on {video_path}...")
    best_weights = run_overfit(generator, loader, device, config)
    
    # 4. Package
    output_path = Path(args.output_dir) / "payload.bin.br"
    package_submission(
        best_weights, 
        str(mask_path), 
        dataset.pose_tensor, 
        str(output_path)
    )

if __name__ == "__main__":
    main()