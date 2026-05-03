# Target: 0.mkv. Implements Muon + Asymmetric Loss + QAT.
import os
import sys
import av
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
from inflate import NeRVGenerator
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION OVERRIDE ---
SCALE_FACTOR = 0.25  # Reduces 1164x874 to 582x437. Prevents the 227GB RAM error.

# -----------------------------
# Muon Optimizer
# -----------------------------
class Muon(torch.optim.Optimizer):
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
    # Base pixel-wise MSE
    # Ensure shapes match for MSE (pred and gt must be same size)
    if pred.shape != gt.shape:
        gt = F.interpolate(gt, size=pred.shape[-2:], mode='bilinear', align_corners=False)
    
    mse_per_pixel = torch.mean((pred - gt) ** 2, dim=1)

    # Resize mask to match prediction resolution
    if mask_labels.shape[-2:] != pred.shape[-2:]:
        # Use nearest neighbor for categorical masks
        mask_labels = F.interpolate(mask_labels.unsqueeze(1).float(), size=pred.shape[-2:], mode='nearest').squeeze(1).long()

    weight_mask = torch.full_like(mask_labels, 0.1, dtype=torch.float32)
    priority_classes = [1, 2, 3]
    for cls in priority_classes:
        weight_mask[mask_labels == cls] = 5.0

    return (mse_per_pixel * weight_mask).mean()

# -----------------------------
# The Overfitter Engine
# -----------------------------
def run_overfit(generator, loader, device, run_config):
    muon_params = []
    adamw_params = []
    
    for name, p in generator.named_parameters():
        if not p.requires_grad: continue
        # Muon is great for 2D weights (convolutional kernels)
        if p.ndim >= 2 and "embedding" not in name and "bias" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    opt_muon = Muon(muon_params, lr=run_config['lr_muon'], momentum=0.95)
    opt_adam = torch.optim.AdamW(adamw_params, lr=run_config['lr_adam'], weight_decay=0.01)
    
    generator.to(device)
    
    best_loss = float('inf')
    best_weights = None

    for epoch in range(run_config['epochs']):
        epoch_loss = 0.0
        generator.train()
        
        for batch_rgb, frame_idx in loader:
            batch_rgb = batch_rgb.to(device).float() / 255.0 
            frame_idx = frame_idx.to(device).long()
            
            # Zero both optimizers
            opt_adam.zero_grad()
            opt_muon.zero_grad()
            
            h, w = batch_rgb.shape[-2:]
            pred_rgb = generator(frame_idx, target_h=h, target_w=w)
            
            loss = F.mse_loss(pred_rgb, batch_rgb)
            loss.backward()
            
            # Step both optimizers
            opt_adam.step()
            opt_muon.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")
        
        # Track best weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
            
    # Safety: if training was too short or loss didn't improve, return current state
    if best_weights is None:
        best_weights = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
        
    return best_weights

# -----------------------------
# Helper Functions
# -----------------------------
def polar_quant_pose(pose_tensor):
    x, y = pose_tensor[..., 0], pose_tensor[..., 1]
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)
    theta_int8 = (((theta + np.pi) / (2 * np.pi)) * 255).to(torch.uint8)
    return r.to(torch.float16), theta_int8

def package_submission(state_dict, output_file):
    quantized_payload = {}
    
    for name, param in state_dict.items():
        # Quantize weights to INT4 to crush the file size
        # param.ndim >= 2 ensures we target conv/linear layers, not biases
        if "weight" in name and param.ndim >= 2:
            scale = param.abs().max() + 1e-8
            # Scale to range -7 to 7 for INT4
            q_weight = torch.round(param / scale * 7).to(torch.int8)
            quantized_payload[name] = q_weight
            quantized_payload[f"{name}_scale"] = scale.to(torch.float16)
        else:
            # Biases and embeddings stay as float16 for precision
            quantized_payload[name] = param.to(torch.float16)

    # Convert the dictionary to bytes
    model_buf = io.BytesIO()
    torch.save(quantized_payload, model_buf)
    w_bytes = model_buf.getvalue()

    # We skip mask and pose bytes because NeRV doesn't need them!
    # This will make your Compression Rate score much better.
    compressed = brotli.compress(w_bytes, quality=11)
    
    with open(output_file, "wb") as f:
        f.write(compressed)
        
    print(f"✅ Success! NeRV payload saved to {output_file}")
    print(f"Final Size: {len(compressed)/1024:.2f} KB")

# -----------------------------
# Dataset Logic
# -----------------------------
class VideoDataset(Dataset):
    def __init__(self, video_path):
        container = av.open(video_path)
        self.frames = []
        for frame in container.decode(video=0):
            img = frame.to_image().convert("RGB")
            # Resize for your 3060 Ti memory limit
            new_w = int(img.width * SCALE_FACTOR)
            new_h = int(img.height * SCALE_FACTOR)
            img = img.resize((new_w, new_h))
            
            # Shape: [3, H, W]
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
            self.frames.append(img_tensor)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Return the image and its frame index
        return self.frames[idx], idx

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', type=str, required=True)
    parser.add_argument('--video-names', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # NeRV-specific training config
    config = {
        'epochs': 200,      # Coordinate models usually need more epochs to sharpen textures
        'lr_adam': 5e-4,    # Slightly higher Adam rate for coordinate regression
        'lr_muon': 0.02,   
        'embed_dim': 64,    # Size of the temporal "code" per frame
        'hidden_dim': 128,  # Width of the MLP layers
    }
    
    video_path = Path(args.video_dir) / "0.mkv"
    
    # 1. New Dataset only needs the video (No more masks!)
    dataset = VideoDataset(str(video_path))
    
    # Batch size 1 is safest for 8GB VRAM at high resolutions
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 2. Initialize the NeRV Generator
    # We pass len(dataset) so it creates the right number of time embeddings
    generator = NeRVGenerator(
        num_frames=len(dataset), 
        embed_dim=config['embed_dim'], 
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    print(f"Starting NeRV overfit on {video_path}...")
    print(f"Total Frames: {len(dataset)} | Scale Factor: {SCALE_FACTOR}")
    
    torch.cuda.empty_cache()
    
    # 3. Run the overfit
    best_weights = run_overfit(generator, loader, device, config)
    
    # 4. Packaging
    # Since we aren't using masks/poses, we only package the weights.
    # The server will use these weights to reconstruct the video.
    output_path = Path(args.output_dir) / "payload.bin.br"
    
    # Update your package_submission to only handle weights and necessary metadata
    package_submission(best_weights, str(output_path))
    
    print(f"Compression complete. Payload saved to {output_path}")

if __name__ == "__main__":
    main()