import io
import sys
import struct
import av
import brotli
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# -----------------------------
# Architecture: NeRV Coordinate Decoder
# -----------------------------

def dequantize_pot(q_weight, scale):
    """Restores INT4 weights to FP16 for the forward pass."""
    return (q_weight.to(torch.float16) / 7.0) * scale

class NeRVGenerator(nn.Module):
    def __init__(self, num_frames, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.time_embed = nn.Embedding(num_frames, embed_dim)
        self.net = nn.Sequential(
            nn.Conv2d(2 + embed_dim, hidden_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 1)
        )

    def make_grid(self, b, h, w, device):
        ys = torch.linspace(-1, 1, h, device=device)
        xs = torch.linspace(-1, 1, w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)
        return grid.expand(b, -1, -1, -1)

    def forward(self, frame_idx, target_h, target_w):
        b = frame_idx.shape[0]
        device = frame_idx.device
        t_emb = self.time_embed(frame_idx) 
        t_emb = t_emb.view(b, -1, 1, 1).expand(-1, -1, target_h, target_w)
        coords = self.make_grid(b, target_h, target_w, device)
        x = torch.cat([coords, t_emb], dim=1)
        return torch.sigmoid(self.net(x))

# -----------------------------
# Reconstruction Logic
# -----------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <data_dir> <out_dir> <video_list_file>")
        return

    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    video_names = Path(sys.argv[3]).read_text().splitlines()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load and Decompress Payload
    payload_path = data_dir / "payload.bin.br"
    with open(payload_path, "rb") as f:
        compressed_data = f.read()
    
    # Simple decompress (matches your new package_submission)
    weight_bytes = brotli.decompress(compressed_data)
    state_dict = torch.load(io.BytesIO(weight_bytes), map_location=device)

    # 2. Rebuild Model
    # We assume 1200 frames for the challenge video 0.mkv
    num_frames = 1200 
    gen = NeRVGenerator(num_frames=num_frames).to(device)

    # 3. Dequantize Weights
    for name in list(state_dict.keys()):
        if f"{name}_scale" in state_dict:
            scale = state_dict.pop(f"{name}_scale")
            state_dict[name] = dequantize_pot(state_dict[name], scale)
            
    gen.load_state_dict(state_dict, strict=False)
    gen.eval()

    # 4. Generate Video
    for video_name in video_names:
        # Change 0.mkv -> 0.raw so the evaluator finds it
        raw_name = video_name.replace(".mkv", ".raw")
        output_path = out_dir / raw_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        width = 1164
        height = 874

        print(f"Reconstructing {raw_name} at {width}x{height} (Raw YUV)...")

        # Open in binary write mode ('wb')
        with open(output_path, "wb") as f_out:
            with torch.no_grad():
                for i in range(num_frames):
                    f_idx = torch.tensor([i], device=device)
                    pred_rgb = gen(f_idx, height, width)
                    
                    # 1. Convert to CPU and get uint8 RGB
                    img = (pred_rgb[0].permute(1, 2, 0) * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
                    
                    # 2. Use PyAV to get a CLEAN YUV444p frame
                    yuv_frame = av.VideoFrame.from_ndarray(img, format='rgb24').reformat(width, height, 'yuv444p')
                    
                    # 3. MANUALLY extract the planes to ignore alignment padding
                    # This ensures exactly 1164 * 874 bytes per plane
                    # 3. MANUALLY extract the planes to ignore alignment padding
                    for p in range(3):
                        plane = yuv_frame.planes[p]
                        # Calculate the expected number of bytes for this plane width
                        # Plane 0, 1, and 2 are all full width for YUV444p
                        expected_stride = plane.line_size
                        
                        # Convert plane to numpy array, then slice off the padding
                        data = np.frombuffer(plane, dtype=np.uint8).reshape(height, expected_stride)
                        clean_plane = data[:, :width].tobytes()
                        f_out.write(clean_plane)

if __name__ == "__main__":
    main()