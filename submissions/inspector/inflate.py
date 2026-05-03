#!/usr/bin/env python
"""
inflate.py — Decompress archive.br → 0.raw

Raw format expected by TensorVideoDataset:
  - N frames, each exactly 874 * 1164 * 3 = 3,054,888 bytes
  - uint8 RGB, no padding, no headers
  - N must be 1200 so TensorVideoDataset produces 600 sequences (pairs of 2)
    which matches AVVideoDataset's 600 sequences from the original video
"""
import sys, av, brotli, numpy as np
from pathlib import Path

def main():
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <archive_dir> <output_dir> <file_list>")
        sys.exit(1)

    archive_dir = Path(sys.argv[1])
    output_dir  = Path(sys.argv[2])
    file_list   = Path(sys.argv[3]).read_text().splitlines()
    output_dir.mkdir(parents=True, exist_ok=True)

    br_path  = archive_dir / 'archive.br'
    ivf_path = output_dir  / '_temp.ivf'

    # ── 1. Decompress brotli → IVF ────────────────────────────────────────────
    print("Decompressing archive.br…")
    ivf_path.write_bytes(brotli.decompress(br_path.read_bytes()))

    # ── 2. Decode video → raw RGB frames ─────────────────────────────────────
    # One .raw file per video listed in file_list
    # (challenge has one video, but we loop to be safe)
    TARGET_W, TARGET_H = 1164, 874
    BYTES_PER_FRAME    = TARGET_H * TARGET_W * 3   # 3,054,888

    container = av.open(str(ivf_path))

    for video_name in file_list:
        if not video_name.strip():
            continue
        raw_name = Path(video_name.strip()).stem + '.raw'
        raw_path = output_dir / raw_name

        print(f"Decoding → {raw_path}")
        written = 0

        with open(raw_path, 'wb') as f:
            for frame in container.decode(video=0):
                # Reformat to rgb24 at exact evaluation resolution
                # PyAV handles YUV→RGB conversion correctly here
                rgb_frame = frame.reformat(
                    width=TARGET_W, height=TARGET_H, format='rgb24'
                )
                # to_ndarray gives (H, W, 3) uint8, contiguous, no padding
                arr = rgb_frame.to_ndarray()          # shape (874, 1164, 3)
                assert arr.nbytes == BYTES_PER_FRAME, \
                    f"frame {written}: unexpected size {arr.nbytes}"
                f.write(arr.tobytes())
                written += 1

        print(f"  Wrote {written} frames  "
              f"({written * BYTES_PER_FRAME / 1e9:.3f} GB)")

        if written != 1200:
            print(f"  WARNING: expected 1200 frames, got {written}.")
            print(f"  TensorVideoDataset will produce {written // 2} sequences "
                  f"but AVVideoDataset produces 600 — mismatch!")

    container.close()
    ivf_path.unlink()   # clean up temp file
    print("Done.")

if __name__ == '__main__':
    main()