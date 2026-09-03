"""Archive codec: pack the QLR decoder + per-pair latents into a tiny blob.

Decoder weights: per-tensor symmetric INT-b quantization (b = weight_bits, e.g.
4-6, vs the field's fixed INT8) -> zigzag -> brotli(11). Lower bit-depth on a
low-rank decoder is the core rate win.

Latents: per-dim uint8 min/max scale -> 1st-order temporal delta -> zigzag ->
lo/hi byte split -> brotli(11). (Driving latents drift slowly, so deltas are
tiny and compress well.)

Round-trip is bit-exact: train.py packs the SAME quantization grid it trained
under (QAT), so the decoded weights equal the fake-quant weights seen in
training -> no quantization surprise at inflate time.
"""
import io
import json
import struct

import brotli
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------
def quantize_tensor(t: torch.Tensor, bits: int):
    n_levels = 2 ** (bits - 1) - 1
    t = t.detach().cpu().float()
    m = t.abs().max().item()
    scale = m / n_levels if m > 0 else 1.0
    q = (t / scale).round().clamp(-n_levels, n_levels).to(torch.int32).numpy().flatten()
    return q.astype(np.int32), scale


def zigzag_i32_to_u32(a):
    return np.where(a >= 0, 2 * a, -2 * a - 1).astype(np.uint32)


def unzigzag_u32_to_i32(a):
    a = a.astype(np.int64)
    return np.where(a % 2 == 0, a // 2, -(a // 2) - 1).astype(np.int32)


# ---------------------------------------------------------------------------
# Decoder weights
# ---------------------------------------------------------------------------
def encode_decoder(sd, bits):
    buf = io.BytesIO()
    buf.write(struct.pack("<II", len(sd), bits))
    for name, tensor in sd.items():
        q, scale = quantize_tensor(tensor, bits)
        nb = name.encode('utf-8')
        buf.write(struct.pack("<I", len(nb))); buf.write(nb)
        shape = tuple(tensor.shape)
        buf.write(struct.pack("<I", len(shape)))
        for s in shape:
            buf.write(struct.pack("<I", s))
        buf.write(struct.pack("<f", scale))
        buf.write(struct.pack("<I", q.size))
        # zigzag then pack as uint8 (INT<=8 fits) for compactness
        zz = zigzag_i32_to_u32(q)
        buf.write(zz.astype(np.uint8).tobytes())
    return brotli.compress(buf.getvalue(), quality=11)


def decode_decoder(data):
    raw = brotli.decompress(data)
    buf = io.BytesIO(raw)
    n, bits = struct.unpack("<II", buf.read(8))
    sd = {}
    for _ in range(n):
        nl = struct.unpack("<I", buf.read(4))[0]
        name = buf.read(nl).decode('utf-8')
        nd = struct.unpack("<I", buf.read(4))[0]
        shape = tuple(struct.unpack("<I", buf.read(4))[0] for _ in range(nd))
        scale = struct.unpack("<f", buf.read(4))[0]
        size = struct.unpack("<I", buf.read(4))[0]
        zz = np.frombuffer(buf.read(size), dtype=np.uint8).astype(np.uint32)
        q = unzigzag_u32_to_i32(zz)
        sd[name] = torch.from_numpy(q.astype(np.float32).reshape(shape)) * scale
    return sd


# ---------------------------------------------------------------------------
# Latents
# ---------------------------------------------------------------------------
def encode_latents(latents: torch.Tensor):
    t = latents.detach().cpu().float()
    n, d = t.shape
    mins = t.min(dim=0).values
    maxs = t.max(dim=0).values
    scales = ((maxs - mins) / 254.0).clamp(min=1e-10)
    q = ((t - mins) / scales).round().clamp(0, 254).to(torch.uint8).numpy()
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0]
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    lo = (zz & 0xFF).astype(np.uint8).tobytes()
    hi = (zz >> 8).astype(np.uint8).tobytes()
    payload = struct.pack("<II", n, d)
    payload += mins.to(torch.float16).numpy().tobytes()
    payload += scales.to(torch.float16).numpy().tobytes()
    payload += lo + hi
    return brotli.compress(payload, quality=11)


def decode_latents(data):
    raw = brotli.decompress(data)
    buf = io.BytesIO(raw)
    n, d = struct.unpack("<II", buf.read(8))
    mins = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    scales = torch.from_numpy(np.frombuffer(buf.read(d * 2), dtype=np.float16).copy()).float()
    total = n * d
    lo = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(buf.read(total), dtype=np.uint8).astype(np.uint16)
    zz = ((hi << 8) | lo).reshape(n, d)
    delta = np.where(zz % 2 == 0, zz.astype(np.int32) // 2,
                     -(zz.astype(np.int32) // 2) - 1).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    q = q.astype(np.uint8)
    return torch.from_numpy(q.astype(np.float32)) * scales + mins


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------
def build_archive(decoder_sd, latents, meta, weight_bits=6):
    meta_blob = brotli.compress(json.dumps(meta).encode('utf-8'), quality=11)
    dec_blob = encode_decoder(decoder_sd, weight_bits)
    lat_blob = encode_latents(latents)
    out = io.BytesIO()
    for blob in (meta_blob, dec_blob, lat_blob):
        out.write(struct.pack("<I", len(blob)))
        out.write(blob)
    return out.getvalue()


def parse_archive(archive_bytes):
    buf = io.BytesIO(archive_bytes)

    def _read():
        ln = struct.unpack("<I", buf.read(4))[0]
        return buf.read(ln)

    meta = json.loads(brotli.decompress(_read()))
    decoder_sd = decode_decoder(_read())
    latents = decode_latents(_read())
    return decoder_sd, latents, meta


if __name__ == "__main__":
    # round-trip self-test
    from model import QLRDecoder
    dec = QLRDecoder(20, 32, 16)
    lat = torch.randn(600, 20)
    meta = {'latent_dim': 20, 'base_channels': 32, 'rank': 16,
            'eval_size': [384, 512], 'n_pairs': 600, 'qat_bits': 6}
    for bits in (4, 5, 6, 8):
        blob = build_archive(dec.state_dict(), lat, meta, weight_bits=bits)
        sd2, lat2, m2 = parse_archive(blob)
        # check decoder keys round-trip and shapes match
        ok = all(torch.equal(torch.as_tensor(sd2[k].shape), torch.as_tensor(v.shape))
                 for k, v in dec.state_dict().items())
        print(f"bits={bits}: archive={len(blob):,} bytes  rate={len(blob)/37545489:.5f}  "
              f"25*rate={25*len(blob)/37545489:.4f}  keys_ok={ok}")
