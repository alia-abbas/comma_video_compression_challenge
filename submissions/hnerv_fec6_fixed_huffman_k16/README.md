<!-- SPDX-License-Identifier: MIT -->

# hnerv_fec6_fixed_huffman_k16

Per-pair perturbation selector bolt-on over PR [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101)
`hnerv_ft_microcodec`. Decoder weights byte-identical to PR
[#95](https://github.com/commaai/comma_video_compression_challenge/pull/95).
No new training was performed.

## Archive identity

| Field | Value |
|---|---|
| Score (CPU) | `0.192051 [contest-CPU]` |
| Score (CUDA T4) | `0.226210 [contest-CUDA T4]` |
| Archive SHA-256 | `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf` |
| Archive bytes | `178517` |
| ZIP members | 1 (`x`, `compression_type=0`, 178417 bytes) |
| Inflate runtime deps | `torch`, `brotli` (see `requirements.txt`) |
| Inflate GPU required | no |

## Quick reproducibility check (≈60 s, CPU only)

```bash
# 1) fetch the archive (or use a local copy)
curl -L -o $TMPDIR/archive.zip \
  https://github.com/adpena/comma_video_compression_challenge/releases/download/fec6-frontier-submission-20260520/archive.zip
shasum -a 256 $TMPDIR/archive.zip
# expect: 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf

# 2) inflate against this submission_dir on CPU
mkdir -p $TMPDIR/data $TMPDIR/out
unzip -oq $TMPDIR/archive.zip -d $TMPDIR/data
echo "0.mkv" > $TMPDIR/list.txt
PACT_INFLATE_DEVICE=cpu ./inflate.sh $TMPDIR/data $TMPDIR/out $TMPDIR/list.txt

# 3) verify byte-stable decode against the canonical SHA in expected_output.sha256
shasum -a 256 $TMPDIR/out/0.raw
# expect: d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c
```

## Files

| Path | Role |
|---|---|
| `compress.sh` | Encoder wrapper. See `encoder/README.md` for the full reproduction recipe. |
| `inflate.sh`, `inflate.py` | Contest-runtime decoder. |
| `src/model.py` | HNeRV decoder (byte-identical to PR #95). |
| `src/codec.py`, `src/codec_sidecar.py` | PR #101 source-payload parsing (reused unchanged). |
| `src/frame_selector.py` | FEC6 K=16 fixed-Huffman per-pair selector (new). |
| `encoder/` | Offline sweep + packet builder. |
| `requirements.txt` | Inflate-time runtime dependencies. |
| `expected_output.sha256` | Canonical CPU decode SHA for byte-stability verification. |
| `LICENSE` | MIT, sole-author Alejandro Peña. |
| `THIRD_PARTY_NOTICES.md` | Upstream attribution (PR #95, PR #101, Brotli, canonical Huffman). |
| `tests/` | Optional regression test for the canonical decode SHA. |

## Reproducibility note: CPU vs CUDA

The archive bytes are deterministic. Decoded RGB frames are bit-stable on a
given device, but **differ across CPU and CUDA** because `F.interpolate(...,
mode='bicubic')` and the `clamp/round/uint8` cast are not bit-identical across
the two backends. Both axes are scored on the same `archive.zip` bytes; CPU is
the leaderboard axis and is reported as `0.192051`. The paired Tesla T4 score
is reported separately as `0.226210 [contest-CUDA T4]`.

## Full reproduction recipe

See `encoder/README.md`. Inputs (not bundled): PR #101's `archive.zip` from
its release, PR #101's `submissions/hnerv_ft_microcodec/` source runtime, and
the upstream contest video at `videos/0.mkv` for the offline scorer-sweep
step.

## Upstream attribution

`THIRD_PARTY_NOTICES.md`.
