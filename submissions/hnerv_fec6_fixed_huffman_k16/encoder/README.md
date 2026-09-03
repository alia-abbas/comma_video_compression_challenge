<!-- SPDX-License-Identifier: MIT -->

# Encoder

This directory contains the offline pipeline that produces this submission's
`archive.zip` from PR [#101](https://github.com/commaai/comma_video_compression_challenge/pull/101)'s
archive plus a precomputed per-frame scorer-sweep.

## Files

| File | Role |
|---|---|
| `frame_exploit_segnet_posenet_sweep.py` | Offline sweep tool. For each of 31 candidate per-frame transforms, runs the transformed pair through the upstream `SegNet` and `PoseNet` scorers and writes per-frame component deltas (Δseg, Δpose) to an artifact directory. |
| `build_pr101_frame_exploit_selector_packet.py` | The encoder. Selects K=16 modes from the sweep table (`--selector-policy-mode compact_exact_k16`), Huffman-codes the per-pair indices against a fixed K=16 codebook (`--compact-selector-codec fec6_fixed_huffman_k16`), and emits the rebuilt submission tree (`archive.zip` + `inflate.sh` + runtime). |
| `_score_geometry.py` | Stdlib-only vendored slice of the canonical contest-score helper. Two symbols: `CONTEST_REFERENCE_BYTES = 37_545_489` + `contest_score(d_seg, d_pose, archive_bytes)`. Used by the sweep tool to rank candidate modes; reviewers can verify it line-by-line against the upstream rate term. |
| `tool_bootstrap.py` | Stdlib-only path helper. |

## Inputs (not bundled)

- **PR #101 archive**: fetch from the PR #101 release. Expected SHA-256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`.
- **PR #101 source runtime**: `submissions/hnerv_ft_microcodec/` from the PR #101 source tree.
- **Upstream contest repo**: `evaluate.py`, `modules.py`, and `videos/0.mkv` from the contest root for the sweep step.
- **Scorer-sweep artifacts**: the submitted archive was built from a full-600
  local sweep plus a 16-pair CPU overlay for the high-impact pairs listed
  below. Regenerating the sweeps on a different backend can change the selected
  policy; the encoder is deterministic for a fixed artifact directory set.

## Reproduce

```bash
# 1) Offline base sweep. Writes per-pair component deltas for all 600 pairs.
python3 encoder/frame_exploit_segnet_posenet_sweep.py \
    --candidate-raw /path/to/pr101/inflated/0.raw \
    --gt-video /path/to/comma_video_compression_challenge/videos/0.mkv \
    --upstream /path/to/comma_video_compression_challenge \
    --baseline-json /path/to/pr101/contest_auth_eval.json \
    --output-dir $TMPDIR/fec6_sweep_artifact

# 1b) Exact submitted policy overlay. These are the 16 high-impact pairs that
#     were rescored on the CPU-local path before final packing.
python3 encoder/frame_exploit_segnet_posenet_sweep.py \
    --candidate-raw /path/to/pr101/inflated/0.raw \
    --gt-video /path/to/comma_video_compression_challenge/videos/0.mkv \
    --upstream /path/to/comma_video_compression_challenge \
    --baseline-json /path/to/pr101/contest_auth_eval.json \
    --pair-indices 502,2,546,515,540,379,535,558,507,581,504,372,320,79,409,410 \
    --n-pairs 16 \
    --batch-size 16 \
    --device cpu \
    --output-dir $TMPDIR/fec6_cpu_top16_overlay

# 2) Selector pack. Reads the sweep table plus overlay, selects K=16 modes, and
#    packs the rebuilt submission tree.
python3 encoder/build_pr101_frame_exploit_selector_packet.py \
    --artifact-dir $TMPDIR/fec6_sweep_artifact \
    --overlay-artifact-dir $TMPDIR/fec6_cpu_top16_overlay \
    --archive /path/to/pr101/archive.zip \
    --source-runtime /path/to/pr101/submissions/hnerv_ft_microcodec \
    --output-dir $TMPDIR/fec6_rebuild \
    --selector-policy-mode compact_exact_k16 \
    --compact-selector-codec fec6_fixed_huffman_k16 \
    --allow-nonpositive-charged-proxy

# (or use the thin wrapper at submissions/hnerv_fec6_fixed_huffman_k16/compress.sh)
```

The rebuilt `archive.zip` will match this submission's SHA-256
(`6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`,
178,517 bytes) when the same PR #101 inputs and the same Huffman codebook are
used with the same base and overlay artifact rows. The selector codebook is
encoder-known and decoder-known and is **not** transmitted in the archive.

## Architecture sketch

Member `x` of the ZIP has the grammar
`FP11 | u32 source_len | source_pr101_payload | u16 selector_len | selector_payload`.
PR #101's payload is read verbatim from `source_pr101_payload`. The
`selector_payload` is the Huffman-coded sequence of per-pair mode indices over
the K=16 alphabet; the decoder (`src/frame_selector.py`) decodes it against the
fixed codebook and dispatches the corresponding inverse transform at
inflate time. Delta versus PR #101's archive: +259 bytes; score delta:
`0.192051 − 0.192840 = −0.000789`.

The HNeRV decoder weights (`src/model.py`) are byte-identical to PR
[#95](https://github.com/commaai/comma_video_compression_challenge/pull/95).
No new training was performed: the PR #101 source payload is reused
byte-for-byte; the bolt-on is the selector + Huffman codebook only.
