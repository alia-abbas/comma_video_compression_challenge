# Third-Party Notices

This submission inherits from prior work in the contest repository and from
two open standards. All upstream code is reused under the contest repository's
MIT license. This document acknowledges the upstream contributions and
identifies the corresponding files in this submission.

## PR #95 — HNeRV decoder

- **Author**: @AaronLeslie138
- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/95
- **License**: MIT (inherited from the contest repository)
- **What this submission uses**: the HNeRV-style decoder architecture (229K
  parameters, per-frame-pair latent → 6 upsample stages → 384×512 RGB pair).
  The decoder code in `src/model.py` is byte-identical to PR #95's
  implementation; no new training was performed.

## PR #101 — `hnerv_ft_microcodec` substrate

- **Author**: @SajayR
- **PR**: https://github.com/commaai/comma_video_compression_challenge/pull/101
- **License**: MIT (inherited from the contest repository)
- **What this submission uses**: the compact decoder schema, latent-payload
  parsing, Brotli source streams, and canonical-Huffman codec for the latent
  sidecar. PR #101's source payload is reused byte-for-byte inside this
  submission's `archive.zip` member `x`. The corresponding inflate-side codec
  primitives live in `src/codec.py` and `src/codec_sidecar.py`. This
  submission's offline encoder fetches PR #101's archive from its release
  (SHA-256: `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`)
  and never redistributes it.

## Lineage acknowledgment

This submission also draws engineering ideas from the broader HNeRV thread on
the contest repository:

- PR #98 (@EthanYangTW): https://github.com/commaai/comma_video_compression_challenge/pull/98
- PR #100 (@BradyMeighan): https://github.com/commaai/comma_video_compression_challenge/pull/100
- PR #102 (@EthanYangTW): https://github.com/commaai/comma_video_compression_challenge/pull/102
- PR #103 (@rem2): https://github.com/commaai/comma_video_compression_challenge/pull/103

## Open standards

- **Brotli** (RFC 7932) — used to decompress the source-payload streams that
  PR #101's grammar emits. Consumed via the `brotli` PyPI package
  (https://github.com/google/brotli, MIT license).
- **Canonical Huffman** — the FEC6 selector adds a fixed 16-symbol Huffman
  codebook for per-pair mode indices over per-frame transforms. The codebook is encoder-known and
  decoder-known; it is **not** transmitted in the archive. Decoder code lives
  in `src/frame_selector.py`.

## This submission's contributions

- `src/frame_selector.py` — FEC6 selector grammar + decoder for the K=16
  fixed-Huffman per-pair mode index. New code.
- `encoder/frame_exploit_segnet_posenet_sweep.py` — offline scorer-sweep tool
  that ranks 31 candidate per-frame transforms against the upstream contest
  scorer. New code.
- `encoder/build_pr101_frame_exploit_selector_packet.py` — encoder that
  selects K=16 modes from the sweep table and packs the submission's
  `archive.zip`. New code.
- `inflate.py` — composes the PR #101 inverse pipeline with the FEC6
  selector's per-pair mode dispatch.

All new code is MIT-licensed under the same terms as the contest repository
(see `LICENSE`).
