#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Thin wrapper around the bundled encoder. See encoder/README.md for the full
# reproduction recipe: inputs (PR #101 archive + source runtime + per-frame
# sweep artifact dir), the offline scorer-sweep step, and outputs.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: compress.sh --artifact-dir DIR --archive PATH --source-runtime DIR \
                   --output-dir DIR [extra build flags]

Required:
  --artifact-dir DIR     Per-frame SegNet+PoseNet sweep artifact directory.
                         Produce with `encoder/frame_exploit_segnet_posenet_sweep.py`.
  --archive PATH         PR #101 `archive.zip` (fetch from PR #101 release;
                         this submission does not redistribute it).
  --source-runtime DIR   PR #101 runtime tree (submissions/hnerv_ft_microcodec
                         from the PR #101 source bundle).
  --output-dir DIR       Where the bundled encoder writes the rebuilt
                         submission tree (`archive.zip` + `inflate.sh` + runtime).

Extra flags accepted by the encoder pass through. For the submitted archive,
also pass `--overlay-artifact-dir DIR`, `--selector-policy-mode compact_exact_k16`,
`--compact-selector-codec fec6_fixed_huffman_k16`, and
`--allow-nonpositive-charged-proxy`; see encoder/README.md.
USAGE
}

if [[ $# -eq 0 ]]; then usage; exit 64; fi

exec python3 "${HERE}/encoder/build_pr101_frame_exploit_selector_packet.py" "$@"
