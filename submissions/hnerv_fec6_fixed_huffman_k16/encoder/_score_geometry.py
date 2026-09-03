# SPDX-License-Identifier: MIT
"""Vendored slice of `tac.score_geometry` for the bundled sweep tool.

Stdlib-only; mirrors the canonical contest objective:

    S = 100 * d_seg + sqrt(10 * d_pose) + 25 * B / 37_545_489

where 37,545,489 is the uncompressed reference video byte count, `d_seg` is
the SegNet argmax disagreement rate, `d_pose` is the PoseNet MSE on the first
six dimensions, and `B` is the archive byte count.

Reviewers can verify this two-symbol slice line-by-line against the upstream
`evaluate.py` rate term without pulling in any third-party package.
"""
from __future__ import annotations

import math

CONTEST_REFERENCE_BYTES = 37_545_489
SEG_COEFFICIENT = 100.0
POSE_COEFFICIENT_INSIDE_SQRT = 10.0
RATE_COEFFICIENT = 25.0


def contest_score(
    d_seg: float,
    d_pose: float,
    archive_bytes: int,
    *,
    reference_bytes: int = CONTEST_REFERENCE_BYTES,
) -> float:
    """Return the exact contest score for (d_seg, d_pose, archive_bytes).

    >>> contest_score(0.001, 0.0001, 178258)  # doctest: +ELLIPSIS
    0.21037...
    """
    if d_seg < 0.0 or d_pose < 0.0 or archive_bytes < 0:
        raise ValueError("contest score inputs must be non-negative")
    seg_term = SEG_COEFFICIENT * d_seg
    pose_term = math.sqrt(POSE_COEFFICIENT_INSIDE_SQRT * d_pose)
    rate_term = RATE_COEFFICIENT * archive_bytes / reference_bytes
    return seg_term + pose_term + rate_term
