# SPDX-License-Identifier: MIT
"""Regression test for the canonical CPU decode SHA-256.

Asserts that `inflate.sh` on the published archive bytes produces a `0.raw`
whose SHA-256 matches the canonical token in `expected_output.sha256`. The
test is opt-in: it requires `torch` + `brotli` to be installed and the
archive bytes to be available locally (either pre-staged at
`ARCHIVE_PATH_ENV` or downloaded by the caller).

Set `PACT_TEST_ARCHIVE_ZIP=/path/to/archive.zip` and
`PACT_TEST_VIDEO_NAMES_FILE=/path/to/public_test_video_names.txt` to run.
Skipped otherwise.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SUBMISSION_DIR = Path(__file__).resolve().parent.parent
EXPECTED_SHA_FILE = SUBMISSION_DIR / "expected_output.sha256"


def _parse_expected_sha256(path: Path) -> str:
    """Return the canonical SHA-256 token recorded in expected_output.sha256."""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Format: "<sha256-hex>  <filename>"
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
            return parts[0]
    raise AssertionError(f"no SHA-256 token found in {path}")


class TestInflateByteStability(unittest.TestCase):

    def test_expected_sha_file_present_and_parseable(self):
        """The canonical reproducibility token must exist and parse."""
        self.assertTrue(EXPECTED_SHA_FILE.exists(), f"missing {EXPECTED_SHA_FILE}")
        sha = _parse_expected_sha256(EXPECTED_SHA_FILE)
        self.assertEqual(len(sha), 64)
        self.assertEqual(sha, sha.lower())

    @unittest.skipUnless(
        os.environ.get("PACT_TEST_ARCHIVE_ZIP") and os.environ.get("PACT_TEST_VIDEO_NAMES_FILE"),
        "set PACT_TEST_ARCHIVE_ZIP and PACT_TEST_VIDEO_NAMES_FILE to run end-to-end",
    )
    def test_inflate_reproduces_canonical_sha(self):
        """End-to-end: inflate.sh on canonical archive yields canonical 0.raw SHA."""
        expected_sha = _parse_expected_sha256(EXPECTED_SHA_FILE)
        archive = Path(os.environ["PACT_TEST_ARCHIVE_ZIP"])
        video_names = Path(os.environ["PACT_TEST_VIDEO_NAMES_FILE"])
        self.assertTrue(archive.is_file())
        self.assertTrue(video_names.is_file())

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            out_dir = Path(tmp) / "out"
            data_dir.mkdir()
            out_dir.mkdir()
            subprocess.run(["unzip", "-oq", str(archive), "-d", str(data_dir)], check=True)
            env = os.environ.copy()
            env["PACT_INFLATE_DEVICE"] = "cpu"
            subprocess.run(
                [str(SUBMISSION_DIR / "inflate.sh"), str(data_dir), str(out_dir), str(video_names)],
                check=True,
                env=env,
            )
            raw_path = out_dir / "0.raw"
            self.assertTrue(raw_path.is_file(), f"inflate did not produce {raw_path}")
            actual_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            self.assertEqual(actual_sha, expected_sha, "decode SHA drift — submission is no longer byte-stable")


if __name__ == "__main__":
    unittest.main()
