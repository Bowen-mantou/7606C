"""Phase 14 regression tests: code revision and dataset identity in metadata."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dp_manip import metadata


ROOT = Path(__file__).resolve().parents[1]


class GitRevisionTest(unittest.TestCase):
    def test_repository_revision_is_a_commit(self) -> None:
        revision = metadata.git_revision(ROOT)
        if revision["commit"] is None:
            self.skipTest("git metadata unavailable")
        self.assertRegex(revision["commit"], r"^[0-9a-f]{40}$")
        self.assertTrue(revision["branch"])
        self.assertIsInstance(revision["dirty"], bool)

    def test_missing_git_is_reported_as_none(self) -> None:
        with mock.patch.object(metadata.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(
                metadata.git_revision(ROOT), {"commit": None, "branch": None, "dirty": None}
            )


class DatasetFingerprintTest(unittest.TestCase):
    def test_fingerprint_covers_the_sidecar_and_file_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.h5"
            path.write_bytes(b"h5-bytes")
            sidecar = '{"episodes": [{"episode_id": 0, "episode_seed": 7}]}\n'
            path.with_suffix(".json").write_text(sidecar, encoding="utf-8")
            info = SimpleNamespace(path=path)

            fingerprint = metadata.dataset_fingerprint(info)
            self.assertEqual(fingerprint["h5_size_bytes"], 8)
            self.assertEqual(
                fingerprint["sidecar_sha256"], hashlib.sha256(sidecar.encode("utf-8")).hexdigest()
            )
            # Equal input -> equal fingerprint; changed metadata -> changed hash.
            self.assertEqual(fingerprint, metadata.dataset_fingerprint(info))
            path.with_suffix(".json").write_text(sidecar + " ", encoding="utf-8")
            self.assertNotEqual(
                fingerprint["sidecar_sha256"],
                metadata.dataset_fingerprint(info)["sidecar_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
