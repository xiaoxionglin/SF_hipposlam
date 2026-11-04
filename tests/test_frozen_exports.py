"""Keep the published starter ZIP and its unpacked copy byte-identical.

The export is a versioned handoff, not a live source tree. Format or refactor the
canonical implementation instead of rewriting files covered by its manifest.
"""

import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

EXPORTS = Path(__file__).resolve().parents[1] / "exports"
STARTER = EXPORTS / "intrmotiv_prediction_starter"
ARCHIVE = EXPORTS / "intrmotiv_prediction_starter_v0.1.0.zip"
MANIFEST = "CONTENTS.sha256.json"


def test_frozen_starter_files_match_published_manifest():
    """Catch formatter drift or other edits without importing the old package."""
    manifest = json.loads((STARTER / MANIFEST).read_text())
    assert manifest, "A frozen distribution must declare its payload"
    generated_directories = {"__pycache__", ".pytest_cache"}
    unpacked_members = {
        path.relative_to(STARTER).as_posix()
        for path in STARTER.rglob("*")
        if (path.is_file() or path.is_symlink())
        and not generated_directories.intersection(path.relative_to(STARTER).parts)
    }
    assert unpacked_members == set(manifest) | {MANIFEST}, "Unpacked export differs from declared payload"
    for name, expected_hash in manifest.items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and ".." not in path.parts, name
        assert re.fullmatch(r"[0-9a-f]{64}", expected_hash), name
        source = STARTER / name
        assert source.is_file() and not source.is_symlink(), name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash, name


def test_frozen_starter_archive_matches_complete_manifest():
    """The archive must contain exactly the declared files and original manifest."""
    manifest_bytes = (STARTER / MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    prefix = STARTER.name + "/"
    expected_members = {prefix + name for name in manifest} | {prefix + MANIFEST}
    with zipfile.ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate archive members"
        assert set(names) == expected_members
        assert archive.read(prefix + MANIFEST) == manifest_bytes
        for name, expected_hash in manifest.items():
            assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == expected_hash, name
