"""Offline migration tests: never delete legacy input or overwrite changed data."""
import os

import pytest

from scripts.migrate_shared_raw import migrate


def test_migration_is_dry_run_then_idempotent(tmp_path):
    old = tmp_path / "runs" / "acquisition" / "raw"
    file = old / "science" / "20101212" / "20101212_002424_d7c2A.fts"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"original")
    dest = tmp_path / "data" / "raw" / "cor2"
    preview = migrate(old, dest)
    assert preview["dry_run"] and preview["discovered"] == 1
    assert not dest.exists()

    moved = migrate(old, dest, apply=True, hardlink=True)
    assert moved["created"] == 1
    copied = dest / "science" / "20101212" / file.name
    assert copied.read_bytes() == file.read_bytes()
    assert copied.stat().st_ino == file.stat().st_ino
    assert file.exists()  # Old absolute manifests still resolve

    again = migrate(old, dest, apply=True, hardlink=True)
    assert again["created"] == 0
    assert again["already_identical"] == 1


def test_migration_rejects_existing_different_file(tmp_path):
    old = tmp_path / "old" / "raw"
    original = old / "beacon" / "20101211" / "20101211_002424_d7c2A.fts"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"one")
    dest = tmp_path / "new"
    conflicting = dest / "beacon" / "20101211" / original.name
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"two")
    with pytest.raises(FileExistsError, match="differs"):
        migrate(old, dest, apply=True)
    assert original.read_bytes() == b"one"
    assert conflicting.read_bytes() == b"two"
