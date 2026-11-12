"""Offline tests for general dual-archive inventory behavior."""
from pathlib import Path

from cme_pipeline import dual_archive


def test_dual_archive_selects_science_and_fallback_beacon(tmp_path, monkeypatch):
    science_only = "20240101_002424_d7c2A.fts"
    shared_same = "20240101_005424_d7c2A.fts"
    shared_different = "20240101_012424_d7c2A.fts"
    beacon_only = "20240101_015424_d7c2A.fts"
    science_d4 = "20240101_002400_d4c2A.fts"

    def fake_download(day, product, directory):
        directory.mkdir(parents=True, exist_ok=True)
        if product == "science_raw":
            files = {
                science_only: b"science-only",
                shared_same: b"same",
                shared_different: b"science-version",
                science_d4: b"high-resolution-science",
            }
        else:
            files = {
                shared_same: b"same",
                shared_different: b"beacon-version",
                beacon_only: b"beacon-only",
            }
        for name, data in files.items():
            (directory / name).write_bytes(data)
        return [directory / name for name in files]

    monkeypatch.setattr(dual_archive, "download_cor2", fake_download)
    selected, inventory, sources = dual_archive.collect_dual_archive(
        ["20240101"], tmp_path
    )
    by_name = {p.name: p for p in selected}
    assert set(by_name) == {
        science_only, shared_same, shared_different, beacon_only
    }
    assert by_name[shared_different].read_bytes() == b"science-version"
    assert by_name[beacon_only].read_bytes() == b"beacon-only"
    assert not (tmp_path / "raw/beacon/20240101" / shared_same).exists()
    assert (tmp_path / "raw/beacon/20240101" / shared_different).exists()
    assert (tmp_path / "raw/science/20240101" / science_d4).exists()
    assert "different_versions_retained" in inventory.read_text()
    assert "identical_removed" in sources.read_text()


def test_dual_archive_rejects_no_d7(tmp_path, monkeypatch):
    def fake_download(day, product, directory):
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / "20240101_002400_d4c2A.fts"
        file.write_bytes(b"science")
        return [file]
    monkeypatch.setattr(dual_archive, "download_cor2", fake_download)
    import pytest
    with pytest.raises(ValueError, match="provided d7"):
        dual_archive.collect_dual_archive(["20240101"], tmp_path)
