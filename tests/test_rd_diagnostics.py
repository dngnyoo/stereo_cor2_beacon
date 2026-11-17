"""Offline checks for opt-in RD diagnostics (no FITS, no network)."""
from __future__ import annotations

import csv
import json
import sys
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from cme_pipeline.__main__ import parse_args
from cme_pipeline.rd_diagnostics import save_rd_diagnostics


def fixture_run(tmp_path):
    root = tmp_path / "preprocessed"
    root.mkdir()
    (root / "preprocessing_metadata.json").write_text(json.dumps({
        "requested_date": "20240509", "unit": "DN/s"
    }))
    rows = [
        ("2024-05-08T23:53:54", 1),
        ("2024-05-09T00:23:54", 2),
        ("2024-05-09T00:53:54", 3),
        ("2024-05-09T01:23:54", 4),
        ("2024-05-09T01:53:54", 5),
        ("2024-05-09T02:23:54", 6),
    ]
    with (root / "01_rd_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_index", "end_time"])
        writer.writeheader()
        writer.writerows({"pair_index": p, "end_time": t} for t, p in rows)
    rd = np.zeros((6, 256, 256), dtype=np.float32)
    rd[:, 128, 128] = 100.
    valid = np.ones_like(rd, dtype=bool)
    valid[:, 0:25, :] = False
    np.save(root / "north_up_RD_DNs_per_second.npy", rd)
    np.save(root / "north_up_pair_valid_masks.npy", valid)
    return root


def test_default_samples_evaluation_date_only_and_preserves_raw(tmp_path):
    root = fixture_run(tmp_path)
    before = (root / "north_up_RD_DNs_per_second.npy").read_bytes()
    paths = save_rd_diagnostics(root)
    assert len(paths) == 3
    assert all("20240509" in p.name for p in paths)
    assert [int(p.name.split("_")[1]) for p in paths] == [2, 4, 6]
    for p in paths:
        assert p.is_file()
        with Image.open(p) as image:
            assert image.size == (864, 840)
    assert (root / "north_up_RD_DNs_per_second.npy").read_bytes() == before
    with pytest.raises(FileExistsError):
        save_rd_diagnostics(root)


def test_all_frames_and_time_filter(tmp_path):
    root = fixture_run(tmp_path)
    paths = save_rd_diagnostics(root, all_frames=True)
    assert len(paths) == 5
    assert all(p.name.endswith("_rd.png") for p in paths)


def test_bad_date_and_missing_frames(tmp_path):
    root = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="No RD pairs"):
        save_rd_diagnostics(root, date="20240510")
    with pytest.raises(ValueError, match="YYYYMMDD"):
        save_rd_diagnostics(root, date="May9")
    with pytest.raises(ValueError, match="positive"):
        save_rd_diagnostics(root, max_frames=0)


def test_cli_parse_opt_in_modes():
    with patch.object(sys, "argv", [
        "cme_pipeline", "prepare-data", "--date", "20240509",
        "--source", "local", "--input-dir", "/tmp/fits",
        "--preprocess", "--visualize-rd", "--rd-all-frames",
    ]):
        args = parse_args()
    assert args.visualize_rd and args.rd_all_frames
    with patch.object(sys, "argv", [
        "cme_pipeline", "visualize-rd",
        "--preprocessed-dir", "/tmp/preprocessed", "--max-frames", "4",
    ]):
        args = parse_args()
    assert args.command == "visualize-rd"
    assert args.max_frames == 4
