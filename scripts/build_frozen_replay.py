#!/usr/bin/env python3
"""Build isolated 60-pair C0 replay from a larger dual-archive RD run.

This is an explicitly frozen-specific regression fixture, not a restriction
on general download/cadence selection. The original 63-pair RD arrays and
manifests are never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def identity(start, end):
    return (Path(str(start)).name, Path(str(end)).name)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or [])


def build(repo: Path, source: Path, destination: Path):
    repo, source, destination = repo.resolve(), source.resolve(), destination.resolve()
    reference = (repo / "validation/frozen/20101212/experiments/"
                 "01_d7_baseline_segmentation/results/00_manifest/01_rd_manifest.csv")
    frozen, _ = read_rows(reference)
    current, fields = read_rows(source / "01_rd_manifest.csv")
    if len(frozen) != 60:
        raise ValueError(f"Expected 60 frozen reference pairs, found {len(frozen)}")
    mapping = {}
    for index, item in enumerate(current):
        key = identity(item["start_file"], item["end_file"])
        if key in mapping:
            raise ValueError(f"Duplicate new RD pair: {key}")
        mapping[key] = index
    expected = [identity(item["prev_file"], item["curr_file"]) for item in frozen]
    if len(set(expected)) != len(expected):
        raise ValueError("Duplicate frozen reference pair")
    missing = [key for key in expected if key not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} frozen pairs missing (first: {missing[0]})")
    indices = [mapping[key] for key in expected]
    selected = []
    for n, index in enumerate(indices, 1):
        old = frozen[n - 1]
        row = dict(current[index])
        if str(row["date_str"]) != str(old["date_str"]):
            raise ValueError(f"Date mismatch at frozen row {n}")
        row["pair_index"] = str(n)
        row["rd_pair_id"] = str(old["rd_pair_id"])
        selected.append(row)
    # Fail before output writes if reference pairs aren't in chronological order.
    times = [row["end_time"] for row in selected]
    if times != sorted(times) or len(times) != len(set(times)):
        raise ValueError("Frozen pair order is not strictly chronological")
    rd = np.load(source / "north_up_RD_DNs_per_second.npy", mmap_mode="r")
    valid = np.load(source / "north_up_pair_valid_masks.npy", mmap_mode="r")
    if rd.shape != valid.shape or len(rd) != len(current) or rd.ndim != 3:
        raise ValueError("Source RD arrays and manifest do not align")
    if rd.shape[1:] != (256, 256):
        raise ValueError("Expected 256x256 images")
    meta = json.loads((source / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    if meta.get("unit") != "DN/s" or meta.get("msb_applied") is not False:
        raise ValueError("C0 replay requires No-MSB DN/s inputs")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing replay: {destination}")
    destination.mkdir(parents=True)
    np.save(destination / "north_up_RD_DNs_per_second.npy", rd[indices])
    np.save(destination / "north_up_pair_valid_masks.npy", valid[indices])
    with (destination / "01_rd_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    replay_meta = dict(meta)
    replay_meta.update({
        "requested_date": "20101212",
        "frozen_reference_pairs": True,
        "n_rd_pairs": len(selected),
        "replay_mode": "frozen_60_pair_detection_regression",
        "replay_source_preprocessed": str(source),
        "replay_reference_pair_manifest": str(reference),
        "replay_excluded_count": len(current) - len(selected),
        "note": "Reuses general pipeline RD; does not independently recalibrate/reproject the 60 frames.",
    })
    (destination / "preprocessing_metadata.json").write_text(
        json.dumps(replay_meta, indent=2) + "\n", encoding="utf-8"
    )
    if (source / "target_wcs_header.txt").exists():
        (destination / "target_wcs_header.txt").write_bytes(
            (source / "target_wcs_header.txt").read_bytes()
        )
    summary = {
        "input_pairs": len(current),
        "replay_pairs": len(selected),
        "excluded_pairs": len(current) - len(selected),
        "warmup_dec11": sum(str(r["date_str"]) == "20101211" for r in selected),
        "evaluation_dec12": sum(str(r["date_str"]) == "20101212" for r in selected),
        "selected_source_row_indices_zero_based": indices,
        "excluded_pairs_detail": [
            {"start_file": Path(r["start_file"]).name, "end_file": Path(r["end_file"]).name}
            for i, r in enumerate(current) if i not in set(indices)
        ],
    }
    (destination / "replay_selection.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "selected_source_row_indices_zero_based"}, indent=2))
    if (summary["warmup_dec11"], summary["evaluation_dec12"]) != (28, 32):
        raise ValueError("Replay does not have expected 28 warmup + 32 evaluation pairs")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--source", type=Path, default=Path("runs/dual_archive_20101212/preprocessed"))
    parser.add_argument("--destination", type=Path, default=Path("runs/dual_archive_20101212/frozen_replay/preprocessed"))
    args = parser.parse_args()
    build(args.repo, args.source, args.destination)
