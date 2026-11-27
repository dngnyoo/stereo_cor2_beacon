import csv
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from scripts import batch_external_validation as batch


def make_plan(tmp_path, rows=None):
    parent = tmp_path / "stage08"
    parent.mkdir()
    (parent / "config_snapshot.json").write_text(json.dumps(
        {"experiment": {"date": "20101212"}, "catalog": {"method": "event_kinematics"}}))
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump({"catalogs": ["SEEDS"], "runs": rows or [
        {"algorithm": a, "date": "20101212", "input_run": str(parent)}
        for a in ("baseline", "seed")]}))
    return path


def test_plan_and_duplicate(tmp_path):
    plan = make_plan(tmp_path)
    settings, _, jobs = batch.load_plan(plan)
    assert len(jobs) == 2
    assert settings["max_time_delta_sec"] == 900
    doc = yaml.safe_load(plan.read_text())
    doc["runs"].append(doc["runs"][0])
    plan.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="Duplicate"):
        batch.load_plan(plan)


def test_pinned_snapshot_and_mutation(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("header\nvalue\n")
    settings = {"catalogs": ["SEEDS"]}
    with patch.object(batch, "prepare", return_value=(
            {"seeds_frames_csv": str(source)}, {"status": "REUSED_NORMALIZED"})):
        frozen = batch.pin_references("20101212", settings, str(tmp_path), tmp_path)
    item = frozen["SEEDS"]["files"]["seeds_frames_csv"]
    source.write_text("changed")
    batch.verify_pins(frozen)
    Path(item["path"]).write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        batch.verify_pins(frozen)


def test_shared_unavailable_reference(tmp_path):
    plan = make_plan(tmp_path)
    output = tmp_path / "out"
    frozen = {"SEEDS": {"status": "REFERENCE_UNAVAILABLE", "reason": "HTTP 404"}}
    with patch.object(batch, "resolve_catalog_parent", return_value=(pd.DataFrame({"pair_index": [1]}),)), \
         patch.object(batch, "pin_references", return_value=frozen) as pin, \
         patch.object(batch, "execute_external") as execute:
        assert batch.run_batch(plan, output)
    assert pin.call_count == 1
    execute.assert_not_called()
    with (output / "comparison_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(r["status"] == "REFERENCE_UNAVAILABLE" and not r["mean_matched_angular_iou"] for r in rows)
    with pytest.raises(FileExistsError):
        batch.run_batch(plan, output)


def test_parent_failure_not_zero(tmp_path):
    plan = make_plan(tmp_path)
    output = tmp_path / "out"
    with patch.object(batch, "resolve_catalog_parent", side_effect=ValueError("bad artifact")), \
         patch.object(batch, "pin_references") as pin:
        assert not batch.run_batch(plan, output)
    pin.assert_not_called()
    with (output / "comparison_summary.csv").open() as f:
        assert all(r["status"] == "INPUT_FAILED" and not r["matched_events"] for r in csv.DictReader(f))


def test_reuses_stage09_and_verifies_reference(tmp_path):
    plan = make_plan(tmp_path)
    output = tmp_path / "out"
    source = tmp_path / "seeds.csv"
    source.write_text("header\n")
    digest = batch._hash_file(source)
    frozen = {"SEEDS": {"status": "PINNED", "files": {
        "seeds_frames_csv": {"path": str(source), "sha256": digest}}}}
    def execute(config, *, input_run, until, run_dir, visualize):
        assert config.data["validation"]["seeds_frames_csv"] == str(source)
        assert until == "validation" and visualize == "off"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text(json.dumps({
            "date": "20101212", "input_catalog_run": str(input_run),
            "upstream_recomputed": False, "catalogs": {"SEEDS": {
                "status": "COMPLETED", "matched_events": 1,
                "mean_matched_angular_iou": .5, "metric": "mean_frame_angular_iou",
                "references": {"seeds_frames_csv": {"sha256": digest}}}}}))
        (run_dir / "matches.csv").write_text(",".join(batch.MATCH_COLUMNS) + "\nSEEDS,r,e,0.5,mean_frame_angular_iou\n")
    meta = {"outputs": {"evaluation_metrics": {"path": "summary.json"},
                        "catalog_matches": {"path": "matches.csv"}}}
    with patch.object(batch, "resolve_catalog_parent", return_value=(pd.DataFrame({"pair_index": [1]}),)), \
         patch.object(batch, "pin_references", return_value=frozen) as pin, \
         patch.object(batch, "execute_external", side_effect=execute) as call, \
         patch.object(batch, "verify_stage", return_value=meta):
        assert batch.run_batch(plan, output)
    assert pin.call_count == 1 and call.call_count == 2
    with (output / "matched_events.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 2
