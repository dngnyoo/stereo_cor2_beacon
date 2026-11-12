"""Safe CLI and orchestration contract tests (no FITS/network required)."""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from cme_pipeline.__main__ import parse_args
from cme_pipeline.live_full_pipeline import run_live_full


def _args(*tail):
    with patch.object(sys, "argv", ["python -m cme_pipeline", "run-data", *tail]):
        return parse_args()


def test_run_data_parses_existing_detection():
    args = _args("--config", "configs/baseline_rule_b.yaml",
                 "--date", "20101212", "--run-name", "test_run",
                 "--detection-dir", "/tmp/detect", "--frozen-regression")
    assert args.command == "run-data"
    assert args.detection_dir == "/tmp/detect"
    assert args.frozen_regression is True


def test_run_data_accepts_full_raw_local_path():
    args = _args("--config", "configs/baseline_rule_b.yaml",
                 "--date", "20240512", "--run-name", "test_run",
                 "--source", "local", "--input-dir", "/tmp/fits",
                 "--phase-sec", "1434")
    assert args.source == "local"
    assert args.phase_sec == 1434.0
    assert not args.frozen_regression


def test_run_data_requires_one_input_source():
    with pytest.raises(SystemExit):
        _args("--config", "configs/baseline_rule_b.yaml",
              "--date", "20101212", "--run-name", "test_run")


def test_frozen_selection_rejected_for_other_dates():
    with pytest.raises(ValueError, match="only defined for 20101212"):
        run_live_full(config_path="not-read.yaml", date="20240512",
                      run_name="test_run", detection_dir="/tmp/detect",
                      frozen_regression=True)


def test_multiple_data_input_routes_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        run_live_full(config_path="not-read.yaml", date="20101212",
                      run_name="test_run", detection_dir="/tmp/detect",
                      preprocessed_dir="/tmp/pre")


def test_historical_validation_only_runs_in_explicit_frozen_mode():
    from types import SimpleNamespace
    from cme_pipeline.live_full_pipeline import select_live_stages

    stages = [SimpleNamespace(name=name) for name in (
        "tracking", "recovery", "association", "birth_gate",
        "size_gate", "segmentation", "products", "validation"
    )]
    for date in ("20101212", "20240512"):
        # Date does not grant permission to index a 32-frame fixed reference.
        selected = select_live_stages(stages, frozen_regression=False)
        assert [stage.name for stage in selected][-1] == "products"
        assert len(selected) == 7
    assert [s.name for s in select_live_stages(stages, frozen_regression=True)][-1] == "validation"
