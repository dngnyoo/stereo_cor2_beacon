"""Smoke tests for relocated reference and background resource paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_fixture_paths():
    fixture = ROOT / "validation/frozen/20101212/experiments"
    assert (fixture / "01_d7_baseline_segmentation/results/00_manifest/01_rd_manifest.csv").is_file()
    assert (fixture / "11_d7_patch_first_tracking_feasibility/results/tables/03_patch_e1_track_states.csv").is_file()


def test_background_structure():
    root = ROOT / "Calibration/backgrounds/monthly_min/a"
    assert (root / "201207").is_dir()
    assert (root / "202405").is_dir()
    # These are historical placeholders in Git, not validated FITS calibration data.
