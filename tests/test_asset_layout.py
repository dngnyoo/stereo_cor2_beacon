"""Tests for reorganized calibration and frozen reference locations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_assets_present():
    assert (ROOT / "validation/frozen/20101212/experiments").is_dir()
    assert (ROOT / "validation/frozen/20101212/catalogs").is_dir()


def test_calibration_assets_present():
    assert (ROOT / "Calibration/cor2A_mask.fts").is_file()
    assert (ROOT / "Calibration/backgrounds/monthly_min/a").is_dir()


def test_background_directory_configuration():
    code = (ROOT / "preprocessing/common_utils.py").read_text()
    assert 'PROJECT_ROOT / "Calibration" / "backgrounds"' in code
    assert '"bkg_base": BKG_BASE_DIR' in code
    assert '"cal_dir": CAL_DIR' in code
