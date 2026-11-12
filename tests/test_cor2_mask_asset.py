"""Official COR2-A asset contract: fail before image calibration if missing."""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from cme_pipeline.north_up_preprocessing import (
    COR2A_MASK_SHA256, load_cor2_mask,
)


def test_official_cor2a_checksum_contract():
    assert COR2A_MASK_SHA256 == (
        "7cde9474fa9d1fe93d9a6ff08464598d5d2250b423b00be449ca24b2ec7ca606"
    )


def test_empty_mask_rejected(tmp_path):
    path = tmp_path / "cor2A_mask.fts"
    path.touch()
    with pytest.raises(FileNotFoundError, match="Missing or empty"):
        load_cor2_mask(path)


def test_invalid_explicit_shape_rejected(tmp_path):
    path = tmp_path / "experiment_mask.fts"
    fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32)).writeto(path)
    with pytest.raises(ValueError, match="Invalid COR2 mask"):
        load_cor2_mask(path)


def test_invalid_fits_rejected(tmp_path):
    path = tmp_path / "experiment_mask.fts"
    path.write_text("not a FITS file")
    with pytest.raises(ValueError, match="Unreadable COR2 mask"):
        load_cor2_mask(path)
