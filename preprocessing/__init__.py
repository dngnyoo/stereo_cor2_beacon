"""COR2 Preprocessing Package for Handover.

Provides:
- Raw FITS physical calibration (exposure normalization, bias, onboard SEBIP decompression)
- Static occulter mask loading from Calibration resources
- WCS North-Up solar-centered reprojection (256x256 grid)
- 30-minute running difference (RD) generation
- Causal 24-hour rolling baseline and dynamic Z-score calculation
- Multi-scale hierarchical patch activity (16x16 -> 8x8 -> 4x4)
"""

from .calibration import get_biasmean, cor2_sebip, load_occulter_mask
from .reproject import derive_target_wcs, reproject_to_target, calculate_geometry
from .detector import causal_rolling_baseline, compute_patch_activities

__all__ = [
    "get_biasmean",
    "cor2_sebip",
    "load_occulter_mask",
    "derive_target_wcs",
    "reproject_to_target",
    "calculate_geometry",
    "causal_rolling_baseline",
    "compute_patch_activities",
]

