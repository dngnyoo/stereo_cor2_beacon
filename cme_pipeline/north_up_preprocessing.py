"""Strict d7 No-MSB calibration -> North-up reprojection -> 30-minute RD.

Scientific source lineage:
- segmentation_pipeline/16_no_MSB_preprocessing.py (trim, exposure, SEBIP, bias)
- handover_archive/preprocessing/reproject.py (WCS-aligned reprojection)
- segmentation_pipeline/17_no_MSB_RD.py (paired difference, no gap bridge)

Outputs are DN/s, never raw DNs or MSB. Detector integration follows separately.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def _legacy_functions():
    """Load the retained SECCHI trim/SEBIP/bias implementations."""
    path = Path(__file__).resolve().parents[1] / "preprocessing" / "functions.py"
    if not path.is_file():
        raise FileNotFoundError(f"SECCHI calibration functions missing: {path}")
    spec = importlib.util.spec_from_file_location("_cor2_legacy_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_sebip(raw, header):
    """Reproduce Style-2's cor2_sebip, without its get_calfac(MSB) call."""
    text = str(header.get("IP_00_19", ""))
    data = raw.copy()
    if not text.strip():
        return data
    text = text.rjust(60)
    ops = [text[i:i + 3] for i in range(0, 60, 3)]
    if "117" in ops:
        ops = ops[ops.index("117"):]
    data *= 2.0 ** ops.count("  1")
    data *= 64.0 ** (ops.count(" 16") + ops.count(" 17"))
    if float(header.get("IPSUM", 0)) > 0:
        data *= 4.0 ** ops.count(" 53")
    return data


def calibrate_no_msb(path: Path, *, trim_function=None, bias_function=None):
    """Return No-MSB DN/s image and matching WCS header.

    Fail closed if trim or bias cannot be applied, rather than silently
    assuming that raw data are already physically corrected.
    """
    if trim_function is None or bias_function is None:
        legacy = _legacy_functions()
        trim_function = trim_function or legacy.scc_img_trim
        bias_function = bias_function or legacy.get_biasmean
    with fits.open(path, memmap=False) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float32).copy()
        header = hdul[0].header.copy()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D FITS image: {path}")
    image, header = trim_function(image, header, silent=True)
    exposure = float(header.get("EXPTIME", 0))
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError(f"Invalid EXPTIME: {path}")
    image = _legacy_sebip(image / exposure, header)
    bias = float(bias_function(header, silent=True))
    image -= bias / exposure
    header["BUNIT"] = "DN/s"
    header["EXPNORM"] = True
    header["SEBIPAPP"] = True
    header["MSBAPP"] = False
    header["CMADBKG"] = False
    header["BIASDN"] = bias
    header.add_history("No-MSB: trim, EXPTIME, SEBIP, bias; no MSB multiplication")
    WCS(header).celestial  # validate WCS metadata before reprojection
    return image, header


def target_north_up_wcs(headers, *, shape=(256, 256)):
    """Set a fixed solar-centered, north-up angular grid from input WCSes.

    Uses input celestial scale and input array extent to retain the physical FOV
    when original FITS resolution differs from 256 x 256. All frames share
    exactly one WCS target.
    """
    from astropy.wcs.utils import proj_plane_pixel_scales

    scales = []
    centers = []
    ctype = None
    cunit = None
    for header in headers:
        w = WCS(header).celestial
        if w.pixel_n_dim != 2 or w.world_n_dim != 2:
            raise ValueError("Image requires 2D solar WCS")
        raw_shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
        angle_scale = np.asarray(proj_plane_pixel_scales(w), dtype=float)
        if not np.all(np.isfinite(angle_scale)) or np.any(angle_scale <= 0):
            raise ValueError("Invalid WCS angular pixel scale")
        scales.append(angle_scale * np.array([raw_shape[1] / shape[1], raw_shape[0] / shape[0]]))
        x0, y0 = w.world_to_pixel_values(0.0, 0.0)
        centers.append([x0 / raw_shape[1] * shape[1], y0 / raw_shape[0] * shape[0]])
        if ctype is None:
            ctype, cunit = w.wcs.ctype, w.wcs.cunit
    if not scales:
        raise ValueError("No source WCS headers")
    s = np.median(scales, axis=0)
    c = np.median(centers, axis=0)
    target = WCS(naxis=2)
    target.wcs.ctype = list(ctype)
    target.wcs.cunit = list(cunit)
    target.wcs.crval = [0.0, 0.0]
    target.wcs.crpix = [float(c[0]) + 1, float(c[1]) + 1]
    target.wcs.cdelt = [-float(s[0]), float(s[1])]
    target.wcs.pc = np.eye(2)
    target.wcs.set()
    return target


def run_no_msb_north_up(manifest_path: str | Path, *,
                         mask_path: str | Path | None = None,
                         shape=(256, 256)) -> Path:
    """Build science RD/mask arrays for previously selected strict d7 pairs."""
    try:
        from reproject import reproject_interp
    except ImportError as exc:
        raise RuntimeError("North-up preprocessing requires: pip install reproject") from exc

    manifest_path = Path(manifest_path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("product") != "d7" or not data.get("cadence_selection"):
        raise ValueError("No-MSB North-up requires a d7 cadence-selection manifest")
    selection = data["cadence_selection"]
    frames = selection["selected_frames"]
    pairs = selection["pairs"]
    if not pairs:
        raise ValueError("At least one valid 30-minute d7 pair is required")
    root = manifest_path.parent
    output_dir = root / "preprocessed"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing preprocessing: {output_dir}")

    # Calibrate exactly once per selected image. Header/WCS are checked before
    # any output writes, so broken observations cannot become empty detections.
    calibrated = {}
    for frame in frames:
        source = str(frame["file"])
        calibrated[source] = calibrate_no_msb(Path(source))
    target = target_north_up_wcs([h for _, h in calibrated.values()], shape=shape)
    if mask_path is None:
        mask_path = Path(__file__).resolve().parents[1] / "Calibration" / "cor2_mask.fts"
    mask_path = Path(mask_path).resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"Required COR2 occulter mask: {mask_path}")
    with fits.open(mask_path, memmap=False) as hdul:
        mask_data = np.asarray(hdul[0].data, dtype=np.float32)
        mask_wcs = WCS(hdul[0].header).celestial
    mask_interp, mask_fp = reproject_interp(
        (mask_data, mask_wcs), target, shape_out=shape, order="nearest-neighbor"
    )
    static_valid = (mask_fp > .95) & np.isfinite(mask_interp) & (mask_interp >= .95)
    if not np.any(static_valid):
        raise ValueError("Occulter mask has no valid pixels after reprojection")

    north_up = {}
    for source, (image, hdr) in calibrated.items():
        reproj, footprint = reproject_interp(
            (image, WCS(hdr).celestial), target,
            shape_out=shape, order="bilinear"
        )
        north_up[source] = (reproj, footprint)
    rows = []
    rds, valids = [], []
    for pair in pairs:
        first, second = pair["start_file"], pair["end_file"]
        before, fp_before = north_up[first]
        after, fp_after = north_up[second]
        valid = (static_valid & (fp_before > .95) & (fp_after > .95)
                 & np.isfinite(before) & np.isfinite(after))
        if not np.any(valid):
            raise ValueError(f"No valid RD pixels for pair {pair['pair_index']}")
        if abs(float(pair["delta_t_sec"]) - 1800.) > 2:
            raise ValueError("Non-30-minute pair in selected manifest")
        rd = np.zeros(shape, dtype=np.float32)
        rd[valid] = (after[valid] - before[valid]).astype(np.float32)
        rds.append(rd)
        valids.append(valid)
        rows.append({
            "pair_index": int(pair["pair_index"]),
            "date_str": int(data["date"]),
            "rd_pair_id": f"{data['date']}_{int(pair['pair_index']):05d}",
            "start_time": pair["start_time"],
            "end_time": pair["end_time"],
            "delta_t_min": pair["delta_t_min"],
            "start_file": first,
            "end_file": second,
            "unit": "DN/s",
        })
    output_dir.mkdir(parents=False, exist_ok=False)
    np.save(output_dir / "north_up_RD_DNs_per_second.npy", np.stack(rds))
    np.save(output_dir / "north_up_pair_valid_masks.npy", np.stack(valids))
    (output_dir / "target_wcs_header.txt").write_text(
        target.to_header().tostring(sep="\n"), encoding="utf-8"
    )
    with (output_dir / "01_rd_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "preprocessing_metadata.json").write_text(json.dumps({
        "source_manifest": str(manifest_path),
        "n_selected_frames": len(frames),
        "n_rd_pairs": len(rows),
        "shape": list(shape),
        "unit": "DN/s",
        "msb_applied": False,
        "north_up": True,
        "cadence_tolerance_sec": 2.,
        "mask": str(mask_path),
        "status": "RD_READY_DETECTION_NOT_RUN",
    }, indent=2), encoding="utf-8")
    return output_dir
