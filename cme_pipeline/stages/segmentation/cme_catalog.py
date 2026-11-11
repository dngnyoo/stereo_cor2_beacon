"""Independent image-plane CME catalog derived only from segmentation outputs."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from cme_pipeline.core.stage import SegmentationOutput
from cme_pipeline.utils.geometry import minimal_circular_enclosing_arc

EVENT_FIELDS = [
    "event_id", "family_id", "member_track_ids",
    "first_observed_time", "last_observed_time", "duration_min",
    "n_observed_frames", "total_mask_pixels", "peak_mask_pixels",
    "peak_pair_index", "cpa_deg", "angular_width_deg", "max_front_rsun",
    "plane_of_sky_speed_km_s", "speed_fit_r2", "speed_fit_frames", "speed_quality",
]
FRAME_FIELDS = [
    "event_id", "family_id", "pair_index", "timestamp",
    "segmented_pixels", "cpa_deg", "angular_width_deg", "front_rsun",
]


def build_cme_catalog(
    segmentation: SegmentationOutput, solar_radius_km: float = 695700.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate event and frame-level measurements without external catalogs.

    CPA and width are the angular extent in the largest segmented frame.
    Plane-of-sky speed is an exploratory linear fit of front height vs time.
    """
    size = segmentation.size_gate_output
    track = size.birth_gate_output.association_output.tracking_output
    manifest = track.manifest_df.reset_index(drop=True)
    geom = track.geometry
    if not np.isfinite(geom.rsun_px) or geom.rsun_px <= 0:
        raise ValueError("Invalid solar radius in pixels")
    pa = np.asarray(geom.pa_raster, dtype=float)
    rad = np.asarray(geom.radius_raster, dtype=float) / geom.rsun_px
    if pa.shape != rad.shape:
        raise ValueError("Inconsistent geometry rasters")
    if not {"pair_index", "end_time"}.issubset(manifest.columns):
        raise ValueError("Manifest requires pair_index and end_time")
    times = pd.to_datetime(manifest["end_time"], utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("Missing timestamps")
    summaries = size.family_summary_df
    summaries = summaries.set_index("family_id") if "family_id" in summaries.columns else None
    event_rows, frame_rows = [], []

    for fid in sorted(size.admitted_family_ids):
        if fid not in segmentation.family_pixel_masks:
            raise ValueError(f"Missing family segmentation: {fid}")
        masks = np.asarray(segmentation.family_pixel_masks[fid], dtype=bool)
        if masks.shape != (len(manifest), *pa.shape):
            raise ValueError(f"Invalid family mask shape for {fid}: {masks.shape}")
        obs = []
        event_id = f"{times.iloc[0].strftime('%Y%m%d')}_{fid}"
        for i, mask in enumerate(masks):
            count = int(mask.sum())
            if not count:
                continue
            arc = minimal_circular_enclosing_arc(pa[mask])
            row = {
                "event_id": event_id, "family_id": fid,
                "pair_index": int(manifest.iloc[i]["pair_index"]),
                "timestamp": times.iloc[i].isoformat(),
                "segmented_pixels": count, "cpa_deg": float(arc["arc_center"]),
                "angular_width_deg": float(arc["arc_span"]),
                "front_rsun": float(np.max(rad[mask])),
            }
            obs.append((i, row))
            frame_rows.append(row)
        if not obs:
            continue
        # Mask-based observations, not first/last tracking candidate timestamps.
        obs.sort(key=lambda item: item[0])
        indices = [i for i, _ in obs]
        first, last = times.iloc[indices[0]], times.iloc[indices[-1]]
        peak = max(obs, key=lambda item: item[1]["segmented_pixels"])[1]
        speed, r2, quality = float("nan"), float("nan"), "INSUFFICIENT_FRAMES"
        if len(obs) >= 3:
            seconds = np.array([(times.iloc[i] - first).total_seconds() for i in indices])
            heights = np.array([row["front_rsun"] * solar_radius_km for _, row in obs])
            if np.unique(seconds).size >= 3:
                slope, intercept = np.polyfit(seconds, heights, 1)
                fitted = slope * seconds + intercept
                total = float(np.sum((heights - heights.mean()) ** 2))
                r2 = (float(1.0 - np.sum((heights - fitted) ** 2) / total)
                      if total > 0 else float("nan"))
                speed = float(slope)
                quality = ("EXPLORATORY" if speed > 0 and np.isfinite(r2) and r2 >= 0.5
                           else "LOW_CONFIDENCE")
            else:
                quality = "DUPLICATE_TIMESTAMPS"
        members = ""
        if summaries is not None and fid in summaries.index and "member_track_ids" in summaries:
            members = str(summaries.loc[fid, "member_track_ids"])
        event_rows.append({
            "event_id": event_id, "family_id": fid, "member_track_ids": members,
            "first_observed_time": first.isoformat(),
            "last_observed_time": last.isoformat(),
            "duration_min": float((last - first).total_seconds() / 60),
            "n_observed_frames": len(obs),
            "total_mask_pixels": sum(row["segmented_pixels"] for _, row in obs),
            "peak_mask_pixels": peak["segmented_pixels"],
            "peak_pair_index": peak["pair_index"],
            "cpa_deg": peak["cpa_deg"], "angular_width_deg": peak["angular_width_deg"],
            "max_front_rsun": max(row["front_rsun"] for _, row in obs),
            "plane_of_sky_speed_km_s": speed, "speed_fit_r2": r2,
            "speed_fit_frames": len(obs), "speed_quality": quality,
        })
    return (pd.DataFrame(event_rows, columns=EVENT_FIELDS),
            pd.DataFrame(frame_rows, columns=FRAME_FIELDS))


def save_cme_catalog(
    segmentation: SegmentationOutput, directory: Path
) -> tuple[Path, Path]:
    """Save standalone CSVs without consulting external validation datasets."""
    events, frames = build_cme_catalog(segmentation)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    events_path = directory / "cme_event_catalog.csv"
    frames_path = directory / "cme_frame_measurements.csv"
    events.to_csv(events_path, index=False)
    frames.to_csv(frames_path, index=False)
    return events_path, frames_path
