"""Single-command full standard Run, preserving immutable internal stage Runs.

The orchestration is deliberately sequential: every completed child is the
explicit input of the next and is verified through its saved stage manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from cme_pipeline.core.standard_stages import STAGES
from cme_pipeline.core.run_layout import verify_stage
from cme_pipeline.standard_runner import execute_to_gates, _revision


@dataclass(frozen=True)
class Step:
    stage: str
    parent_stage: str | None


STEPS = (
    Step("detection", None),
    Step("tracking", "detection"),
    Step("association", "tracking"),
    Step("gates", "association"),
    Step("recovery", "gates"),
    Step("segmentation", "recovery"),
    Step("catalog", "segmentation"),
    Step("validation", "catalog"),
)


def _apply_verified_geometry(config, first_run: Path):
    """Use actual evaluation-date FITS RSUN and angular pixel scale."""
    settings = config.data.setdefault("catalog", {})
    if settings.get("rsun_px") is not None:
        return
    from astropy.io import fits
    from astropy import units as u
    manifest_path = first_run / "01_preprocessing/data/acquisition/input_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "Automatic catalog.rsun_px requires Stage 01 Raw FITS processing; "
            "set catalog.rsun_px explicitly for an imported RD Run"
        )
    manifest = json.loads(manifest_path.read_text())
    date = str(config.data["experiment"]["date"])
    frames = manifest["cadence_selection"]["selected_frames"]
    dated = [item for item in frames if str(item["date_obs"]).startswith(
        f"{date[:4]}-{date[4:6]}-{date[6:]}")]
    if not dated:
        raise ValueError("No evaluation-date raw FITS available for RSUN calibration")
    header = fits.getheader(dated[0]["file"])
    if "RSUN" not in header:
        raise ValueError("Raw FITS lacks source-native RSUN; set verified catalog.rsun_px")
    unit = str(header.get("CUNIT1", "")).strip().lower()
    if unit not in ("arcsec", "deg"):
        raise ValueError(f"Unsupported source pixel angular unit: {unit!r}")
    scale = abs(float(header["CDELT1"])) * (3600 if unit == "deg" else 1)
    radius = float(header["RSUN"]) / scale
    import math
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid source-native solar radius / pixel scale")
    settings["rsun_px"] = radius
    settings.setdefault("solar_radius_km", 695700.0)


def _ensure_seeds(config):
    """Prepare official reference if absent; no frozen references or fake data."""
    sec = config.data.setdefault("validation", {})
    if sec.get("method", "seeds") != "seeds":
        raise ValueError("Full run currently supports SEEDS validation only")
    sec["method"] = "seeds"
    date = str(config.data["experiment"]["date"])
    target = Path(sec.get("seeds_frames_csv") or
                  f"data/catalogs/seeds/{date}/seeds_frames.csv").expanduser().resolve()
    sec["seeds_frames_csv"] = str(target)
    if target.is_file():
        return
    from scripts.fetch_seeds_monthly import acquire
    from scripts.fetch_seeds_details import fetch_details
    from scripts.parse_seeds_monthly import parse_monthly
    from scripts.parse_seeds_details import parse_directory
    from datetime import datetime
    dt = datetime.strptime(date, "%Y%m%d")
    raw_root = Path("data/catalogs/seeds/raw")
    detail_root = Path("data/catalogs/seeds/details")
    month = raw_root / f"{dt:%Y}" / f"{dt:%m}" / "monthly.txt"
    # A previously interrupted monthly fetch may have downloaded valid
    # source bytes before the parser failed. Never overwrite those bytes.
    if not month.is_file():
        month_dir = month.parent
        if month_dir.exists() and any(month_dir.iterdir()):
            raise FileNotFoundError(
                f"SEEDS monthly source incomplete at {month_dir}; inspect before refetch"
            )
        acquire(date, output_root=raw_root)
    events_expected = parse_monthly(month.read_text(encoding="utf-8-sig"), date=date)
    details = detail_root / date
    # fetch_details verifies and reuses each cached event file, and fetches
    # only missing official source event details.
    fetch_details(date, raw_root=raw_root, output_root=detail_root)
    events, frames = parse_directory(details, date)
    if frames.empty or events.empty or len(events) != len(events_expected):
        raise ValueError(f"Incomplete official SEEDS observations for {date}")
    if set(events["event_id"]) != set(events_expected["event_id"]):
        raise ValueError("Parsed SEEDS details do not match source monthly event IDs")
    target.parent.mkdir(parents=True, exist_ok=True)
    events_path = target.with_name("seeds_events.csv")
    if events_path.exists():
        raise FileExistsError(
            f"SEEDS events output already exists without frame reference: {events_path}"
        )
    frames.to_csv(target, index=False)
    events.to_csv(events_path, index=False)


def run_full_standard(config, *, until: str, run_dir, visualize=None):
    """Create exactly one top-level Run with immutable child-stage lineage."""
    from cme_pipeline.standard_input_run import execute_from_detection
    from cme_pipeline.standard_tracking_parent import execute_from_tracking
    from cme_pipeline.standard_association_parent import execute_from_association
    from cme_pipeline.standard_recovery_parent import execute_from_gates
    from cme_pipeline.standard_segmentation_parent import execute_from_recovery
    from cme_pipeline.standard_catalog_parent import execute_from_segmentation
    from cme_pipeline.standard_seeds_validation_parent import execute_from_catalog
    from cme_pipeline.standard_full_visualization import render_full_stage

    if until not in {stage.name.value for stage in STAGES}:
        raise ValueError(f"Unknown stage {until}")
    mode = config.vis_mode("preprocessing", visualize)
    # Stage-specific overrides are read for each stage; "all" means all
    # evaluation-date frames where frame-oriented graphics are applicable.
    target = Path(run_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Top-level Run already exists: {target}")
    if until in ("preprocessing", "detection"):
        raise ValueError("Full orchestration is for 03-09; use normal Stage 01-02 runner")
    if "preprocessed_dir" in (config.data.get("input") or {}) and (
            config.data.get("catalog") or {}).get("rsun_px") is None:
        raise ValueError("Imported RD requires explicit verified catalog.rsun_px")
    target.mkdir(parents=True)
    children = target / "stage_runs"
    children.mkdir()
    summary = {
        "date": str(config.data["experiment"]["date"]),
        "source_revision": _revision(),
        "target_stage": until,
        "status": "RUNNING",
        "run_layout": "single_top_level_run_with_immutable_explicit_parent_stage_runs",
        "steps": [],
    }

    def save():
        (target / "full_run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n")

    save()
    try:
        first = children / "01_02_preprocessing_detection"
        execute_to_gates(config, until="detection", run_dir=first, visualize="off")
        verify_stage(first, "preprocessing")
        verify_stage(first, "detection")
        for name in ("preprocessing", "detection"):
            row = {"stage": name, "run": str(first), "status": "COMPLETED"}
            render_full_stage(config, name, first, target, visualize)
            summary["steps"].append(row)
            save()
        if until == "detection":
            summary["status"] = "COMPLETED"
            save()
            return target
        _apply_verified_geometry(config, first)
        if until == "validation":
            _ensure_seeds(config)
        runners = {
            "tracking": execute_from_detection,
            "association": execute_from_tracking,
            "gates": execute_from_association,
            "recovery": execute_from_gates,
            "segmentation": execute_from_recovery,
            "catalog": execute_from_segmentation,
            "validation": execute_from_catalog,
        }
        parent = first
        for step in STEPS[1:]:
            stage = step.stage
            ordinal = next(x.ordinal for x in STAGES if x.name.value == stage)
            child = children / f"{ordinal:02d}_{stage}"
            runners[stage](config, input_run=parent, until=stage, run_dir=child,
                           visualize="off")
            verify_stage(child, stage)
            render_full_stage(config, stage, child, target, visualize)
            summary["steps"].append({
                "stage": stage, "run": str(child),
                "parent_run": str(parent), "status": "COMPLETED"
            })
            save()
            parent = child
            if stage == until:
                break
        summary["status"] = "COMPLETED"
        save()
        return target
    except Exception as exc:
        summary["status"] = "FAILED"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
