"""Stable nine-stage names, order and I/O contracts for the COR2 refactor.

This module is intentionally independent of historical stage implementations.
The adapter/runner must validate the contracts before marking a stage complete.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional


class StageName(str, Enum):
    PREPROCESSING = "preprocessing"
    DETECTION = "detection"
    TRACKING = "tracking"
    ASSOCIATION = "association"
    GATES = "gates"
    RECOVERY = "recovery"
    SEGMENTATION = "segmentation"
    CATALOG = "catalog"
    VALIDATION = "validation"


@dataclass(frozen=True)
class StageSpec:
    name: StageName
    ordinal: int
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    substeps: tuple[str, ...]

    @property
    def folder(self) -> str:
        return f"{self.ordinal:02d}_{self.name.value}"


STAGES: tuple[StageSpec, ...] = (
    StageSpec(StageName.PREPROCESSING, 1, ("fits_sources",),
              ("rd", "valid_mask", "manifest", "geometry", "source_provenance"),
              ("acquisition", "frame_cadence_selection", "image_preprocessing")),
    StageSpec(StageName.DETECTION, 2, ("rd", "valid_mask", "manifest", "geometry"),
              ("patch_activation", "component_labels", "frame_qc", "detection_scores"),
              ("hierarchical_detection", "thresholding", "frame_qc")),
    StageSpec(StageName.TRACKING, 3, ("patch_activation", "component_labels", "frame_qc", "geometry"),
              ("track_states", "track_ownership", "tracking_provenance"),
              ("pre_size_gate", "a5b_tracking", "strict_continuity")),
    StageSpec(StageName.ASSOCIATION, 4, ("track_states", "track_ownership"),
              ("family_membership", "association_decisions"), ("identity_association",)),
    StageSpec(StageName.GATES, 5, ("family_membership", "track_states", "track_ownership", "geometry"),
              ("gate_decisions", "admitted_family_ids", "qualified_pair_ranges"),
              ("birth_gate", "size_qualified_persistence")),
    StageSpec(StageName.RECOVERY, 6, ("admitted_family_ids", "track_ownership", "rd"),
              ("recovered_patch_masks", "recovery_metrics"), ("recovery",)),
    StageSpec(StageName.SEGMENTATION, 7, ("recovered_patch_masks", "rd", "valid_mask"),
              ("pixel_masks", "segmentation_metrics"), ("segmentation",)),
    StageSpec(StageName.CATALOG, 8, ("pixel_masks", "manifest", "geometry"),
              ("events", "height_time", "kinematics"), ("catalog_generation", "kinematics")),
    StageSpec(StageName.VALIDATION, 9, ("events", "kinematics", "external_catalogs"),
              ("catalog_matches", "evaluation_metrics"), ("external_validation",)),
)

STAGE_BY_NAME: Mapping[str, StageSpec] = {s.name.value: s for s in STAGES}
VIS_MODES = frozenset({"off", "diagnostic", "all"})


def stages_until(until: str) -> tuple[StageSpec, ...]:
    if until not in STAGE_BY_NAME:
        raise ValueError(f"Unknown stage {until!r}; choose {', '.join(STAGE_BY_NAME)}")
    return STAGES[:STAGE_BY_NAME[until].ordinal]


def resolve_visualization_mode(
    stage_name: str, *, config: Mapping, cli_override: Optional[str] = None
) -> str:
    """CLI > stage config > shared config > off. Never alter science hashes."""
    if stage_name not in STAGE_BY_NAME:
        raise ValueError(f"Unknown stage: {stage_name}")
    shared = config.get("visualization", {}) or {}
    stage_cfg = (config.get(stage_name, {}) or {}).get("visualization", {}) or {}
    mode = (cli_override if cli_override is not None else
            stage_cfg.get("mode", shared.get("default", "off")))
    if mode not in VIS_MODES:
        raise ValueError(f"Invalid visualization mode {mode!r}")
    return str(mode)


def stage_data_path(run_dir: str | Path, stage_name: str) -> Path:
    """Within-run output location; no cross-run auto-discovery."""
    if stage_name not in STAGE_BY_NAME:
        raise ValueError(f"Unknown stage: {stage_name}")
    return Path(run_dir) / STAGE_BY_NAME[stage_name].folder / "data"
