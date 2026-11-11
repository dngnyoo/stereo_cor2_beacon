"""Common Stage Interface and typed Data Models for CME Pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Typed Stage Output Data Models
# -----------------------------------------------------------------------------

@dataclass
class BirthDirectionReference:
    """Two-frame causal birth direction locking reference for an active track."""
    track_id: int
    birth_pair_id: int
    birth_time: datetime
    birth_r_min_px: float
    is_anchor: bool
    birth_pa: float                       # Observed PA of solar root / first centroid
    cpa_first: float                      # CPA at first observation
    cpa_second: Optional[float] = None    # CPA at second consecutive observation
    reference_cpa: Optional[float] = None # Circular mean locked reference CPA
    initial_angular_extent: float = 0.0   # Initial angular span (deg)
    status: str = "PROVISIONAL"           # PROVISIONAL, LOCKED, UNCONFIRMED, UNANCHORED
    confirmation_time: Optional[datetime] = None
    confidence: float = 1.0
    flags: List[str] = field(default_factory=list)


@dataclass
class GeometryContext:
    """WCS geometry and precomputed raster coordinates."""
    sx: float
    sy: float
    rsun_px: float
    occulter_inner_radius_px: float
    occulter_r_min_px: float
    fov_r_max_px: float
    pa_raster: np.ndarray          # (256, 256) float
    radius_raster: np.ndarray      # (256, 256) float
    patch_pa: np.ndarray           # (64, 64) float
    patch_rad: np.ndarray          # (64, 64) float
    patch_x: np.ndarray            # (64, 64) float
    patch_y: np.ndarray            # (64, 64) float
    wcs_header_str: str = ""


@dataclass
class DetectionOutput:
    """Output from Frozen Preprocessing / Hierarchical Detection Loader."""
    manifest_df: pd.DataFrame
    patch_activation_grid4: np.ndarray    # (N, 64, 64) bool
    raw_component_labels: np.ndarray      # (N, 64, 64) int32 (true 8-connected components)
    frame_cache: List[Dict[str, Any]]
    geometry: GeometryContext
    dataset_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackingOutput:
    """Unified Output Contract from any Tracking Stage (E1, A3, A4, A5b, future A6+).

    Semantic Distinction:
    - patch_component_labels : (N, 64, 64) int32 — Raw 8-connected spatial component IDs from detection (0 = inactive).
    - track_ownership_masks  : (N, 64, 64) int32 — Mutually exclusive assigned Track ID per active patch (0 = unassigned).
    """
    manifest_df: pd.DataFrame
    track_states_df: pd.DataFrame
    patch_component_labels: np.ndarray    # (N, 64, 64) int32 (spatial components)
    patch_activation_grid4: np.ndarray    # (N, 64, 64) bool
    frame_cache: List[Dict[str, Any]]
    geometry: GeometryContext
    total_tracks: int = 0
    anchor_tracks: List[int] = field(default_factory=list)
    birth_direction_references: Dict[int, BirthDirectionReference] = field(default_factory=dict)
    track_ownership_masks: Optional[np.ndarray] = None  # (N, 64, 64) int32 (Track IDs)
    component_states_df: Optional[pd.DataFrame] = None
    leading_front_states_df: Optional[pd.DataFrame] = None
    tracking_method: Optional[str] = None
    tracking_parameters: Optional[Dict[str, Any]] = None
    dataset_metadata: Optional[Dict[str, Any]] = None

    @property
    def ownership_masks(self) -> Optional[np.ndarray]:
        """Backwards-compatible alias returning track_ownership_masks."""
        if self.track_ownership_masks is not None:
            return self.track_ownership_masks
        return self.__dict__.get("ownership_masks", None)

    @ownership_masks.setter
    def ownership_masks(self, value: Optional[np.ndarray]) -> None:
        self.track_ownership_masks = value


@dataclass
class RecoveryOutput:
    """Output from Signed Dynamic PA Recovery Stage."""
    tracking_output: TrackingOutput
    recovery_func: Optional[Callable[..., np.ndarray]] = None
    target_track_ids: Optional[List[int]] = None
    recovered_patch_masks: Optional[np.ndarray] = None  # (N, 64, 64) bool if computed
    family_recovery_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    total_recovered_patches: int = 0


@dataclass
class AssociationOutput:
    """Output from Family Association Stage."""
    tracking_output: TrackingOutput
    recovery_output: Optional[RecoveryOutput]
    track_to_family: Dict[int, str]
    family_members: Dict[str, List[int]]
    family_anchors: Dict[str, List[int]]
    edges_df: pd.DataFrame
    track_membership_df: pd.DataFrame
    candidate_family_count: int = 0
    family_reference_cpas: Dict[str, float] = field(default_factory=dict)
    birth_direction_references: Dict[int, BirthDirectionReference] = field(default_factory=dict)


@dataclass
class BirthGateOutput:
    """Output from Family-level Birth Radius Gate Stage."""
    association_output: AssociationOutput
    birth_gated_family_ids: List[str]
    rejected_family_reasons: Dict[str, str] = field(default_factory=dict)
    max_birth_radius_px: float = 30.0


@dataclass
class SizeGateOutput:
    """Output from Family-level Size Gate Stage."""
    birth_gate_output: BirthGateOutput
    admitted_family_ids: List[str]
    admitted_track_ids: List[int]
    family_summary_df: pd.DataFrame
    size_gate_decisions_df: pd.DataFrame
    recovered_patch_masks: np.ndarray            # (N, 64, 64) bool (sum = 5,174 on baseline)
    family_recovered_patch_masks: Dict[str, np.ndarray]
    min_detected_patches: int = 5
    total_recovered_patches: int = 0


@dataclass
class SegmentationOutput:
    """Output from Island-First Pixel Segmentation Stage."""
    size_gate_output: SizeGateOutput
    final_pixel_segmentation_union: np.ndarray   # (N, 256, 256) bool (sum = 50,665 on baseline)
    family_pixel_masks: Dict[str, np.ndarray]    # {family_id: (N, 256, 256) bool}
    framewise_pixel_summary_df: pd.DataFrame
    family_pixel_statistics_df: pd.DataFrame
    total_segmented_pixels: int = 0


@dataclass
class CMEProductsOutput:
    """Catalog and diagnostic visualization products, preserving segmentation."""
    segmentation_output: SegmentationOutput
    event_catalog_df: pd.DataFrame
    frame_measurements_df: pd.DataFrame
    artifact_files: List[str] = field(default_factory=list)


@dataclass
class ValidationOutput:
    """Output from External Catalogue Validation Stage."""
    segmentation_output: SegmentationOutput
    comparison_metrics_df: pd.DataFrame
    frame_level_scores_df: pd.DataFrame
    event_level_scores_df: pd.DataFrame
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Base PipelineStage Abstract Class
# -----------------------------------------------------------------------------

class PipelineStage(ABC):
    """Abstract Base Class for all CME Pipeline stages."""
    name: str = "base_stage"
    version: str = "1.0.0"

    @abstractmethod
    def run(self, inputs: Any, config: Any, context: Any) -> Any:
        """Executes the stage computation.
        
        Args:
            inputs: Typed output dataclass from upstream stage (or None for initial stage).
            config: Stage-specific or pipeline configuration.
            context: PipelineContext providing run environment, cache, and logger.
            
        Returns:
            Typed output dataclass specific to this stage.
        """
        raise NotImplementedError

    def compute_fingerprint(self, inputs: Any, config: Any, context: Any) -> str:
        """Computes a deterministic hash representing inputs, config, and version."""
        from cme_pipeline.utils.io import compute_dict_hash
        stage_desc = {
            "stage_name": self.name,
            "version": self.version,
            "config": config if isinstance(config, dict) else (config.to_dict() if hasattr(config, "to_dict") else str(config)),
        }
        return compute_dict_hash(stage_desc)

