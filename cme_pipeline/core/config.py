"""Configuration models for CME Pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cme_pipeline.utils.io import load_yaml, save_yaml


@dataclass
class TrackingConfig:
    method: str = "e1"
    alpha: float = 0.00
    patch_size: int = 4
    min_track_duration_pairs: int = 1
    max_cadence_gap_sec: float = 4500.0
    direction_threshold_deg: float = 30.0
    use_birth_direction: bool = False
    lock_consecutive_frames: int = 2
    max_direction_deviation_deg: float = 45.0


@dataclass
class RecoveryConfig:
    method: str = "signed_dynamic_pa"
    radial_window_px: float = 13.35
    pa_margin_deg: float = 5.0
    gaussian_sigma: float = 1.0


@dataclass
class AssociationConfig:
    method: str = "pure_do"
    do_threshold: float = 0.05
    prevent_auto_merger: bool = True
    direction_threshold_deg: float = 45.0
    use_birth_direction: bool = False
    enforce_anchor_separation: bool = True


@dataclass
class BirthGateConfig:
    max_birth_radius_px: float = 30.0


@dataclass
class SizeGateConfig:
    min_detected_patches: int = 5


@dataclass
class SegmentationConfig:
    method: str = "island_first"
    min_island_area_px: int = 16
    max_hole_fill_px: int = 16
    occulter_r_min_px: float = 22.0
    fov_r_max_px: float = 122.5


@dataclass
class ValidationConfig:
    catalogs: List[str] = field(
        default_factory=lambda: ["SEEDS", "CACTus", "MVCC", "Göttingen"]
    )
    compare_baseline: bool = True


@dataclass
class PipelineConfig:
    run_name: str = "baseline_rule_b"
    date_str: int = 20101212
    repo_root: str = "/home/aababc/stereo_cor2"
    output_base_dir: str = "runs"
    cache_enabled: bool = True
    force_rerun: bool = False
    from_stage: Optional[str] = None
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    birth_gate: BirthGateConfig = field(default_factory=BirthGateConfig)
    size_gate: SizeGateConfig = field(default_factory=SizeGateConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PipelineConfig:
        """Instantiates PipelineConfig from nested dictionary."""
        cfg = cls()
        for k, v in d.items():
            if k == "tracking" and isinstance(v, dict):
                cfg.tracking = TrackingConfig(**v)
            elif k == "recovery" and isinstance(v, dict):
                cfg.recovery = RecoveryConfig(**v)
            elif k == "association" and isinstance(v, dict):
                cfg.association = AssociationConfig(**v)
            elif k == "birth_gate" and isinstance(v, dict):
                cfg.birth_gate = BirthGateConfig(**v)
            elif k == "size_gate" and isinstance(v, dict):
                cfg.size_gate = SizeGateConfig(**v)
            elif k == "segmentation" and isinstance(v, dict):
                cfg.segmentation = SegmentationConfig(**v)
            elif k == "validation" and isinstance(v, dict):
                cfg.validation = ValidationConfig(**v)
            elif hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> PipelineConfig:
        """Loads PipelineConfig from a YAML file."""
        d = load_yaml(path)
        return cls.from_dict(d)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes PipelineConfig to a dictionary."""
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Saves PipelineConfig to a YAML file."""
        save_yaml(self.to_dict(), path)

