"""Tracking Algorithm Registry & Strict Parameter Validation.

Provides a unified registry mapping method names and aliases (`e1`, `a3`, `a4`,
`a5b`, plus legacy pilot methods) to their corresponding `PipelineStage` classes
and supported parameter schemas.

Future algorithms (A6, A7, etc.) can register themselves via `register_tracking_method`
without modifying pipeline orchestration or downstream compatibility code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Tuple, Type

from cme_pipeline.core.config import TrackingConfig
from cme_pipeline.core.stage import PipelineStage


@dataclass(frozen=True)
class TrackingMethodSpec:
    """Metadata and parameter schema for a registered tracking algorithm."""
    canonical_name: str
    stage_cls: Type[PipelineStage]
    aliases: Tuple[str, ...] = ()
    supported_parameters: FrozenSet[str] = field(default_factory=frozenset)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


_REGISTRY: Dict[str, TrackingMethodSpec] = {}
_ALIAS_MAP: Dict[str, str] = {}
_DEFAULTS_INITIALIZED = False


def register_tracking_method(spec: TrackingMethodSpec) -> None:
    """Registers a tracking algorithm and its aliases."""
    key = spec.canonical_name.lower()
    _REGISTRY[key] = spec
    _ALIAS_MAP[key] = key
    for alias in spec.aliases:
        _ALIAS_MAP[alias.lower()] = key


def _ensure_defaults_registered() -> None:
    """Lazily registers built-in validated tracking methods to avoid circular imports."""
    global _DEFAULTS_INITIALIZED
    if _DEFAULTS_INITIALIZED:
        return

    from cme_pipeline.stages.tracking.e1_tracking import E1TrackingStage
    from cme_pipeline.stages.tracking.component_first_tracking import ComponentFirstAdaptiveROITrackingStage
    from cme_pipeline.stages.tracking.multi_component_tracking import MultiComponentAdaptiveROITrackingStage
    from cme_pipeline.stages.tracking.pa_width_weighted_cpa_tracking import PAWidthWeightedCPAAdaptiveROITrackingStage
    from cme_pipeline.stages.tracking.adaptive_roi_tracking import BirthSeededAdaptiveROITrackingStage
    from cme_pipeline.stages.tracking.pa_width_proportional_tracking import PAWidthProportionalAdaptiveROITrackingStage
    from cme_pipeline.stages.tracking.birth_anchored_tracking import BirthAnchoredTrackingStage

    a3_a4_params = frozenset({
        "patch_size",
        "max_cadence_gap_sec",
        "outward_gate_enabled",
        "outward_gate_tolerance_px",
        "ema_alpha",
        "birth_gate_threshold_px",
        "default_roi_half_width_deg",
        "roi_padding_deg",
        "max_missed_frames",
    })
    a3_a4_defaults = {
        "patch_size": 4,
        "max_cadence_gap_sec": 4500.0,
        "outward_gate_enabled": False,
        "outward_gate_tolerance_px": 4.0,
        "ema_alpha": 0.1,
        "birth_gate_threshold_px": 30.0,
        "default_roi_half_width_deg": 15.0,
        "roi_padding_deg": 5.0,
        "max_missed_frames": 3,
    }

    register_tracking_method(
        TrackingMethodSpec(
            canonical_name="e1",
            stage_cls=E1TrackingStage,
            aliases=("baseline",),
            supported_parameters=frozenset({"patch_size", "max_cadence_gap_sec", "alpha"}),
            default_parameters={"patch_size": 4, "max_cadence_gap_sec": 4500.0, "alpha": 0.0},
            description="E1 1:1 Patch-First Baseline Tracking (219 tracks on 2010-12-12).",
        )
    )
    register_tracking_method(
        TrackingMethodSpec(
            canonical_name="a3",
            stage_cls=ComponentFirstAdaptiveROITrackingStage,
            aliases=("component_first",),
            supported_parameters=a3_a4_params,
            default_parameters=dict(a3_a4_defaults),
            description="A3 Component-First 1:1 Adaptive PA-ROI Tracking (50 tracks w/ outward_gate=True, 41 w/ False).",
        )
    )
    register_tracking_method(
        TrackingMethodSpec(
            canonical_name="a4",
            stage_cls=MultiComponentAdaptiveROITrackingStage,
            aliases=("multi_component",),
            supported_parameters=a3_a4_params,
            default_parameters=dict(a3_a4_defaults),
            description="A4 Multi-Component Adaptive PA-ROI Tracking (18 tracks on 2010-12-12).",
        )
    )
    register_tracking_method(
        TrackingMethodSpec(
            canonical_name="a5b",
            stage_cls=PAWidthWeightedCPAAdaptiveROITrackingStage,
            aliases=("pa_width_weighted_cpa", "weighted_cpa"),
            supported_parameters=a3_a4_params | frozenset({"min_pa_width_deg"}),
            default_parameters={**a3_a4_defaults, "min_pa_width_deg": 8.0},
            description="A5b PA-Width-Normalized CPA Distance Multi-Component Tracking (20 tracks w/ outward_gate=True).",
        )
    )
    register_tracking_method(
        TrackingMethodSpec(
            canonical_name="a0",
            stage_cls=BirthSeededAdaptiveROITrackingStage,
            aliases=("adaptive_roi", "adaptive_pa_roi", "a2"),
            supported_parameters=frozenset({
                "patch_size", "max_cadence_gap_sec", "outward_gate_enabled",
                "outward_gate_tolerance_px", "ema_alpha",
            }),
            default_parameters={
                "patch_size": 4, "max_cadence_gap_sec": 4500.0,
                "outward_gate_enabled": False, "outward_gate_tolerance_px": 4.0, "ema_alpha": 0.1,
            },
            description="Legacy A0/A2 Patch-First Adaptive PA-ROI Pilot.",
        )
    )
    register_tracking_method(
        TrackingMethodSpec(
            canonical_name="a5",
            stage_cls=PAWidthProportionalAdaptiveROITrackingStage,
            aliases=("pa_width_proportional",),
            supported_parameters=a3_a4_params | frozenset({"min_pa_width_deg"}),
            default_parameters={**a3_a4_defaults, "min_pa_width_deg": 8.0},
            description="Legacy A5 Global PA-Width Quota Pilot.",
        )
    )
    register_tracking_method(
        TrackingMethodSpec(
            canonical_name="birth_anchored",
            stage_cls=BirthAnchoredTrackingStage,
            aliases=("directional",),
            supported_parameters=frozenset({
                "alpha", "patch_size", "max_cadence_gap_sec", "direction_threshold_deg",
                "use_birth_direction", "lock_consecutive_frames", "max_direction_deviation_deg",
            }),
            default_parameters={
                "alpha": 0.0, "patch_size": 4, "max_cadence_gap_sec": 4500.0,
                "direction_threshold_deg": 30.0, "use_birth_direction": False,
                "lock_consecutive_frames": 2, "max_direction_deviation_deg": 45.0,
            },
            description="Legacy Birth-Anchored Directional Tracking.",
        )
    )

    _DEFAULTS_INITIALIZED = True


def resolve_tracking_method_name(method: str) -> str:
    """Resolves a method name or alias to its canonical registered name."""
    _ensure_defaults_registered()
    key = str(method).strip().lower()
    if key not in _ALIAS_MAP:
        available = sorted(_REGISTRY.keys())
        raise ValueError(f"Unknown tracking method '{method}'. Registered methods: {available}")
    return _ALIAS_MAP[key]


def get_tracking_spec(method: str) -> TrackingMethodSpec:
    """Returns the TrackingMethodSpec for a registered tracking method."""
    canonical = resolve_tracking_method_name(method)
    return _REGISTRY[canonical]


def get_tracking_stage(method: str) -> PipelineStage:
    """Instantiates the PipelineStage for the requested tracking method."""
    spec = get_tracking_spec(method)
    return spec.stage_cls()


def list_tracking_methods() -> List[str]:
    """Lists all canonical registered tracking method names."""
    _ensure_defaults_registered()
    return sorted(_REGISTRY.keys())


def resolve_and_validate_tracking_parameters(
    method: str,
    tracking_cfg: TrackingConfig,
) -> Dict[str, Any]:
    """Resolves effective parameters for `method` and strictly rejects unknown keys.

    Raises:
        ValueError: If `tracking_cfg.parameters` contains any key not in the selected
            method's `supported_parameters`.
    """
    spec = get_tracking_spec(method)
    user_params = dict(getattr(tracking_cfg, "parameters", {}) or {})

    # Strictly fail on unsupported keys inside tracking.parameters
    unsupported = set(user_params.keys()) - spec.supported_parameters
    if unsupported:
        raise ValueError(
            f"Unsupported parameter(s) {sorted(unsupported)} in tracking.parameters for method "
            f"'{spec.canonical_name}'. Supported parameters: {sorted(spec.supported_parameters)}"
        )

    effective = dict(spec.default_parameters)

    # Apply top-level flat fields from TrackingConfig if supported by this method
    # Note: do NOT copy flat `TrackingConfig.alpha` into `ema_alpha` because legacy
    # configs sometimes set `alpha: 0.2` when A4/A5b actually used `ema_alpha: 0.1`.
    for param_name in spec.supported_parameters:
        if hasattr(tracking_cfg, param_name):
            effective[param_name] = getattr(tracking_cfg, param_name)

    # Explicit entries in tracking.parameters take highest precedence
    for k, v in user_params.items():
        effective[k] = v

    return effective

