"""Tracking stages, compatibility adapter, and algorithm registry for CME Pipeline."""

from cme_pipeline.stages.tracking.v01_baseline_tracking import E1TrackingStage
from cme_pipeline.stages.tracking.birth_anchored_tracking import BirthAnchoredTrackingStage
from cme_pipeline.stages.tracking.v02_adaptive_pa_roi import BirthSeededAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.v04_component_first import ComponentFirstAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.v05_multi_component import MultiComponentAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.v06_pa_width_proportional import PAWidthProportionalAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.v07_pa_width_weighted_cpa import PAWidthWeightedCPAAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.adapter import (
    adapt_tracking_output,
    get_track_observed_mask,
    get_track_family_observed_mask,
)
from cme_pipeline.stages.tracking.registry import (
    TrackingMethodSpec,
    register_tracking_method,
    get_tracking_stage,
    get_tracking_spec,
    list_tracking_methods,
    resolve_tracking_method_name,
    resolve_and_validate_tracking_parameters,
)

__all__ = [
    "E1TrackingStage",
    "BirthAnchoredTrackingStage",
    "BirthSeededAdaptiveROITrackingStage",
    "ComponentFirstAdaptiveROITrackingStage",
    "MultiComponentAdaptiveROITrackingStage",
    "PAWidthProportionalAdaptiveROITrackingStage",
    "PAWidthWeightedCPAAdaptiveROITrackingStage",
    "adapt_tracking_output",
    "get_track_observed_mask",
    "get_track_family_observed_mask",
    "TrackingMethodSpec",
    "register_tracking_method",
    "get_tracking_stage",
    "get_tracking_spec",
    "list_tracking_methods",
    "resolve_tracking_method_name",
    "resolve_and_validate_tracking_parameters",
]
