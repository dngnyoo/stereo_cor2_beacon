"""Tracking stages, compatibility adapter, and algorithm registry for CME Pipeline."""

from cme_pipeline.stages.tracking.e1_tracking import E1TrackingStage
from cme_pipeline.stages.tracking.birth_anchored_tracking import BirthAnchoredTrackingStage
from cme_pipeline.stages.tracking.adaptive_roi_tracking import BirthSeededAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.component_first_tracking import ComponentFirstAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.multi_component_tracking import MultiComponentAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.pa_width_proportional_tracking import PAWidthProportionalAdaptiveROITrackingStage
from cme_pipeline.stages.tracking.pa_width_weighted_cpa_tracking import PAWidthWeightedCPAAdaptiveROITrackingStage
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
