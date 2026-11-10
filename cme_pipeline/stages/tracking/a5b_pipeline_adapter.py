"""Legacy Backwards-Compatibility Wrapper for A5b Experiment 07 Runners.

Consolidated into `cme_pipeline.stages.tracking.adapter.adapt_tracking_output`.
This module is preserved solely so untouched Experiment 07 scripts
(`run_07_a5b_end_to_end_experiment.py`, `run_07_phase3_b2_and_visualizations.py`)
continue to execute with zero breaking changes. New code should use
`adapt_tracking_output` directly, which keeps Component IDs and Track IDs distinct.
"""

from __future__ import annotations

import warnings

from cme_pipeline.core.stage import TrackingOutput
from cme_pipeline.stages.tracking.adapter import adapt_tracking_output


def adapt_a5b_tracking_output(
    a5b_output: TrackingOutput,
    legacy_conflate_ids: bool = True,
) -> TrackingOutput:
    """Deprecated wrapper around `adapt_tracking_output`.

    Args:
        a5b_output: Output from PAWidthWeightedCPAAdaptiveROITrackingStage.
        legacy_conflate_ids: When True (default for legacy Exp 07 runner parity),
            populates legacy `component_id = track_id` and aliases
            `patch_component_labels` to `track_ownership_masks` on the shallow copy
            without mutating `a5b_output`.
    """
    warnings.warn(
        "adapt_a5b_tracking_output is deprecated; use cme_pipeline.stages.tracking.adapter.adapt_tracking_output",
        DeprecationWarning,
        stacklevel=2,
    )
    adapted = adapt_tracking_output(a5b_output)
    if legacy_conflate_ids:
        df = adapted.track_states_df.copy()
        df["component_id"] = df["track_id"]
        adapted.track_states_df = df
        if adapted.track_ownership_masks is not None:
            adapted.patch_component_labels = adapted.track_ownership_masks
    return adapted
