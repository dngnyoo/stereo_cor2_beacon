"""Optional 8-connected component size gate applied to DetectionOutput before A5b.

Preserves original C3 arrays and unavailable-frame semantics. Does not relabel
or recompute RD/P04 scores. Strictly for the pre-tracking boundary.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.ndimage import label

from cme_pipeline.core.stage import DetectionOutput

CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def filter_small_components(
    detected: DetectionOutput, *, min_component_patches: int = 2,
) -> DetectionOutput:
    """Return a copy of DetectionOutput with small P04 components removed.

    Tracking receives a consistent binary activation grid and spatial component
    labels. Unavailable observations keep their existing -1 component sentinels.
    """
    if min_component_patches < 1:
        raise ValueError("min_component_patches must be >= 1")
    active = np.asarray(detected.patch_activation_grid4, dtype=bool)
    labels = np.asarray(detected.raw_component_labels)
    if active.shape != labels.shape or active.ndim != 3 or active.shape[1:] != (64, 64):
        raise ValueError("Expected aligned (N, 64, 64) detection arrays")

    updated = active.copy()
    relabeled = labels.copy().astype(np.int32)
    removed_components = 0
    removed_patches = 0
    for i in range(len(updated)):
        usable = bool(detected.manifest_df.iloc[i].get("usable_detection", True))
        if not usable:
            if np.any(active[i]) or not np.all(labels[i] == -1):
                raise ValueError("Unavailable frame has nonempty detection")
            continue
        if np.any((labels[i] > 0) != active[i]):
            raise ValueError(f"Component labels and activation disagree at frame {i}")
        identified, count = label(active[i], structure=CONNECTIVITY_8)
        sizes = np.bincount(identified.ravel(), minlength=count + 1)
        keep = (identified > 0) & (sizes[identified] >= min_component_patches)
        removed_patches += int(np.count_nonzero(active[i] & ~keep))
        removed_components += int(np.count_nonzero(sizes[1:] < min_component_patches))
        updated[i] = keep
        # Renumber spatial component IDs contiguously after filtering.
        clean, _ = label(keep, structure=CONNECTIVITY_8)
        relabeled[i] = clean.astype(np.int32)

    metadata = dict(detected.dataset_metadata)
    metadata["pre_tracking_component_gate"] = {
        "min_component_patches": int(min_component_patches),
        "connectivity": 8,
        "removed_components": removed_components,
        "removed_patches": removed_patches,
    }
    return replace(
        detected,
        patch_activation_grid4=updated,
        raw_component_labels=relabeled,
        dataset_metadata=metadata,
    )
