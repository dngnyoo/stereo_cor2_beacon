"""Reusable Per-Track Mask Extraction & Overlap Analysis Utilities.

Provides standardized extraction of `(n_tracks, n_frames, H, W)` boolean tensors
for active patches, recovered patches, and final sub-pixel segmentations from
any completed `SegmentationOutput` without modifying union masks or forcing
arbitrary mutual exclusion on overlapping recovery corridors.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from cme_pipeline.core.stage import SegmentationOutput


def build_track_level_masks(seg_out: SegmentationOutput) -> Dict[str, Any]:
    """Extracts 3D per-track active/recovered/final pixel tensors and overlap metrics.

    Works directly on 1:1 track-to-family mappings (Identity Association or
    single-member families such as A5b + Rule B). Preserves recovery overlaps
    without arbitrary reassignment and verifies bitwise equality with the union mask.
    """
    size_out = seg_out.size_gate_output
    birth_out = size_out.birth_gate_output
    assoc_out = birth_out.association_output
    track_out = assoc_out.tracking_output

    man_df = track_out.manifest_df
    n_frames = len(man_df)

    # Determine full set of Track IDs
    track_ids = sorted(
        int(tid) for tid in track_out.track_states_df["track_id"].unique().tolist()
    )
    n_tracks = len(track_ids)
    tid_to_idx = {tid: idx for idx, tid in enumerate(track_ids)}

    own_masks = getattr(track_out, "track_ownership_masks", None)
    if own_masks is None:
        own_masks = getattr(track_out, "ownership_masks", None)
    if own_masks is None:
        raise ValueError("TrackingOutput lacks track_ownership_masks required for per-track extraction.")

    track_active_patch_masks = np.zeros((n_tracks, n_frames, 64, 64), dtype=bool)
    track_recovered_patch_masks = np.zeros((n_tracks, n_frames, 64, 64), dtype=bool)
    track_final_pixel_masks = np.zeros((n_tracks, n_frames, 256, 256), dtype=bool)

    admitted_fams = set(size_out.admitted_family_ids)
    fam_rec = size_out.family_recovered_patch_masks
    fam_pix = seg_out.family_pixel_masks

    unsplit_multi_member_fams: List[str] = []

    for tid in track_ids:
        idx = tid_to_idx[tid]
        track_active_patch_masks[idx] = (own_masks == tid)

        fid = assoc_out.track_to_family.get(tid)
        if fid is None or fid not in admitted_fams:
            continue

        members = assoc_out.family_members.get(fid, [])
        if len(members) == 1:
            rec_key = f"rec_{fid}"
            if rec_key in fam_rec:
                track_recovered_patch_masks[idx] = fam_rec[rec_key]
            if fid in fam_pix:
                track_final_pixel_masks[idx] = fam_pix[fid]
        else:
            # Multi-track family (e.g., E1 Rule B): assign family mask or track-isolated
            # Note: record multi-member status rather than arbitrarily splitting family mask
            unsplit_multi_member_fams.append(fid)

    # Bitwise union consistency audit (valid when all admitted families are 1:1)
    union_pix = seg_out.final_pixel_segmentation_union
    union_rec = size_out.recovered_patch_masks
    counts_pix = track_final_pixel_masks.sum(axis=0)  # (n_frames, 256, 256)

    exact_pixel_union_match = bool(np.array_equal(np.any(track_final_pixel_masks, axis=0), union_pix))
    exact_patch_union_match = bool(np.array_equal(np.any(track_recovered_patch_masks, axis=0), union_rec))

    shared_pixel_locations = int((counts_pix > 1).sum())
    individual_pixel_sum = int(track_final_pixel_masks.sum())
    union_pixel_sum = int(union_pix.sum())

    # Pairwise overlap table
    overlap_rows = []
    for i in range(n_tracks):
        tid_a = track_ids[i]
        tot_a = int(track_final_pixel_masks[i].sum())
        if tot_a == 0:
            continue
        for j in range(i + 1, n_tracks):
            tid_b = track_ids[j]
            tot_b = int(track_final_pixel_masks[j].sum())
            if tot_b == 0:
                continue
            ov = int((track_final_pixel_masks[i] & track_final_pixel_masks[j]).sum())
            if ov > 0:
                overlap_rows.append({
                    "track_pair": f"T{tid_a:02d}_T{tid_b:02d}",
                    "track_a_id": tid_a,
                    "track_b_id": tid_b,
                    "overlap_pixels": ov,
                    "track_a_pixels": tot_a,
                    "track_b_pixels": tot_b,
                })

    return {
        "track_ids": track_ids,
        "tid_to_idx": tid_to_idx,
        "track_active_patch_masks": track_active_patch_masks,
        "track_recovered_patch_masks": track_recovered_patch_masks,
        "track_final_pixel_masks": track_final_pixel_masks,
        "union_pixel_sum": union_pixel_sum,
        "individual_pixel_sum": individual_pixel_sum,
        "shared_pixel_locations": shared_pixel_locations,
        "shared_overlap_difference": individual_pixel_sum - union_pixel_sum,
        "exact_pixel_union_match": exact_pixel_union_match,
        "exact_patch_union_match": exact_patch_union_match,
        "unsplit_multi_member_families": sorted(set(unsplit_multi_member_fams)),
        "overlap_summary_df": pd.DataFrame(overlap_rows),
    }

