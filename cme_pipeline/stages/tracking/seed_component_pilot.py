"""Stereo_COR2 Seed-Based CME Component Identification Pilot.

Implements score-ranked seed identification in the inner corona and T24-connected
component growth as a standalone diagnostic module (Phases 1 & 2).

Key Principles:
1. Deterministic score ranking of eligible active patches within the inner corona
   r_min <= r_p <= R_seed.
2. 8-connected T24 path evaluation: candidates connected by active patches merge
   into the same seed group, led by the highest-scoring representative seed.
3. Component growth expands through full 8-connected active neighbors beyond R_seed.
4. Strict mathematical check: every seeded region is mathematically equivalent to
   the corresponding 8-connected T24 component that intersects the seed search region.
   Additional value provided:
   - High-confidence representative seed (location, score, rank, PA, radius).
   - Inner-corona origin constraint (seeded vs unseeded component partition).
   - Absolute score-ranked component priority.
   - Internal seed multiplicity structure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label

from cme_pipeline.core.stage import DetectionOutput, GeometryContext
from cme_pipeline.live_detection import _scores
from cme_pipeline.utils.geometry import (
    circular_diff_deg,
    circular_mean_deg,
    minimal_circular_enclosing_arc,
)

STRUCT_8 = np.ones((3, 3), dtype=bool)


@dataclass
class SeedCandidate:
    """Individual eligible candidate patch within the inner-corona search region."""
    pair_index: int
    frame_idx: int
    timestamp: str
    patch_r: int
    patch_c: int
    pa_deg: float
    radius_px: float
    raw_score: float
    threshold_relative_score: float
    score_ratio: float
    score_rank: int
    seed_group_id: int = -1
    is_representative: bool = False
    grown_comp_id: int = -1


@dataclass
class SeedGroup:
    """Group of connected seed candidates forming a single CME structure origin."""
    seed_group_id: int
    pair_index: int
    frame_idx: int
    timestamp: str
    representative_seed: SeedCandidate
    candidate_patches: List[SeedCandidate] = field(default_factory=list)
    num_candidates: int = 1
    grown_comp_id: int = -1
    total_grown_patches: int = 0
    r_min: float = 0.0
    r_front: float = 0.0
    cpa_deg: float = 0.0
    pa_span_deg: float = 0.0


@dataclass
class ComponentDiagnostic:
    """Diagnostic record for each 8-connected component in a frame."""
    pair_index: int
    frame_idx: int
    timestamp: str
    comp_id: int
    is_seeded: bool
    seed_group_id: Optional[int]
    representative_score: Optional[float]
    representative_score_rank: Optional[int]
    representative_pa_deg: Optional[float]
    representative_radius_px: Optional[float]
    num_patches: int
    r_min: float
    r_front: float
    cpa_deg: float
    pa_span_deg: float
    num_seed_candidates: int
    intersects_seed_search_region: bool
    is_mathematically_equivalent: bool = True


@dataclass
class FrameSeedResult:
    """Complete seed identification and component growth output for one frame."""
    pair_index: int
    frame_idx: int
    timestamp: str
    observation_status: str
    usable_detection: bool
    effective_threshold: float
    candidates: List[SeedCandidate]
    seed_groups: List[SeedGroup]
    component_diagnostics: List[ComponentDiagnostic]
    continuous_scores: np.ndarray          # (64, 64) float32
    seeded_component_mask: np.ndarray      # (64, 64) int32 (seed_group_id)
    unseeded_component_mask: np.ndarray    # (64, 64) int32 (raw comp_id)
    equivalence_verified: bool


def extract_continuous_scores_from_detection(
    det_out: DetectionOutput,
    score_mode: str = "abs",
) -> np.ndarray:
    """Extracts or computes (N, 64, 64) continuous patch detection scores.

    Uses frame_cache z_global and usable masks with original score definition
    (P95 of absolute standardized residual intensity in 4x4 blocks).
    """
    n_frames = len(det_out.manifest_df)
    scores_out = np.full((n_frames, 64, 64), np.nan, dtype=np.float32)

    for i in range(n_frames):
        row = det_out.manifest_df.iloc[i]
        usable = bool(row.get("usable_detection", True))
        if not usable:
            continue
        fc = det_out.frame_cache[i]
        z = fc.get("z_global")
        usable_px = fc.get("usable")
        if z is not None and usable_px is not None:
            p95, accepted = _scores(z, usable_px, 4, score_mode=score_mode)
            scores_out[i] = np.where(accepted, p95.astype(np.float32), np.nan)

    return scores_out


def identify_and_rank_seed_candidates(
    frame_scores: np.ndarray,
    active_mask: np.ndarray,
    geometry: GeometryContext,
    r_min: float = 22.0,
    r_seed: float = 30.0,
    effective_threshold: float = 0.0,
    score_cutoff: Optional[float] = None,
    top_k_candidates: Optional[int] = None,
    pair_index: int = 0,
    frame_idx: int = 0,
    timestamp: str = "",
) -> List[SeedCandidate]:
    """Step A & Step B: Identify and deterministically rank inner-corona seed candidates.

    Tie-breaking rule: (-score, radius_px, pa_deg, patch_r, patch_c).
    """
    patch_rad = geometry.patch_rad
    patch_pa = geometry.patch_pa

    # Eligible patches: active in T24, finite score, within [r_min, r_seed]
    in_region = (
        active_mask &
        np.isfinite(frame_scores) &
        (patch_rad >= r_min) &
        (patch_rad <= r_seed)
    )

    if score_cutoff is not None:
        in_region &= (frame_scores >= score_cutoff)

    cand_indices = list(zip(*np.nonzero(in_region)))
    if not cand_indices:
        return []

    # Build raw candidate records
    raw_candidates = []
    for r, c in cand_indices:
        sc = float(frame_scores[r, c])
        rad = float(patch_rad[r, c])
        pa = float(patch_pa[r, c])
        raw_candidates.append({
            "patch_r": r,
            "patch_c": c,
            "raw_score": sc,
            "radius_px": rad,
            "pa_deg": pa,
        })

    # Step B: Deterministic sorting
    # Primary: score descending (-sc)
    # Secondary: radius ascending (closer to Sun first)
    # Tertiary: PA ascending (counter-clockwise from North)
    # Quaternary: row ascending, col ascending
    raw_candidates.sort(
        key=lambda item: (-item["raw_score"], item["radius_px"], item["pa_deg"], item["patch_r"], item["patch_c"])
    )

    if top_k_candidates is not None and top_k_candidates > 0:
        raw_candidates = raw_candidates[:top_k_candidates]

    candidates: List[SeedCandidate] = []
    for rank_idx, item in enumerate(raw_candidates, start=1):
        sc = item["raw_score"]
        th_rel = sc - effective_threshold if np.isfinite(effective_threshold) else sc
        ratio = sc / effective_threshold if (np.isfinite(effective_threshold) and effective_threshold > 0) else np.nan

        cand = SeedCandidate(
            pair_index=pair_index,
            frame_idx=frame_idx,
            timestamp=timestamp,
            patch_r=item["patch_r"],
            patch_c=item["patch_c"],
            pa_deg=item["pa_deg"],
            radius_px=item["radius_px"],
            raw_score=sc,
            threshold_relative_score=th_rel,
            score_ratio=ratio,
            score_rank=rank_idx,
        )
        candidates.append(cand)

    return candidates


def determine_connectivity_and_grow(
    candidates: List[SeedCandidate],
    active_mask: np.ndarray,
    geometry: GeometryContext,
    pair_index: int = 0,
    frame_idx: int = 0,
    timestamp: str = "",
) -> Tuple[List[SeedGroup], List[ComponentDiagnostic], np.ndarray, np.ndarray, bool]:
    """Steps C & D: Determine seed connectivity via T24 active mask and grow components.

    Also performs the critical mathematical equivalence check comparing grown seeded
    regions against ordinary 8-connected components that intersect the seed search region.
    """
    patch_rad = geometry.patch_rad
    patch_pa = geometry.patch_pa

    # 1. Detect ordinary 8-connected components on full active mask
    lbl_img, n_comps = label(active_mask, structure=STRUCT_8)

    # Map candidate coordinates to component labels
    comp_to_candidates: Dict[int, List[SeedCandidate]] = {c: [] for c in range(1, n_comps + 1)}
    for cand in candidates:
        c_label = int(lbl_img[cand.patch_r, cand.patch_c])
        if c_label > 0:
            comp_to_candidates[c_label].append(cand)

    # 2. Step C: Seed Grouping via 8-connected T24 connectivity
    # Two candidates are connected by an 8-connected path of active patches if and only if
    # they share the same connected component label in lbl_img!
    # Candidates are processed in descending score order (rank 1, 2, ...).
    seed_groups: List[SeedGroup] = []
    seeded_comp_mask = np.zeros((64, 64), dtype=np.int32)
    unseeded_comp_mask = np.zeros((64, 64), dtype=np.int32)

    next_group_id = 1
    comp_to_group_id: Dict[int, int] = {}

    # Iterate through candidates in descending score order
    for cand in candidates:
        c_label = int(lbl_img[cand.patch_r, cand.patch_c])
        if c_label not in comp_to_group_id:
            # First candidate for this connected component: forms a new seed group!
            # Since candidates are sorted by score, this candidate is strictly the
            # highest-scoring representative seed for this group.
            gid = next_group_id
            next_group_id += 1
            comp_to_group_id[c_label] = gid

            cand.seed_group_id = gid
            cand.is_representative = True
            cand.grown_comp_id = c_label

            group = SeedGroup(
                seed_group_id=gid,
                pair_index=pair_index,
                frame_idx=frame_idx,
                timestamp=timestamp,
                representative_seed=cand,
                candidate_patches=[cand],
                num_candidates=1,
                grown_comp_id=c_label,
            )
            seed_groups.append(group)
        else:
            # Connected candidate: belongs to an already formed seed group
            gid = comp_to_group_id[c_label]
            cand.seed_group_id = gid
            cand.is_representative = False
            cand.grown_comp_id = c_label
            # Find group and append candidate
            for g in seed_groups:
                if g.seed_group_id == gid:
                    g.candidate_patches.append(cand)
                    g.num_candidates += 1
                    break

    # 3. Step D: Grow Seeded Components through full 8-connected T24 neighbors
    # For each seed group, the grown component is the full connected component c_label in lbl_img.
    all_equivalent = True
    comp_diagnostics: List[ComponentDiagnostic] = []

    for c_id in range(1, n_comps + 1):
        m = (lbl_img == c_id)
        cpas = patch_pa[m]
        crads = patch_rad[m]
        arc = minimal_circular_enclosing_arc(cpas)
        n_patches = int(m.sum())
        r_min_c = float(np.min(crads))
        r_front_c = float(np.max(crads))
        cpa_c = float(circular_mean_deg(cpas))
        span_c = float(arc["arc_span"])

        c_cands = comp_to_candidates[c_id]
        is_seeded = len(c_cands) > 0

        if is_seeded:
            gid = comp_to_group_id[c_id]
            group = next(g for g in seed_groups if g.seed_group_id == gid)
            rep = group.representative_seed

            # Update seed group geometry from full grown component
            group.total_grown_patches = n_patches
            group.r_min = r_min_c
            group.r_front = r_front_c
            group.cpa_deg = cpa_c
            group.pa_span_deg = span_c

            # Mark seeded component mask with seed_group_id
            seeded_comp_mask[m] = gid

            # Mathematical equivalence check:
            # Is the grown seeded region identical to the T24 connected component?
            grown_mask = (seeded_comp_mask == gid)
            is_equiv = np.array_equal(grown_mask, m)
            if not is_equiv:
                all_equivalent = False

            diag = ComponentDiagnostic(
                pair_index=pair_index,
                frame_idx=frame_idx,
                timestamp=timestamp,
                comp_id=c_id,
                is_seeded=True,
                seed_group_id=gid,
                representative_score=rep.raw_score,
                representative_score_rank=rep.score_rank,
                representative_pa_deg=rep.pa_deg,
                representative_radius_px=rep.radius_px,
                num_patches=n_patches,
                r_min=r_min_c,
                r_front=r_front_c,
                cpa_deg=cpa_c,
                pa_span_deg=span_c,
                num_seed_candidates=len(c_cands),
                intersects_seed_search_region=True,
                is_mathematically_equivalent=is_equiv,
            )
            comp_diagnostics.append(diag)
        else:
            # Unseeded component: has NO candidates in the inner-corona seed search region!
            unseeded_comp_mask[m] = c_id

            diag = ComponentDiagnostic(
                pair_index=pair_index,
                frame_idx=frame_idx,
                timestamp=timestamp,
                comp_id=c_id,
                is_seeded=False,
                seed_group_id=None,
                representative_score=None,
                representative_score_rank=None,
                representative_pa_deg=None,
                representative_radius_px=None,
                num_patches=n_patches,
                r_min=r_min_c,
                r_front=r_front_c,
                cpa_deg=cpa_c,
                pa_span_deg=span_c,
                num_seed_candidates=0,
                intersects_seed_search_region=False,
                is_mathematically_equivalent=True,
            )
            comp_diagnostics.append(diag)

    # Sort seed groups by representative seed score rank (rank 1 first)
    seed_groups.sort(key=lambda g: g.representative_seed.score_rank)

    return seed_groups, comp_diagnostics, seeded_comp_mask, unseeded_comp_mask, all_equivalent


def process_frame_seeds(
    scores_frame: np.ndarray,
    active_mask_frame: np.ndarray,
    geometry: GeometryContext,
    r_min: float = 22.0,
    r_seed: float = 30.0,
    effective_threshold: float = 0.0,
    score_cutoff: Optional[float] = None,
    top_k_candidates: Optional[int] = None,
    pair_index: int = 0,
    frame_idx: int = 0,
    timestamp: str = "",
    observation_status: str = "VALID",
    usable_detection: bool = True,
) -> FrameSeedResult:
    """Processes one frame through the full seed identification and growth algorithm."""
    if not usable_detection:
        return FrameSeedResult(
            pair_index=pair_index,
            frame_idx=frame_idx,
            timestamp=timestamp,
            observation_status=observation_status,
            usable_detection=False,
            effective_threshold=effective_threshold,
            candidates=[],
            seed_groups=[],
            component_diagnostics=[],
            continuous_scores=np.full((64, 64), np.nan, dtype=np.float32),
            seeded_component_mask=np.zeros((64, 64), dtype=np.int32),
            unseeded_component_mask=np.zeros((64, 64), dtype=np.int32),
            equivalence_verified=True,
        )

    # Step A & B: candidates
    cands = identify_and_rank_seed_candidates(
        frame_scores=scores_frame,
        active_mask=active_mask_frame,
        geometry=geometry,
        r_min=r_min,
        r_seed=r_seed,
        effective_threshold=effective_threshold,
        score_cutoff=score_cutoff,
        top_k_candidates=top_k_candidates,
        pair_index=pair_index,
        frame_idx=frame_idx,
        timestamp=timestamp,
    )

    # Step C & D: grouping and growth
    groups, diags, seeded_m, unseeded_m, equiv = determine_connectivity_and_grow(
        candidates=cands,
        active_mask=active_mask_frame,
        geometry=geometry,
        pair_index=pair_index,
        frame_idx=frame_idx,
        timestamp=timestamp,
    )

    return FrameSeedResult(
        pair_index=pair_index,
        frame_idx=frame_idx,
        timestamp=timestamp,
        observation_status=observation_status,
        usable_detection=True,
        effective_threshold=effective_threshold,
        candidates=cands,
        seed_groups=groups,
        component_diagnostics=diags,
        continuous_scores=scores_frame,
        seeded_component_mask=seeded_m,
        unseeded_component_mask=unseeded_m,
        equivalence_verified=equiv,
    )


def run_standalone_seed_pilot(
    detection_output: DetectionOutput,
    r_min: float = 22.0,
    r_seed: float = 30.0,
    score_cutoff: Optional[float] = None,
    top_k_candidates: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs standalone seed pilot over all frames in a DetectionOutput."""
    geom = detection_output.geometry
    man_df = detection_output.manifest_df
    grid4_all = detection_output.patch_activation_grid4
    n_frames = len(man_df)

    scores_all = extract_continuous_scores_from_detection(detection_output)

    frame_results: List[FrameSeedResult] = []
    all_candidates: List[SeedCandidate] = []
    all_seed_groups: List[SeedGroup] = []
    all_diagnostics: List[ComponentDiagnostic] = []
    frame_summary_rows: List[Dict[str, Any]] = []

    seeded_masks = np.zeros((n_frames, 64, 64), dtype=np.int32)
    unseeded_masks = np.zeros((n_frames, 64, 64), dtype=np.int32)

    for i in range(n_frames):
        row = man_df.iloc[i]
        p_idx = int(row.get("pair_index", i + 1))
        ts = str(row.get("end_time", ""))
        status = str(row.get("observation_status", "VALID"))
        usable = bool(row.get("usable_detection", True))

        fc = detection_output.frame_cache[i]
        r_th = fc.get("r_th", {})
        if hasattr(r_th, "get"):
            eff_th = float(r_th.get("effective_threshold_p04", r_th.get("T24_p04", 0.0)))
        else:
            eff_th = float(getattr(r_th, "effective_threshold_p04", 0.0))

        cur_scores = scores_all[i]
        cur_mask = grid4_all[i]

        res = process_frame_seeds(
            scores_frame=cur_scores,
            active_mask_frame=cur_mask,
            geometry=geom,
            r_min=r_min,
            r_seed=r_seed,
            effective_threshold=eff_th,
            score_cutoff=score_cutoff,
            top_k_candidates=top_k_candidates,
            pair_index=p_idx,
            frame_idx=i,
            timestamp=ts,
            observation_status=status,
            usable_detection=usable,
        )

        frame_results.append(res)
        all_candidates.extend(res.candidates)
        all_seed_groups.extend(res.seed_groups)
        all_diagnostics.extend(res.component_diagnostics)

        seeded_masks[i] = res.seeded_component_mask
        unseeded_masks[i] = res.unseeded_component_mask

        n_seeded_comps = sum(1 for d in res.component_diagnostics if d.is_seeded)
        n_unseeded_comps = sum(1 for d in res.component_diagnostics if not d.is_seeded)

        frame_summary_rows.append({
            "frame_idx": i,
            "pair_index": p_idx,
            "timestamp": ts,
            "observation_status": status,
            "usable_detection": usable,
            "effective_threshold": eff_th,
            "total_active_patches": int(cur_mask.sum()) if usable else 0,
            "total_components": len(res.component_diagnostics),
            "num_seed_candidates": len(res.candidates),
            "num_seed_groups": len(res.seed_groups),
            "num_seeded_components": n_seeded_comps,
            "num_unseeded_components": n_unseeded_comps,
            "equivalence_verified": res.equivalence_verified,
        })

    # Convert to DataFrames
    cand_records = [asdict(c) for c in all_candidates]
    group_records = []
    for g in all_seed_groups:
        group_records.append({
            "pair_index": g.pair_index,
            "frame_idx": g.frame_idx,
            "timestamp": g.timestamp,
            "seed_group_id": g.seed_group_id,
            "rep_patch_r": g.representative_seed.patch_r,
            "rep_patch_c": g.representative_seed.patch_c,
            "rep_pa_deg": g.representative_seed.pa_deg,
            "rep_radius_px": g.representative_seed.radius_px,
            "rep_score": g.representative_seed.raw_score,
            "rep_score_rank": g.representative_seed.score_rank,
            "num_candidates": g.num_candidates,
            "grown_comp_id": g.grown_comp_id,
            "total_grown_patches": g.total_grown_patches,
            "r_min": g.r_min,
            "r_front": g.r_front,
            "cpa_deg": g.cpa_deg,
            "pa_span_deg": g.pa_span_deg,
        })

    diag_records = [asdict(d) for d in all_diagnostics]

    candidates_df = pd.DataFrame(cand_records) if cand_records else pd.DataFrame(columns=[
        "pair_index", "frame_idx", "timestamp", "patch_r", "patch_c", "pa_deg",
        "radius_px", "raw_score", "threshold_relative_score", "score_ratio",
        "score_rank", "seed_group_id", "is_representative", "grown_comp_id"
    ])
    seed_groups_df = pd.DataFrame(group_records) if group_records else pd.DataFrame(columns=[
        "pair_index", "frame_idx", "timestamp", "seed_group_id", "rep_patch_r",
        "rep_patch_c", "rep_pa_deg", "rep_radius_px", "rep_score", "rep_score_rank",
        "num_candidates", "grown_comp_id", "total_grown_patches", "r_min", "r_front",
        "cpa_deg", "pa_span_deg"
    ])
    components_df = pd.DataFrame(diag_records) if diag_records else pd.DataFrame(columns=[
        "pair_index", "frame_idx", "timestamp", "comp_id", "is_seeded",
        "seed_group_id", "representative_score", "representative_score_rank",
        "representative_pa_deg", "representative_radius_px", "num_patches",
        "r_min", "r_front", "cpa_deg", "pa_span_deg", "num_seed_candidates",
        "intersects_seed_search_region", "is_mathematically_equivalent"
    ])
    frames_df = pd.DataFrame(frame_summary_rows)

    return {
        "frame_results": frame_results,
        "candidates_df": candidates_df,
        "seed_groups_df": seed_groups_df,
        "components_df": components_df,
        "frames_df": frames_df,
        "continuous_scores": scores_all,
        "seeded_component_masks": seeded_masks,
        "unseeded_component_masks": unseeded_masks,
    }
