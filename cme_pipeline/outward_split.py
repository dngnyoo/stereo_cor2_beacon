"""Split saved A5b tracks into new candidate identities at significant observed retreat.

This is a post-tracking experiment. It never modifies the source A5b run.
Output is a compatible virtual A5b tracking run for existing run-gates and
gate_visualization commands. Newborn segments must pass their own Birth Gate.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.live_tracking_input import load_live_tracking_input


def split_saved_a5b_on_retreat(
    tracking_run: str | Path, *, run_name: str, retreat_tolerance_px: float = 4.0,
) -> Path:
    source = Path(tracking_run).expanduser().resolve()
    meta = json.loads((source / "tracking_metadata.json").read_text())
    if meta.get("method") != "a5b":
        raise ValueError("Only saved A5b tracking runs are supported")
    if not np.isfinite(retreat_tolerance_px) or retreat_tolerance_px < 0:
        raise ValueError("retreat_tolerance_px must be finite and nonnegative")

    from cme_pipeline.core.config import PipelineConfig
    cfg = PipelineConfig()
    out = (Path(cfg.repo_root).resolve() / cfg.output_base_dir / run_name).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {out}")

    states = pd.read_csv(source / "tables" / "live_tracking_states.csv")
    masks = np.load(source / "arrays" / "track_ownership_masks.npy")
    detected = load_live_tracking_input(Path(meta["detection_dir"]).resolve())
    manifest = detected.manifest_df.reset_index(drop=True)
    radii = detected.geometry.patch_rad
    if masks.shape != detected.patch_activation_grid4.shape:
        raise ValueError("Saved ownership and detection have different shapes")
    if radii.shape != masks.shape[1:]:
        raise ValueError("Patch radius raster does not match saved tracking masks")

    index_by_pair = {int(p): i for i, p in enumerate(manifest["pair_index"])}
    observed = states[states.state.eq("OBSERVED")].copy()
    if observed.empty:
        raise ValueError("No OBSERVED states available for splitting")
    if observed.duplicated(["pair_index", "track_id"]).any():
        raise ValueError("Duplicate OBSERVED track/pair states")
    if not set(observed.pair_index.astype(int)).issubset(index_by_pair):
        raise ValueError("Observed track refers to a pair missing from manifest")

    # Preserve initial identities; allocate new IDs only when a split is needed.
    next_id = max(int(masks.max()), int(states.track_id.max()), int(meta["total_tracks"])) + 1
    relabeled = np.zeros_like(masks, dtype=np.int32)
    new_states = []
    audit = []
    for original_id, group in observed.groupby("track_id", sort=True):
        original_id = int(original_id)
        current_id = original_id
        previous_front = None
        previous_index = None
        segment = 1
        for _, row in group.sort_values("pair_index").iterrows():
            pair = int(row.pair_index)
            frame = index_by_pair[pair]
            mask = masks[frame] == original_id
            if not np.any(mask):
                raise ValueError(f"OBSERVED track {original_id} has no owned patches at pair {pair}")
            front = float(np.max(radii[mask]))
            delta = float(front - previous_front) if previous_front is not None else float("nan")

            # Splits on substantive retreat AND on any discontinuity in the
            # source observation timeline (including missing/QC frames).
            cause = None
            if previous_index is not None and frame != previous_index + 1:
                cause = "OBSERVATION_GAP"
            elif previous_front is not None and delta < -retreat_tolerance_px:
                cause = "RADIAL_RETREAT"
            if cause is not None:
                current_id = next_id
                next_id += 1
                segment += 1

            relabeled[frame][mask] = current_id
            record = row.to_dict()
            record["track_id"] = current_id
            # Birth metrics are recomputed by downstream adapter from the
            # first actual ownership mask for each new identity.
            record["source_track_id"] = original_id
            record["outward_segment_number"] = segment
            record["observed_front_radius_px"] = front
            record["observed_front_delta_px"] = delta if cause is None else float("nan")
            new_states.append(record)
            audit.append({
                "source_track_id": original_id, "new_track_id": current_id,
                "segment": segment, "pair_index": pair, "frame_index": frame,
                "observed_front_radius_px": front, "delta_from_previous_px": delta,
                "split_reason": cause or "CONTINUATION",
                "observed_patch_count": int(mask.sum()),
                "birth_r_min_px_at_first_observation": (
                    float(np.min(radii[mask])) if cause or previous_front is None else np.nan
                ),
            })
            previous_front = front
            previous_index = frame

    if not np.array_equal(relabeled > 0, masks > 0):
        raise ValueError("Split operation changed original owned patch support")
    if int(np.count_nonzero(relabeled)) != int(np.count_nonzero(masks)):
        raise ValueError("Split operation changed owned patch count")

    clean = pd.DataFrame(new_states)
    # Recalculate birth and geometry fields for each virtual identity.
    for col in ("birth_r_min_px", "is_solar_root_anchor", "original_birth_cpa",
                "original_birth_pa", "pa_min", "pa_max"):
        if col in clean:
            clean = clean.drop(columns=col)
    audit_df = pd.DataFrame(audit)
    out.mkdir(parents=True)
    (out / "arrays").mkdir()
    (out / "tables").mkdir()
    np.save(out / "arrays" / "track_ownership_masks.npy", relabeled)
    for filename in ("tracking_input_activation_p04.npy", "tracking_input_component_labels.npy"):
        src = source / "arrays" / filename
        if src.exists():
            shutil.copy2(src, out / "arrays" / filename)
    clean.to_csv(out / "tables" / "live_tracking_states.csv", index=False)
    audit_df.to_csv(out / "tables" / "outward_split_frame_audit.csv", index=False)
    beginnings = audit_df.groupby("new_track_id", sort=True).first().reset_index()
    beginnings.to_csv(out / "tables" / "outward_split_family_births.csv", index=False)
    original = int(observed.track_id.nunique())
    resulting = int(clean.track_id.nunique())
    output_meta = dict(meta)
    output_meta.update({
        "total_tracks": resulting,
        "source_tracking_run": str(source),
        "outward_split_enabled": True,
        "retreat_tolerance_px": float(retreat_tolerance_px),
        "split_original_tracks": original,
        "split_result_tracks": resulting,
        "split_added_families": resulting - original,
        "split_evaluation": "max observed owned patch radius (not predicted leading front)",
        "split_birth_policy": "each segment reevaluated at first observed patch radius",
    })
    (out / "tracking_metadata.json").write_text(json.dumps(output_meta, indent=2) + "\n")
    print(json.dumps({
        "virtual_tracking_run": str(out),
        "source_tracks": original,
        "resulting_tracks": resulting,
        "added_segments": resulting - original,
        "retreat_splits": int(audit_df.split_reason.eq("RADIAL_RETREAT").sum()),
        "gap_splits": int(audit_df.split_reason.eq("OBSERVATION_GAP").sum()),
    }, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--retreat-tolerance-px", type=float, default=4.0)
    args = parser.parse_args()
    split_saved_a5b_on_retreat(
        args.tracking_run, run_name=args.run_name,
        retreat_tolerance_px=args.retreat_tolerance_px,
    )


if __name__ == "__main__":
    main()
