"""Execute Gates from an explicitly selected, completed Association Run.

The saved Association membership is authoritative: this module reconstructs
family membership from recorded CSV, never reruns Identity Association.
Tracking and Detection ancestors are verified before creating a new Run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.run_layout import initialize_run, complete_stage, verify_stage
from cme_pipeline.core.stage import AssociationOutput
from cme_pipeline.core.standard_stages import stages_until
from cme_pipeline.standard_runner import _revision, _context, _params, _stage_outputs
from cme_pipeline.standard_tracking_parent import resolve_tracking_parent
from cme_pipeline.stages.gates.v01_birth_radius_gate import FamilyBirthRadiusGateStage
from cme_pipeline.stages.gates.v01_family_size_gate import FamilySizeGateStage


def restore_association(membership: pd.DataFrame, edges: pd.DataFrame,
                        tracking, *, birth_radius_px: float) -> AssociationOutput:
    """Restore canonical identity membership and gate-dependent anchor eligibility.

    Family identity comes *only* from stored Association output. Gate anchors
    are evaluated from original observed birth_r_min_px against the current
    gate threshold, so threshold experiments do not re-run Association.
    """
    required = {"pair_index", "track_id", "family_id", "is_solar_root_anchor",
                "membership_status", "link_type"}
    if membership.empty or not required.issubset(membership.columns):
        raise ValueError("Saved Association membership has missing columns or no observations")
    if membership[list(("pair_index", "track_id"))].duplicated().any():
        raise ValueError("Duplicate pair/track membership")
    observed = tracking.track_states_df
    observed = observed.loc[observed["state"] == "OBSERVED", ["pair_index", "track_id"]].copy()
    saved_keys = membership[["pair_index", "track_id"]].copy()
    if (len(saved_keys) != len(observed)
            or set(map(tuple, saved_keys.to_numpy())) != set(map(tuple, observed.to_numpy()))):
        raise ValueError("Association observations differ from verified Tracking states")
    if not membership["link_type"].eq("DIRECT_IDENTITY_MAPPING").all():
        raise ValueError("Only stored Identity Association memberships are supported")
    family_members = {}
    track_to_family = {}
    for tid, group in membership.groupby("track_id", sort=True):
        tid = int(tid)
        fids = group["family_id"].unique().tolist()
        expected = f"FAM_T{tid:02d}"
        if fids != [expected]:
            raise ValueError(f"Non-canonical or inconsistent identity family for track {tid}")
        track_to_family[tid] = expected
        family_members[expected] = [tid]
        recorded_anchors = group["is_solar_root_anchor"].unique().tolist()
        if len(recorded_anchors) != 1 or not isinstance(recorded_anchors[0], (bool, np.bool_)):
            raise ValueError(f"Inconsistent saved anchor flag for track {tid}")
    if not np.isfinite(birth_radius_px) or birth_radius_px <= 0:
        raise ValueError("birth_max_radius_px must be finite and positive")
    family_anchors = {}
    first = tracking.track_states_df.loc[
        tracking.track_states_df["state"] == "OBSERVED"
    ].sort_values(["track_id", "pair_index"]).groupby("track_id").first()
    if "birth_r_min_px" not in first.columns:
        raise ValueError("Tracking state lacks birth_r_min_px; cannot evaluate birth Gate")
    for tid, family in track_to_family.items():
        rmin = float(first.loc[tid, "birth_r_min_px"])
        if np.isfinite(rmin) and rmin <= birth_radius_px:
            family_anchors[family] = [tid]
    return AssociationOutput(
        tracking_output=tracking,
        recovery_output=None,
        track_to_family=track_to_family,
        family_members=family_members,
        family_anchors=family_anchors,
        edges_df=edges,
        track_membership_df=membership,
        candidate_family_count=len(family_members),
    )


def resolve_association_parent(config, input_run):
    parent = Path(input_run).expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    manifest = json.loads((parent / "run_manifest.json").read_text())
    if manifest["stages"].get("association") != "COMPLETED":
        raise ValueError("Input Run has no completed Association stage")
    stage = verify_stage(parent, "association")
    if stage.get("implementation") != "identity" or config.method("association") != "identity":
        raise ValueError("Only saved Identity Association is currently supported")
    old = json.loads((parent / "config_snapshot.json").read_text())
    if str(old["experiment"]["date"]) != str(config.data["experiment"]["date"]):
        raise ValueError("Association Run date mismatch")
    if (old.get("association") or {}) != _params(config, "association"):
        raise ValueError("Association configuration changed; create a new Association Run")

    origin = manifest.get("input_provenance") or {}
    if manifest["stages"].get("tracking") == "COMPLETED":
        tracking_parent = parent
    elif origin.get("input_stage") == "tracking" and origin.get("input_run"):
        tracking_parent = Path(origin["input_run"]).resolve()
    else:
        raise ValueError("Association Run has no verifiable Tracking ancestor")
    tracking_root, tracking, tracking_meta, detection_root, extra = resolve_tracking_parent(
        config, tracking_parent
    )

    def artifact(key):
        item = stage["outputs"][key]
        file = (parent / item["path"]).resolve()
        if not file.is_relative_to(parent):
            raise ValueError("Association artifact outside its Run")
        return file

    membership = pd.read_csv(artifact("family_membership"))
    edges = pd.read_csv(artifact("association_decisions"))
    gate_radius = float((_params(config, "gates")).get("birth_max_radius_px", 30.0))
    association = restore_association(
        membership, edges, tracking, birth_radius_px=gate_radius
    )
    # The saved anchor flag represents the *original* association threshold,
    # and must agree with original tracking birth radii. The new Gate threshold
    # is applied separately above, enabling parameter sweeps.
    original_radius = float((old.get("gates") or {}).get("birth_max_radius_px", 30.0))
    original_anchor_ids = {
        tid for tid, group in membership.groupby("track_id")
        if bool(group["is_solar_root_anchor"].iloc[0])
    }
    recorded = restore_association(membership, edges, tracking,
                                   birth_radius_px=original_radius)
    inferred = {tid for ids in recorded.family_anchors.values() for tid in ids}
    if original_anchor_ids != inferred:
        raise ValueError("Saved Association anchor flags do not match source Tracking radii")
    return parent, stage, tracking_root, tracking_meta, detection_root, association


def execute_from_association(config, *, input_run, until, run_dir, visualize=None):
    if len(stages_until(until)) != 5:
        raise ValueError("Association input currently supports --until gates only")
    if config.vis_mode("gates", visualize) != "off":
        raise NotImplementedError("Gate visualization is not integrated")
    gates = _params(config, "gates")
    if config.method("gates") != "consecutive_size_qualified":
        raise ValueError("Only consecutive-size-qualified Gates supported")
    if (gates.get("outward_split") or {}).get("enabled", False):
        raise ValueError("Outward split is not part of the baseline")
    persistence = gates.get("persistence") or {}
    if persistence.get("method", "consecutive_size_qualified") != "consecutive_size_qualified":
        raise ValueError("Unsupported persistence method")
    parent, stage, tracking_root, tracking_meta, detection_root, assoc = resolve_association_parent(
        config, input_run
    )
    output = Path(run_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    cfg = PipelineConfig()
    cfg.date_str = int(config.data["experiment"]["date"])
    cfg.association.method = "identity"
    cfg.birth_gate.max_birth_radius_px = float(gates.get("birth_max_radius_px", 30.0))
    cfg.size_gate.min_detected_patches = int(persistence.get("min_detected_patches", 10))
    cfg.size_gate.min_observed_pairs = int(persistence.get("min_observed_pairs", 3))
    cfg.size_gate.size_qualified_persistence = True
    cfg.size_gate.consecutive_size_qualified = True
    if cfg.size_gate.min_detected_patches < 1 or cfg.size_gate.min_observed_pairs < 1:
        raise ValueError("Gate persistence thresholds must be >=1")
    initialize_run(output, config.data, _revision(), {
        "mode": "explicit_input_run",
        "input_run": str(parent),
        "input_stage": "association",
        "tracking_ancestor": str(tracking_root),
        "detection_ancestor": str(detection_root),
        "input_outputs": stage["outputs"],
        "tracking_outputs": tracking_meta["outputs"],
    })
    ctx = _context(output, "gates")
    birth = FamilyBirthRadiusGateStage().run(assoc, cfg, ctx)
    size = FamilySizeGateStage().run(birth, cfg, ctx)
    admitted = ctx.run_dir / "admitted_families.json"
    admitted.write_text(json.dumps({
        "family_ids": sorted(size.admitted_family_ids),
        "track_ids": sorted(size.admitted_track_ids),
    }, indent=2))
    qualified = ctx.run_dir / "qualified_pairs.csv"
    size.family_summary_df[["family_id", "size_qualified_pairs",
                            "longest_consecutive_size_qualified"]].to_csv(qualified, index=False)
    output_files = _stage_outputs(output, {
        "gate_decisions": ctx.tables_dir / "04_full_day_family_summary.csv",
        "admitted_family_ids": admitted,
        "qualified_pair_ranges": qualified,
    })
    complete_stage(output, "gates", {
        "parent_run": str(parent),
        "family_membership": stage["outputs"]["family_membership"]["path"],
    }, output_files, "birth+consecutive_size_qualified", gates)
    return output
