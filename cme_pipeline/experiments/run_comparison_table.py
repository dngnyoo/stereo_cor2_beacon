"""Cross-Run Quantitative Comparison Generator for CME Pipeline.

Compares:
- Run A: Baseline Rule B (Original E1 Tracking + Pure DO Association)
- Run B: Variant B (Original E1 Tracking + Birth-Anchored Association)
- Run C: Variant C (Birth-Anchored Tracking + Birth-Anchored Association)

Generates:
- tracking_association_comparison.csv
- recovery_segmentation_comparison.csv
- external_validation_comparison.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from cme_pipeline.utils.io import load_json


def compute_comparison_tables(repo_root: Path, output_dir: Path) -> Dict[str, pd.DataFrame]:
    runs_dir = repo_root / "runs"
    run_a_dir = runs_dir / "baseline_rule_b"
    run_b_dir = runs_dir / "variant_b_association_only"
    run_c_dir = runs_dir / "variant_c_full_birth_anchored"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Tracking & Association Comparison Table
    # ---------------------------------------------------------
    # Load manifests & summaries
    man_a = load_json(run_a_dir / "manifest.json")
    man_b = load_json(run_b_dir / "manifest.json")
    man_c = load_json(run_c_dir / "manifest.json")

    fam_a = pd.read_csv(run_a_dir / "tables/04_full_day_family_summary.csv")
    fam_b = pd.read_csv(run_b_dir / "tables/04_full_day_family_summary.csv")
    fam_c = pd.read_csv(run_c_dir / "tables/04_full_day_family_summary.csv")

    edges_a = pd.read_csv(run_a_dir / "tables/03_full_day_parent_child_edges.csv")
    edges_b = pd.read_csv(run_b_dir / "tables/03_full_day_parent_child_edges.csv")
    edges_c = pd.read_csv(run_c_dir / "tables/03_full_day_parent_child_edges.csv")

    states_a = pd.read_csv(run_a_dir / "tables/02_full_day_e1_track_states.csv")
    states_b = pd.read_csv(run_b_dir / "tables/02_full_day_e1_track_states.csv")
    states_c = pd.read_csv(run_c_dir / "tables/02_full_day_e1_track_states.csv")

    # Birth direction references
    dir_b_file = run_b_dir / "tables/00_birth_direction_references.csv"
    dir_c_file = run_c_dir / "tables/00_birth_direction_references.csv"

    dir_b = pd.read_csv(dir_b_file) if dir_b_file.is_file() else pd.DataFrame()
    dir_c = pd.read_csv(dir_c_file) if dir_c_file.is_file() else pd.DataFrame()

    # Anchor counts
    anchors_a = states_a[states_a["is_solar_root_anchor"] == True]["track_id"].nunique()
    anchors_b = states_b[states_b["is_solar_root_anchor"] == True]["track_id"].nunique()
    anchors_c = states_c[states_c["is_solar_root_anchor"] == True]["track_id"].nunique()

    # Direction locking stats
    locked_b = (dir_b["status"] == "LOCKED").sum() if not dir_b.empty else 0
    prov_b = (dir_b["status"] == "PROVISIONAL").sum() if not dir_b.empty else 0

    locked_c = (dir_c["status"] == "LOCKED").sum() if not dir_c.empty else 0
    prov_c = (dir_c["status"] == "PROVISIONAL").sum() if not dir_c.empty else 0

    # Separations
    sep_b = (states_b["decision"] == "SEPARATED_INDEPENDENT_FAMILY").sum()
    sep_c = (states_c["decision"] == "SEPARATED_INDEPENDENT_FAMILY").sum()

    # Main CME Family
    adm_a = fam_a[fam_a["status"] == "ADMITTED"].sort_values("max_detected_patches", ascending=False)
    adm_b = fam_b[fam_b["status"] == "ADMITTED"].sort_values("max_detected_patches", ascending=False)
    adm_c = fam_c[fam_c["status"] == "ADMITTED"].sort_values("max_detected_patches", ascending=False)

    main_fam_a = adm_a.iloc[0]["family_id"] if not adm_a.empty else "N/A"
    main_members_a = adm_a.iloc[0]["member_count"] if not adm_a.empty else 0
    main_max_p_a = adm_a.iloc[0]["max_detected_patches"] if not adm_a.empty else 0

    main_fam_b = adm_b.iloc[0]["family_id"] if not adm_b.empty else "N/A"
    main_members_b = adm_b.iloc[0]["member_count"] if not adm_b.empty else 0
    main_max_p_b = adm_b.iloc[0]["max_detected_patches"] if not adm_b.empty else 0

    main_fam_c = adm_c.iloc[0]["family_id"] if not adm_c.empty else "N/A"
    main_members_c = adm_c.iloc[0]["member_count"] if not adm_c.empty else 0
    main_max_p_c = adm_c.iloc[0]["max_detected_patches"] if not adm_c.empty else 0

    tracking_assoc_data = [
        {"Metric / Attribute": "Tracking Stage Algorithm", "Run A (Baseline Rule B)": "E1 Standard (Unconstrained)", "Run B (Variant B)": "E1 Standard (Unconstrained)", "Run C (Variant C)": "Birth-Anchored Directional (Locked PA)"},
        {"Metric / Attribute": "Association Stage Algorithm", "Run A (Baseline Rule B)": "Pure Direct Overlap (DO > 0.05)", "Run B (Variant B)": "Birth-Anchored DO (Delta CPA <= 45 deg)", "Run C (Variant C)": "Birth-Anchored DO (Delta CPA <= 45 deg)"},
        {"Metric / Attribute": "Observation Pairs", "Run A (Baseline Rule B)": "32", "Run B (Variant B)": "32", "Run C (Variant C)": "32"},
        {"Metric / Attribute": "Total Gate OFF Tracks", "Run A (Baseline Rule B)": f"{states_a['track_id'].nunique()}", "Run B (Variant B)": f"{states_b['track_id'].nunique()}", "Run C (Variant C)": f"{states_c['track_id'].nunique()}"},
        {"Metric / Attribute": "Solar-Root Anchors (r_min <= 30px)", "Run A (Baseline Rule B)": f"{anchors_a}", "Run B (Variant B)": f"{anchors_b}", "Run C (Variant C)": f"{anchors_c}"},
        {"Metric / Attribute": "Direction References (Locked / Provisional)", "Run A (Baseline Rule B)": "N/A (Unanchored)", "Run B (Variant B)": f"{locked_b} / {prov_b}", "Run C (Variant C)": f"{locked_c} / {prov_c}"},
        {"Metric / Attribute": "Candidate Families Spawned", "Run A (Baseline Rule B)": f"{len(fam_a)}", "Run B (Variant B)": f"{len(fam_b)}", "Run C (Variant C)": f"{len(fam_c)}"},
        {"Metric / Attribute": "Directional Anchor Separations", "Run A (Baseline Rule B)": "0 (Pure Overlap Merged)", "Run B (Variant B)": f"{sep_b}", "Run C (Variant C)": f"{sep_c}"},
        {"Metric / Attribute": "Admitted Families (Birth + Size Gate)", "Run A (Baseline Rule B)": f"{len(adm_a)}", "Run B (Variant B)": f"{len(adm_b)}", "Run C (Variant C)": f"{len(adm_c)}"},
        {"Metric / Attribute": "Admitted Member Tracks", "Run A (Baseline Rule B)": f"{adm_a['member_count'].sum()}", "Run B (Variant B)": f"{adm_b['member_count'].sum()}", "Run C (Variant C)": f"{adm_c['member_count'].sum()}"},
        {"Metric / Attribute": "Primary CME Family ID", "Run A (Baseline Rule B)": f"{main_fam_a}", "Run B (Variant B)": f"{main_fam_b}", "Run C (Variant C)": f"{main_fam_c}"},
        {"Metric / Attribute": "Primary CME Member Tracks", "Run A (Baseline Rule B)": f"{main_members_a}", "Run B (Variant B)": f"{main_members_b}", "Run C (Variant C)": f"{main_members_c}"},
        {"Metric / Attribute": "Primary CME Max Detected Patches", "Run A (Baseline Rule B)": f"{main_max_p_a}", "Run B (Variant B)": f"{main_max_p_b}", "Run C (Variant C)": f"{main_max_p_c}"},
    ]
    df_tracking_assoc = pd.DataFrame(tracking_assoc_data)
    df_tracking_assoc.to_csv(output_dir / "tracking_association_comparison.csv", index=False)

    # ---------------------------------------------------------
    # 2. Recovery & Segmentation Comparison Table
    # ---------------------------------------------------------
    rec_a = np.load(run_a_dir / "arrays/recovered_patch_masks_pipeline_b_full_day.npy")
    rec_b = np.load(run_b_dir / "arrays/recovered_patch_masks_pipeline_b_full_day.npy")
    rec_c = np.load(run_c_dir / "arrays/recovered_patch_masks_pipeline_b_full_day.npy")

    seg_a = np.load(run_a_dir / "arrays/final_pixel_segmentation_union.npy")
    seg_b = np.load(run_b_dir / "arrays/final_pixel_segmentation_union.npy")
    seg_c = np.load(run_c_dir / "arrays/final_pixel_segmentation_union.npy")

    sum_rec_a = int(rec_a.sum())
    sum_rec_b = int(rec_b.sum())
    sum_rec_c = int(rec_c.sum())

    sum_seg_a = int(seg_a.sum())
    sum_seg_b = int(seg_b.sum())
    sum_seg_c = int(seg_c.sum())

    # Active frames (where seg > 0)
    act_a = int((seg_a.sum(axis=(1, 2)) > 0).sum())
    act_b = int((seg_b.sum(axis=(1, 2)) > 0).sum())
    act_c = int((seg_c.sum(axis=(1, 2)) > 0).sum())

    # IoU comparisons
    iou_ab = float(np.logical_and(seg_a, seg_b).sum() / max(1, np.logical_or(seg_a, seg_b).sum()))
    iou_ac = float(np.logical_and(seg_a, seg_c).sum() / max(1, np.logical_or(seg_a, seg_c).sum()))
    iou_bc = float(np.logical_and(seg_b, seg_c).sum() / max(1, np.logical_or(seg_b, seg_c).sum()))

    rec_seg_data = [
        {"Metric": "Total Recovered 4x4 Patches", "Run A (Baseline Rule B)": f"{sum_rec_a:,}", "Run B (Variant B)": f"{sum_rec_b:,}", "Run C (Variant C)": f"{sum_rec_c:,}"},
        {"Metric": "Total Segmented Pixels", "Run A (Baseline Rule B)": f"{sum_seg_a:,}", "Run B (Variant B)": f"{sum_seg_b:,}", "Run C (Variant C)": f"{sum_seg_c:,}"},
        {"Metric": "Active CME Frames", "Run A (Baseline Rule B)": f"{act_a} / 32", "Run B (Variant B)": f"{act_b} / 32", "Run C (Variant C)": f"{act_c} / 32"},
        {"Metric": "Mean Pixels / Active Frame", "Run A (Baseline Rule B)": f"{sum_seg_a / max(1, act_a):.1f}", "Run B (Variant B)": f"{sum_seg_b / max(1, act_b):.1f}", "Run C (Variant C)": f"{sum_seg_c / max(1, act_c):.1f}"},
        {"Metric": "Peak Frame Segmented Pixels", "Run A (Baseline Rule B)": f"{int(seg_a.sum(axis=(1, 2)).max()):,}", "Run B (Variant B)": f"{int(seg_b.sum(axis=(1, 2)).max()):,}", "Run C (Variant C)": f"{int(seg_c.sum(axis=(1, 2)).max()):,}"},
        {"Metric": "Peak Frame Index", "Run A (Baseline Rule B)": f"Pair {29 + int(seg_a.sum(axis=(1, 2)).argmax())}", "Run B (Variant B)": f"Pair {29 + int(seg_b.sum(axis=(1, 2)).argmax())}", "Run C (Variant C)": f"Pair {29 + int(seg_c.sum(axis=(1, 2)).argmax())}"},
        {"Metric": "Pixel IoU with Baseline Run A", "Run A (Baseline Rule B)": "1.0000 (Reference)", "Run B (Variant B)": f"{iou_ab:.4f}", "Run C (Variant C)": f"{iou_ac:.4f}"},
        {"Metric": "Pixel IoU (Variant B vs Variant C)", "Run A (Baseline Rule B)": "-", "Run B (Variant B)": "-", "Run C (Variant C)": f"{iou_bc:.4f}"},
    ]
    df_rec_seg = pd.DataFrame(rec_seg_data)
    df_rec_seg.to_csv(output_dir / "recovery_segmentation_comparison.csv", index=False)

    # ---------------------------------------------------------
    # 3. External Validation Comparison Table
    # ---------------------------------------------------------
    val_a = pd.read_csv(run_a_dir / "tables/08_comparison_external_validation_metrics.csv")
    val_b = pd.read_csv(run_b_dir / "tables/08_comparison_external_validation_metrics.csv")
    val_c = pd.read_csv(run_c_dir / "tables/08_comparison_external_validation_metrics.csv")

    ext_val_rows = []
    for i in range(len(val_a)):
        row_a = val_a.iloc[i]
        ref = row_a["Reference"]
        row_b = val_b[val_b["Reference"] == ref].iloc[0]
        row_c = val_c[val_c["Reference"] == ref].iloc[0]

        iou_a = float(row_a["Angular IoU"])
        iou_b = float(row_b["Angular IoU"])
        iou_c = float(row_c["Angular IoU"])

        delta_iou_ca = iou_c - iou_a
        pct_iou_ca = (delta_iou_ca / iou_a) * 100.0 if iou_a > 0 else 0.0

        r_err_a = row_a["Radial Error"]
        r_err_b = row_b["Radial Error"]
        r_err_c = row_c["Radial Error"]

        # Parse numeric radial error if present
        if "R_sun" in str(r_err_a) and "R_sun" in str(r_err_c):
            val_r_a = float(str(r_err_a).replace(" R_sun", ""))
            val_r_c = float(str(r_err_c).replace(" R_sun", ""))
            delta_r = val_r_c - val_r_a
            pct_r = (delta_r / val_r_a) * 100.0
            r_diff_str = f"{delta_r:+.4f} R_sun ({pct_r:+.1f}%)"
        else:
            r_diff_str = "N/A"

        ext_val_rows.append({
            "Reference Catalogue": ref,
            "Evaluable Frames": int(row_a["Evaluable Frames"]),
            "Run A Angular IoU": f"{iou_a:.4f}",
            "Run B Angular IoU": f"{iou_b:.4f}",
            "Run C Angular IoU": f"{iou_c:.4f}",
            "IoU Delta (C - A)": f"{delta_iou_ca:+.4f} ({pct_iou_ca:+.2f}%)",
            "Run A Radial Error": str(r_err_a),
            "Run B Radial Error": str(r_err_b),
            "Run C Radial Error": str(r_err_c),
            "Radial Error Delta (C - A)": r_diff_str,
        })

    df_ext_val = pd.DataFrame(ext_val_rows)
    df_ext_val.to_csv(output_dir / "external_validation_comparison.csv", index=False)

    return {
        "tracking_association": df_tracking_assoc,
        "recovery_segmentation": df_rec_seg,
        "external_validation": df_ext_val,
    }


if __name__ == "__main__":
    repo = Path("/home/aababc/stereo_cor2")
    out = repo / "runs/comparison"
    results = compute_comparison_tables(repo, out)

    print("==========================================================================================")
    print("1. TRACKING & ASSOCIATION COMPARISON")
    print("==========================================================================================")
    print(results["tracking_association"].to_string(index=False))
    print("\n==========================================================================================")
    print("2. RECOVERY & SEGMENTATION COMPARISON")
    print("==========================================================================================")
    print(results["recovery_segmentation"].to_string(index=False))
    print("\n==========================================================================================")
    print("3. EXTERNAL VALIDATION COMPARISON")
    print("==========================================================================================")
    print(results["external_validation"].to_string(index=False))
    print("==========================================================================================")

