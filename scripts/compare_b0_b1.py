#!/usr/bin/env python3
"""Controlled comparison analysis between B0 (A5b) and B1 (A5b-N)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

b0_dir = Path("runs/exp_20101212_a5b_baseline")
b1_dir = Path("runs/exp_20101212_a5b_n")

print("=" * 70)
print("STEREO_COR2 A5b vs A5b-N CONTROLLED EVALUATION (2010-12-12)")
print("=" * 70)

# 1. Manifest and metadata comparison
meta_b0 = json.loads((b0_dir / "03_tracking/data/tracking_metadata.json").read_text())
meta_b1 = json.loads((b1_dir / "03_tracking/data/tracking_metadata.json").read_text())
print(f"B0 method: {meta_b0['method']}, total tracks: {meta_b0['total_tracks']}")
print(f"B1 method: {meta_b1['method']}, total tracks: {meta_b1['total_tracks']}")

# 2. Gate outcomes
gates_b0 = json.loads((b0_dir / "05_gates/data/admitted_families.json").read_text())
gates_b1 = json.loads((b1_dir / "05_gates/data/admitted_families.json").read_text())
print(f"B0 Admitted Families: {gates_b0['family_ids']}, Admitted Tracks: {gates_b0['track_ids']}")
print(f"B1 Admitted Families: {gates_b1['family_ids']}, Admitted Tracks: {gates_b1['track_ids']}")

# 3. Ownership masks
own_b0 = np.load(b0_dir / "03_tracking/data/arrays/track_ownership_masks.npy")
own_b1 = np.load(b1_dir / "03_tracking/data/arrays/track_ownership_masks.npy")
print(f"Ownership shape: {own_b0.shape}")
mask_identical = np.array_equal(own_b0, own_b1)
print(f"Track ownership masks bit-for-bit identical: {mask_identical}")
if not mask_identical:
    diff_voxels = int(np.sum(own_b0 != own_b1))
    print(f"Total differing patch assignments across run: {diff_voxels}")
    for f in range(own_b0.shape[0]):
        f_diff = int(np.sum(own_b0[f] != own_b1[f]))
        if f_diff > 0:
            print(f"  Frame {f}: {f_diff} differing patches")

# 4. Tracking state comparison
df_b0 = pd.read_csv(b0_dir / "03_tracking/data/tables/live_tracking_states.csv")
df_b1 = pd.read_csv(b1_dir / "03_tracking/data/tables/live_tracking_states.csv")
print(f"\nB0 track states rows: {len(df_b0)}")
print(f"B1 track states rows: {len(df_b1)}")

tids = sorted(set(df_b0["track_id"]).union(set(df_b1["track_id"])))
print(f"{'Track':<6} | {'B0 Obs':<8} | {'B1 Obs':<8} | {'B0 Pairs':<18} | {'B1 Pairs':<18} | Status")
print("-" * 75)
for tid in tids:
    sub0 = df_b0[df_b0["track_id"] == tid]
    sub1 = df_b1[df_b1["track_id"] == tid]
    obs0 = len(sub0[sub0["state"] == "OBSERVED"])
    obs1 = len(sub1[sub1["state"] == "OBSERVED"])
    p0_str = f"{sub0.iloc[0]['pair_index']}-{sub0.iloc[-1]['pair_index']}" if len(sub0) > 0 else "N/A"
    p1_str = f"{sub1.iloc[0]['pair_index']}-{sub1.iloc[-1]['pair_index']}" if len(sub1) > 0 else "N/A"
    status = "Identical" if len(sub0) == len(sub1) and obs0 == obs1 else "DIFF"
    print(f"{tid:<6} | {obs0:<8} | {obs1:<8} | {p0_str:<18} | {p1_str:<18} | {status}")

# 5. Candidate Diagnostics
diag_csv = b1_dir / "03_tracking/data/tables/normalized_candidate_scores.csv"
if diag_csv.is_file():
    df_diag = pd.read_csv(diag_csv)
    print(f"\nCandidate Diagnostics: {len(df_diag)} total candidate evaluations")
    print(f"Columns: {list(df_diag.columns)}")
    print("\nScore Component Summary:")
    print(df_diag[["O", "A", "K", "F", "candidate_score", "raw_a5b_score"]].describe().to_string())

    print("\nOutcome Breakdown:")
    print(df_diag["selected_rejected_outcome"].value_counts().to_string())

    print("\nFinal Ownership Breakdown:")
    print(df_diag["final_ownership_outcome"].value_counts().to_string())

    print("\nRejection Reasons:")
    print(df_diag["rejection_reason"].value_counts().to_string())

    print("\nEligibility Categories:")
    print(df_diag["candidate_eligibility_category"].value_counts().to_string())

    # Check zero scores
    n_zeros = int((df_diag["candidate_score"] == 0.0).sum())
    print(f"\nCandidates with S_N == 0.0: {n_zeros} / {len(df_diag)} ({100.0 * n_zeros / len(df_diag):.1f}%)")
    zero_df = df_diag[df_diag["candidate_score"] == 0.0]
    print(f"  Causes of zero: O=0: {(zero_df['O'] == 0).sum()}, A=0: {(zero_df['A'] == 0).sum()}, K=0: {(zero_df['K'] == 0).sum()}, F=0: {(zero_df['F'] == 0).sum()}")
    print(f"  Selected despite S_N==0 (via overlap/corridor continuity): {(zero_df['selected_rejected_outcome'] == 'SELECTED').sum()}")

    # Check clipping saturation
    n_o_clipped = int((df_diag["raw_overlap"] > 1.0).sum())
    print(f"\nOverlap term saturation (raw_overlap > 1.0): {n_o_clipped} / {len(df_diag)} ({100.0 * n_o_clipped / len(df_diag):.1f}%)")

    # Check ranking changes
    # For frames with multiple candidates for the same track
    ranking_flips = 0
    multi_cand_tracks = 0
    for (f_idx, tid), grp in df_diag.groupby(["frame_index", "track_id"]):
        if len(grp) > 1:
            multi_cand_tracks += 1
            # Sort by A5b score vs A5b-N score
            order_a5b = grp.sort_values("raw_a5b_score", ascending=False)["component_id"].tolist()
            order_a5bn = grp.sort_values("candidate_score", ascending=False)["component_id"].tolist()
            if order_a5b != order_a5bn:
                ranking_flips += 1
                print(f"  Ranking flip at frame {f_idx}, track {tid}: A5b={order_a5b} vs A5b-N={order_a5bn}")
    print(f"Multi-candidate track-frames: {multi_cand_tracks}, Ranking flips: {ranking_flips}")
