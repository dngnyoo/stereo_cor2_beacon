import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

b0_dir = (ROOT / "runs/exp_20101212_a5b_baseline")
b1_dir = (ROOT / "runs/exp_20101212_a5b_n")
nr1_dir = (ROOT / "runs/exp_20101212_a5b_nr1")
nr2_dir = (ROOT / "runs/exp_20101212_a5b_nr2")

# 1. Compare ownership masks
mask_b0 = np.load(b0_dir / "03_tracking/data/arrays/track_ownership_masks.npy")
mask_b1 = np.load(b1_dir / "03_tracking/data/arrays/track_ownership_masks.npy")
mask_nr1 = np.load(nr1_dir / "03_tracking/data/arrays/track_ownership_masks.npy")
mask_nr2 = np.load(nr2_dir / "03_tracking/data/arrays/track_ownership_masks.npy")

print("=== TRACK OWNERSHIP MASKS COMPARISON ===")
print("mask_b0 == mask_nr1 (bitwise):", np.array_equal(mask_b0, mask_nr1))
print("mask_b0 == mask_nr2 (bitwise):", np.array_equal(mask_b0, mask_nr2))
print("mask_nr1 == mask_nr2 (bitwise):", np.array_equal(mask_nr1, mask_nr2))

diff_nr1_nr2 = int((mask_nr1 != mask_nr2).sum())
print(f"Total patch differences between NR1 and NR2: {diff_nr1_nr2}")

# Per-frame differences
diff_frames = []
for f_idx in range(len(mask_nr1)):
    if not np.array_equal(mask_nr1[f_idx], mask_nr2[f_idx]):
        d_cnt = int((mask_nr1[f_idx] != mask_nr2[f_idx]).sum())
        diff_frames.append((f_idx, d_cnt))
        print(f"  Frame {f_idx} differs by {d_cnt} patches:")
        # Track counts in NR1 vs NR2 on this frame
        u_nr1, c_nr1 = np.unique(mask_nr1[f_idx], return_counts=True)
        u_nr2, c_nr2 = np.unique(mask_nr2[f_idx], return_counts=True)
        print(f"    NR1 tracks: {dict(zip(u_nr1, c_nr1))}")
        print(f"    NR2 tracks: {dict(zip(u_nr2, c_nr2))}")

# 2. Gate Results Comparison
print("\n=== GATE RESULTS COMPARISON ===")
for name, rdir in [("B0 (A5b)", b0_dir), ("B1 (A5b-N)", b1_dir), ("NR1 (A5b-NR1)", nr1_dir), ("NR2 (A5b-NR2)", nr2_dir)]:
    t_states = pd.read_csv(rdir / "03_tracking/data/tables/live_tracking_states.csv")
    man = pd.read_csv(rdir / "03_tracking/data/tables/live_observation_manifest.csv")
    eval_states = t_states[t_states["state"] == "OBSERVED"]
    total_tracks = t_states["track_id"].nunique()
    total_patches = mask_nr1.sum() if "NR1" in name else (mask_nr2.sum() if "NR2" in name else mask_b0.sum())
    # check gate tables
    gate_dir = rdir / "05_gates/data/tables"
    if (gate_dir / "03_size_gate_family_admissions.csv").is_file():
        df_adm = pd.read_csv(gate_dir / "03_size_gate_family_admissions.csv")
        n_adm = int(df_adm["admitted"].sum()) if "admitted" in df_adm.columns else len(df_adm)
    else:
        n_adm = "N/A"
    print(f"{name:15s}: Total tracks={total_tracks}, Admitted tracks={n_adm}")

# 3. Candidate Scores & Zero-Collapse
df_b1 = pd.read_csv(b1_dir / "03_tracking/data/tables/normalized_candidate_scores.csv")
df_nr1 = pd.read_csv(nr1_dir / "03_tracking/data/tables/normalized_candidate_scores.csv")
df_nr2 = pd.read_csv(nr2_dir / "03_tracking/data/tables/normalized_candidate_scores.csv")

print("\n=== ZERO-COLLAPSE AND SATURATION AUDIT ===")
print(f"B1:  Zero-collapse = {(df_b1['candidate_score'] == 0).sum()} / {len(df_b1)} ({(df_b1['candidate_score'] == 0).mean()*100:.1f}%), Overlap > 1 = {(df_b1['raw_overlap'] > 1.0).sum()}")
print(f"NR1: Zero-collapse = {(df_nr1['candidate_score'] == 0).sum()} / {len(df_nr1)} ({(df_nr1['candidate_score'] == 0).mean()*100:.1f}%), Overlap > 1 = {(df_nr1['C'] > 1.0).sum()}")
print(f"NR2: Zero-collapse = {(df_nr2['candidate_score'] == 0).sum()} / {len(df_nr2)} ({(df_nr2['candidate_score'] == 0).mean()*100:.1f}%), Overlap > 1 = {(df_nr2['C'] > 1.0).sum()}")
