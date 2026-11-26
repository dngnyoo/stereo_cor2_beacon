import numpy as np
import hashlib
from pathlib import Path
import pandas as pd

def hash_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

ref_dir = Path("runs/shared_d4_20101212/02_detection/data/qc")
c01_dir = Path("runs/exp_20101212_c01_a/02_detection/data/qc")

print(f"Comparing {ref_dir} vs {c01_dir}")
for fname in ["patch_state_p04.npy", "patch_component_labels.npy"]:
    ref_f = ref_dir / fname
    c01_f = c01_dir / fname
    ref_h = hash_file(ref_f)
    c01_h = hash_file(c01_f)
    ref_arr = np.load(ref_f)
    c01_arr = np.load(c01_f)
    equal = np.array_equal(ref_arr, c01_arr)
    print(f"{fname}: equal={equal}, sha={ref_h}")

ref_raw = Path("runs/shared_d4_20101212/02_detection/data/raw/frame_detection_summary.csv")
c01_raw = Path("runs/exp_20101212_c01_a/02_detection/data/raw/frame_detection_summary.csv")

df_ref = pd.read_csv(ref_raw)
df_c01 = pd.read_csv(c01_raw)

val_ref = df_ref[df_ref["status"] == "VALID"]
val_c01 = df_c01[df_c01["status"] == "VALID"]

print(f"VALID rows: {len(val_ref)} ref vs {len(val_c01)} c01")
print(f"  n_active_p04 equal in VALID: {(val_ref['n_active_p04'].values == val_c01['n_active_p04'].values).all()}")
print(f"  effective_threshold_p04 equal in VALID: {np.allclose(val_ref['effective_threshold_p04'].values, val_c01['effective_threshold_p04'].values)}")

# What about non-valid rows?
diff_cols = []
for c in df_ref.columns:
    s_ref = df_ref[c].fillna(-999).to_numpy()
    s_c01 = df_c01[c].fillna(-999).to_numpy()
    if not np.array_equal(s_ref, s_c01):
        diff_cols.append(c)

print(f"Columns with differences: {diff_cols}")
for c in diff_cols:
    diff_idx = df_ref.index[df_ref[c].fillna(-999) != df_c01[c].fillna(-999)].tolist()
    print(f"  Col '{c}' diff at indices: {diff_idx[:5]}")
    for idx in diff_idx[:3]:
        print(f"    idx={idx}: ref={df_ref.loc[idx, c]} vs c01={df_c01.loc[idx, c]}")
