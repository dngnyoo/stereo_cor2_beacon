import pandas as pd
import numpy as np

qc_csv = "/home/aababc/stereo_cor2_a5bn/runs/shared_d4_20101212/02_detection/data/qc/frame_detection_summary.csv"
raw_csv = "/home/aababc/stereo_cor2_a5bn/runs/shared_d4_20101212/02_detection/data/raw/frame_detection_summary.csv"

df = pd.read_csv(qc_csv)
val = df[df["status"] == "VALID"]
print(f"Total valid frames in run: {len(val)}")
print(f"Min T24_p04: {val['T24_p04'].min():.4f}")
print(f"Max T24_p04: {val['T24_p04'].max():.4f}")
print(f"Min outer_p95 (floor_p04): {val['floor_p04'].min():.4f}")
print(f"Max outer_p95 (floor_p04): {val['floor_p04'].max():.4f}")
print(f"Min effective threshold: {val['effective_threshold_p04'].min():.4f}")
print(f"Max effective threshold: {val['effective_threshold_p04'].max():.4f}")

print("\nChecking if absolute floor 0.50 was ever bound in any frame:")
floor_active_rows = val[val["effective_threshold_p04"] <= 0.50]
print(f"Frames where effective_threshold <= 0.50: {len(floor_active_rows)}")

margin_to_floor = val["effective_threshold_p04"] - 0.50
print(f"Minimum margin above 0.50 floor: {margin_to_floor.min():.4f} (at pair {val.loc[margin_to_floor.idxmin(), 'pair_index']})")
