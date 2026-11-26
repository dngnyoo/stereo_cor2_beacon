from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import pandas as pd

qc_csv = str(ROOT / "runs/shared_d4_20101212/02_detection/data/qc/frame_detection_summary.csv")
df = pd.read_csv(qc_csv)

print("Checking ALL 63 rows in frame_detection_summary.csv:")
for idx, r in df.iterrows():
    p_idx = int(r["pair_index"])
    time_str = str(r["end_time"])
    status = str(r["status"])
    t24 = float(r["T24_p04"]) if pd.notnull(r["T24_p04"]) else None
    floor = float(r["floor_p04"]) if pd.notnull(r["floor_p04"]) else None
    eff = float(r["effective_threshold_p04"]) if pd.notnull(r["effective_threshold_p04"]) else None

    if eff is not None and eff <= 0.50:
        print(f"Row {idx}: Pair {p_idx} [{time_str}] Status={status} Eff={eff}")

print("Done checking all rows.")
