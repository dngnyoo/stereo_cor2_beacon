from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import pandas as pd

states_csv = str(ROOT / "runs/exp_20101212_a5b_nr1/03_tracking/data/tables/live_tracking_states.csv")
df = pd.read_csv(states_csv)
print("States distribution in reference NR1 run:")
print(df["state"].value_counts())
print("\nTracks summary:")
for tid in sorted(df["track_id"].unique()):
    sub = df[df["track_id"] == tid]
    print(f"Track {tid}: total={len(sub)}, observed={(sub['state']=='OBSERVED').sum()}, predicted={(sub['state']=='PREDICTED_ONLY').sum()}")
