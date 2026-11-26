from pathlib import Path
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv(Path(__file__).resolve().parent / "experiments/ret_tracking/results/tracks_RET2_MotionPred.csv")
for tid in [42, 49]:
    sub = df[df['track_id'] == tid]
    print(f"=== Track {tid} ===")
    print(sub.to_string())
