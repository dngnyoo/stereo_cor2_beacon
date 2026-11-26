import pandas as pd
manifest = pd.read_csv('/home/aababc/stereo_cor2_ret/runs/dual_archive_20101212/preprocessed/01_rd_manifest.csv')
for idx in range(34, 44):
    row = manifest.iloc[idx]
    print(f"frame_idx={idx} | pair_index={row['pair_index']} | rd_pair_id={row['rd_pair_id']} | end_time={row['end_time']}")
