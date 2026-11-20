import tempfile
from pathlib import Path
import pandas as pd
from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.pipeline import Pipeline

with tempfile.TemporaryDirectory() as tmp_dir:
    for method in ["a5b", "a5b_n", "a5b_nr1", "a5b_nr2"]:
        cfg = PipelineConfig.from_yaml(f"configs/tracking_a5b_identity.yaml")
        cfg.tracking.method = method
        cfg.output_base_dir = tmp_dir
        cfg.run_name = f"eval_{method}"
        cfg.force_rerun = True
        res = Pipeline(cfg).run()
        val_out = res["final_output"]
        seg_out = val_out.segmentation_output
        track_out = res["stage_outputs"]["tracking"]
        cat_df = val_out.comparison_metrics_df
        print(f"\n=== Method: {method} ===")
        print(f"Total tracks: {track_out.total_tracks}, Recovered patches: {seg_out.size_gate_output.total_recovered_patches}, Pixels: {seg_out.total_segmented_pixels}")
        for _, r in cat_df.iterrows():
            print(f"  {r['Reference']:20s}: Angular IoU = {r['Angular IoU']:.4f}")
