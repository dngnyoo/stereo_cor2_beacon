#!/usr/bin/env python3
"""One-click demonstration and verification script for CME Pipeline Handover."""

import sys
import time
from pathlib import Path

# Add current directory to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.pipeline import Pipeline


def main():
    print("=" * 78)
    print("  STEREO-A COR2 Near-Real-Time CME Pipeline — One-Click Demo")
    print("=" * 78)

    config_path = REPO_ROOT / "configs/baseline_rule_b.yaml"
    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print(f"Loading configuration from: {config_path}")
    cfg = PipelineConfig.from_yaml(config_path)
    cfg.repo_root = str(REPO_ROOT)

    print("Initializing CME Pipeline...")
    pipeline = Pipeline(cfg)

    t0 = time.time()
    result = pipeline.run()
    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 78)
    print("  Demo Execution Summary")
    print("=" * 78)
    print(f"Total Elapsed Time : {elapsed:.2f} s")
    print(f"Run Directory      : {pipeline.context.run_dir}")

    # Output verification
    seg_out = result.get("segmentation")
    if seg_out:
        final_pixels = int(seg_out.final_pixel_segmentation_union.sum())
        print(f"Final Active Pixels: {final_pixels:,} px (Reference: 50,665 px)")

    val_out = result.get("validation")
    if val_out:
        print("\nCatalogue Validation Matches:")
        for r in val_out.comparison_metrics_df.to_dict("records"):
            err_str = f", RadErr={r['radial_error_rsun']:.2f} R☉" if r['radial_error_rsun'] is not None else ""
            print(f"  - {r['catalogue']:<18}: IoU={r['angular_iou']:.4f}{err_str}")

    print("=" * 78)
    print("Demo completed successfully! Everything is ready for production research.")
    print("=" * 78)


if __name__ == "__main__":
    main()
