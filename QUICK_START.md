# Quick Start — STEREO-A COR2 Frozen CME Benchmark

This guide is written for researchers who want to reproduce the **2010-12-12 frozen benchmark** and inspect all available outputs. The repository includes selected precomputed inputs required for this benchmark.

**Default branch: `main` — eight-stage CME Pipeline.** The repository now includes the independent CME Products Stage, which writes candidate-event catalogs and scientific PNG/GIF visualizations in addition to segmentation and external validation. No branch switch is needed.

**Validation note:** the original seven-stage baseline was reproduced in a fresh GitHub clone; the newly merged eight-stage code still needs a complete regression run and fresh-environment installation check.

## 1. Requirements and environment

Recommended test platform: Linux (including Ubuntu on WSL), Python 3.12, Git, and adequate space for the included benchmark, Python dependencies, caches, and image outputs.

The seven-stage benchmark was successfully executed on 2026-11-11 from a fresh GitHub clone using an **already-existing** Python environment. Installing the requirements in a totally fresh environment remains to be verified.

```bash
git clone https://github.com/dngnyoo/stereo_cor2_beacon.git
cd stereo_cor2_beacon

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

On Windows, virtual-environment activation differs; Linux/WSL is the supported research workflow. Do not run the historical `download.py` script for this benchmark: its required frozen inputs are included.

## 2. Execute the full benchmark

From the repository root:

```bash
python -m cme_pipeline run \
  --config configs/baseline_rule_b.yaml \
  --force
```

- `--force`: execute every stage from scratch and invalidate existing stage caches.
- Without `--force`: reuse valid cached stages.
- `--config`: select the run configuration and output directory.

The baseline configuration uses **E1 tracking** and **Rule B family association**. The eight-stage run writes results under:

```text
runs/baseline_rule_b/
  arrays/
  tables/
  stages/
  visualizations/
  report.md
  manifest.json
  pipeline.log
```

The `products` stage populates `visualizations/` with scientific PNG/GIF results. If it is empty, inspect `pipeline.log`, ensure `products.visualization: true`, and rerun with `--from-stage products`.

## 3. Verify benchmark results

A successful run should reproduce the following **E1 + Rule B, 2010-12-12** reference figures:

| Metric | Expected |
| --- | ---: |
| Observation pairs | 32 |
| Tracks | 219 |
| Candidate families | 185 |
| Admitted families | 15 |
| Admitted tracks | 37 |
| Cumulative recovered patches | 5,174 |
| Cumulative segmented union pixels | 50,665 |
| SEEDS angular IoU | 0.5242 |
| CACTus all-event angular IoU | 0.2715 |
| CACTus CME 0024 angular IoU | 0.7414 |
| MVCC/CORSET angular IoU | 0.4032 |
| Göttingen angular IoU | 0.5547 |

Check the pipeline log and generated report:

```bash
tail -n 35 runs/baseline_rule_b/pipeline.log
cat runs/baseline_rule_b/report.md
ls -lh runs/baseline_rule_b/arrays/
```

These validation metrics are **angular comparisons**, not universal pixel-mask IoU.

## 4. Segmentation and benchmark outputs on `main`

| Relative to `runs/baseline_rule_b/` | Purpose |
| --- | --- |
| `arrays/final_pixel_segmentation_union.npy` | Union segmentation mask, shape `(32, 256, 256)` |
| `arrays/family_pixel_masks.npz` | Per-Family segmentation masks |
| `stages/segmentation/framewise_pixel_summary.csv` | Framewise pixel statistics |
| `stages/segmentation/family_pixel_statistics.csv` | Per-Family pixel statistics |
| `tables/06_full_day_framewise_pixel_summary.csv` | Consolidated framewise output |
| `tables/07_full_day_family_pixel_statistics.csv` | Consolidated Family statistics |
| `report.md` | Summary and external validation |
| `manifest.json` | Configuration and code provenance |
| `pipeline.log` | Stage-by-stage execution log |

Example inspection:

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

root = Path("runs/baseline_rule_b/arrays")
union = np.load(root / "final_pixel_segmentation_union.npy")
with np.load(root / "family_pixel_masks.npz") as families:
    print("Union shape:", union.shape)
    print("Cumulative union pixels:", int(union.sum()))
    print("Family mask arrays:", len(families.files))
PY
```

Family mask pixel counts may overlap; adding counts across families is not equivalent to counting union-mask pixels.

## 5. CME Products outputs on `main`

The **seventh stage**, `products`, automatically creates independent candidate catalog CSVs, labeled COR2 overlays, and a GIF. It runs after Segmentation and before External Validation.

Expected additional output locations **when the Products Stage completes successfully**:

```text
runs/baseline_rule_b/
  tables/
    cme_event_catalog.csv
    cme_frame_measurements.csv
  visualizations/
    segmentation_overlays/
      pair_032_20101212T062424Z_families.png
      ... (Pair/UTC filenames for other observations)
    family_tracking/
      FAM_T19_pair_..._peak.png
      ...
    height_time/
      FAM_T19_..._height_time.png
      ...
    cme_animation.gif
```

The date/time example above corresponds to Pair 032 at 2010-12-12 06:24:24 UTC; the actual generated files depend on observation and positive-mask availability.

The scientific benchmark visualization uses **fixed RD [-600,+600] DN, `coolwarm`, north-up, black invalid pixels, solar/occulter/FOV/birth-gate geometry, and accepted Family labels** (`F19` for `FAM_T19`). Predicted-only tracking candidates are not shown as admitted families.

### Recreate only the catalog and visuals

Once Tracking, Recovery, Association, Gates, and Segmentation have been cached:

```bash
python -m cme_pipeline run \
  --config configs/baseline_rule_b.yaml \
  --from-stage products
```

This re-executes Products and downstream Validation; upstream stages can be loaded from cache.

Catalog and visualization switches (in the current `main` YAML configuration):

```yaml
products:
  catalog: true
  visualization: true
  animation: true
```

You can disable rendering with `visualization: false` while preserving CSV generation.

### Check generated files

```bash
ls -lh runs/baseline_rule_b/tables/cme_*.csv
find runs/baseline_rule_b/visualizations -type f | sort
```

Run catalog and product-stage tests:

```bash
python -m unittest discover -s tests -p 'test_independent_cme_catalog.py' -v
```

The latest family-label and Pair/UTC naming changes have been merged into `main`, but **their full regression run in a fresh clone is not yet confirmed**.

## 6. Alternate tracking configuration

To run A5b tracking with identity Family association instead of the E1 + Rule B baseline:

```bash
python -m cme_pipeline run --config configs/tracking_a5b_identity.yaml
```

This produces a separate run and is not expected to reproduce E1 baseline numbers.

## 7. Troubleshooting and reproducibility notes

- **Empty `visualizations/`**: confirm `products.visualization: true`, review `pipeline.log`, and rerun `--from-stage products`.
- **Missing CSVs or PNGs**: confirm `git branch --show-current` is `main`; rerun with `--from-stage products`.
- **Cache surprises**: `python -m cme_pipeline status --config configs/baseline_rule_b.yaml` shows cached stages. Use `--force` for a complete recomputation.
- **Local virtual environment**: an external clone using an existing `.venv` does not prove a fresh requirements installation works.
- **NumPy `All-NaN slice encountered` warnings**: these occurred during the successful reproduced seven-stage run; investigate invalid/missing data handling if they recur.
- **Different run directories**: `configs/baseline_rule_b.yaml` determines `run_name`; do not assume all configurations write to the same path.
- **Physical interpretation**: Family IDs are algorithmic candidates, not externally verified CME identities. CPA, angular width and maximum-mask-radius-based speed can be misleading when families merge, masks fragment or an event reaches the FOV edge.

## 8. Further documentation

- [Main project README](README.md)
- [Pipeline architecture and developer guide](cme_pipeline/README.md)
- [Tracking implementation architecture](cme_pipeline/TRACKING_ARCHITECTURE.md)

### Reproduction scope

This quick start reproduces an **already prepared 2010-12-12 benchmark**. It does not establish a general workflow from arbitrary raw FITS files. The repo also contains older standalone preprocessing scripts, but they are not prerequisites of the modular benchmark.
