# STEREO-A COR2 CME Tracking and Segmentation Pipeline

A modular, reproducible research pipeline for tracking and segmenting **coronal mass ejection (CME) candidates** in STEREO-A COR2 observations, producing pixel-level masks, an independent candidate-event catalog, and comparisons against external CME catalogs.

**Current scope:** the runnable benchmark starts from **precomputed hierarchical detection results**, not raw FITS. Raw-image preprocessing and fully integrated detection are under development.

> **Catalog status:** independent CME candidate catalog export is available on the `feature/independent-cme-catalog` branch. The default `main` branch contains the validated seven-stage tracking-to-validation baseline but does not yet include this catalog export.

## Quick start: reproduce the benchmark and generate outputs

The repository includes the selected input files for the **2010-12-12 STEREO-A COR2 frozen benchmark** (32 observation pairs); no separate dataset download is required for this benchmark.

Tested pipeline environment: Linux, Python 3.12. Installation from scratch in a clean virtual environment has not yet been independently verified.

```bash
git clone https://github.com/dngnyoo/stereo_cor2_beacon.git
cd stereo_cor2_beacon
git switch feature/independent-cme-catalog

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check

python -m cme_pipeline run --config configs/baseline_rule_b.yaml --force
```

Run these commands **from the repository root**. On Windows, activate the virtual environment with `.venv\Scripts\activate` (or the PowerShell equivalent); Windows execution has not been verified.

The `--force` option recomputes all stages. Omit it to reuse valid stage caches on later runs.

### Pipeline

```text
Frozen Detection Input (precomputed masks + RD frames)
  -> Tracking (E1 / A3 / A4 / A5b)
  -> Signed Dynamic PA Recovery
  -> Family Association
  -> Birth Radius Gate
  -> Size Gate
  -> Island-First Pixel Segmentation
       -> Independent CME Candidate Catalog CSV export [feature branch]
  -> External Catalog Validation (SEEDS, CACTus, MVCC/CORSET, Göttingen)
```

The catalog is generated **without using external catalog labels or validation results**. It describes the pipeline's candidate families, not independently confirmed physical CME events.

## Output files

For the baseline configuration, outputs are written to `runs/baseline_rule_b/`.

| Path | Description |
| --- | --- |
| `tables/cme_event_catalog.csv` | One row per admitted family with positive segmented pixels: observed time span, central position angle (CPA), angular width, projected front height, exploratory speed estimate, quality flags |
| `tables/cme_frame_measurements.csv` | Per-family, per-observed-frame mask area, CPA, angular width, projected front height |
| `arrays/final_pixel_segmentation_union.npy` | Boolean union mask, shape `(32, 256, 256)` for this benchmark |
| `arrays/family_pixel_masks.npz` | Separate Boolean masks for admitted families; these may overlap |
| `stages/segmentation/framewise_pixel_summary.csv` | Frame-level segmentation statistics |
| `stages/segmentation/family_pixel_statistics.csv` | Family-level segmentation statistics |
| `report.md` | Pipeline run summary and external validation metrics |
| `manifest.json` | Configuration, code revision, algorithm metadata, and source paths |
| `pipeline.log` | Execution and cache logs |

The two `cme_*.csv` outputs require the **`feature/independent-cme-catalog`** branch. They are currently written as a side effect of executing the Segmentation Stage; a Segmentation cache hit does **not** re-create deleted CSV files. To regenerate them using existing upstream caches:

```bash
python -m cme_pipeline run \
  --config configs/baseline_rule_b.yaml \
  --from-stage segmentation
```

This also reruns downstream Validation. The catalog is not yet a separate registered pipeline stage.

## Expected frozen benchmark results

The following values were reproduced from a **fresh GitHub clone** on November 11, 2026, using an already-existing Python environment:

| Metric | Expected value |
| --- | ---: |
| Observation pairs | 32 |
| Tracks (E1 baseline) | 219 |
| Candidate families | 185 |
| Admitted families | 15 |
| Admitted member tracks | 37 |
| Recovered patches (cumulative) | 5,174 |
| Segmented union pixels (cumulative) | 50,665 |
| SEEDS angular IoU | 0.5242 |
| CACTus angular IoU (all CMEs) | 0.2715 |
| CACTus angular IoU (CME 0024) | 0.7414 |
| MVCC/CORSET angular IoU | 0.4032 |
| Göttingen angular IoU | 0.5547 |

These IoUs refer to catalog-specific **angular comparisons**, not general pixelwise segmentation IoU. Evaluation samples and aggregation may differ between reference catalogs.

The benchmark's reference data and precomputed detection masks are under `validation_20101212/`. Files in that directory are tracked selectively; the rest of the original large research datasets are not included.

### Validate the catalog feature

```bash
python -m unittest discover -s tests -p 'test_independent_cme_catalog.py' -v
ls -lh runs/baseline_rule_b/tables/cme_*.csv
```

Four synthetic-data unit tests for the catalog passed in the development environment. The full original baseline was also reproduced from a fresh clone. A full clean-environment dependency installation and full-suite test of this feature branch are still recommended.

## Change the tracking algorithm

The default `configs/baseline_rule_b.yaml` runs the **E1** baseline with Rule B family association. To run **A5b** with identity association:

```bash
python -m cme_pipeline run --config configs/tracking_a5b_identity.yaml
```

Other tracking implementations (A3, A4) are available through the YAML configuration and algorithm registry. Different methods can yield different candidate families and masks, so the benchmark numbers above apply specifically to the E1 configuration.

Inspect cache status:

```bash
python -m cme_pipeline status --config configs/baseline_rule_b.yaml
```

Restart from a specific stage (reusing preceding stage caches):

```bash
python -m cme_pipeline run \
  --config configs/baseline_rule_b.yaml \
  --from-stage association
```

## Interpreting the independent candidate catalog

- **Event ID / family ID:** internal algorithm-generated identifier; one family does not necessarily equal one physically distinct CME.
- **First/last observed times:** first and last frames with positive segmentation pixels, **not** independently estimated CME initiation or launch times.
- **CPA and angular width:** computed from the family mask in the **largest segmented frame**. Broad or merged masks can distort these values.
- **Maximum front height:** maximum radial extent of a segmented pixel across observed frames, in projected solar radii (`R_sun`).
- **Plane-of-sky speed:** exploratory linear regression of the maximum segmented-mask radius against time when at least three distinct observed timestamps exist; not a deprojected 3D speed or necessarily a reliably tracked leading-front speed.
- **Speed quality:** `EXPLORATORY` and `LOW_CONFIDENCE` are heuristic flags, **not** external scientific validation.
- **Family mask pixel totals:** can exceed the union-mask total because multiple families' masks may overlap.

Check the per-frame measurements and segmentation overlays before interpreting catalog values as physical CME properties.

## Repository layout

```text
cme_pipeline/
  __main__.py             CLI entry point (run, status)
  core/                   Configuration, orchestration, cache, typed stage outputs
  stages/                 Tracking, recovery, association, gates, segmentation, validation
  experiments/            Comparison and historical experiment utilities
  TRACKING_ARCHITECTURE.md
configs/                  Baseline and alternative algorithm configurations
tests/                    Automated tests
validation_20101212/      Selected frozen benchmark input and external-reference files
runs/                     Generated results (ignored by Git)
```

## Documentation

- [Pipeline execution and developer guide](cme_pipeline/README.md): stage contracts, cache behavior, configuration, experimentation, and catalog outputs.
- [Unified Tracking Architecture and Extension Guide](cme_pipeline/TRACKING_ARCHITECTURE.md): standard tracking interfaces, component/track separation, A3/A4/A5b, and extensibility.

## Limitations and ongoing work

1. The current reproducible seven-stage benchmark **loads frozen detection masks**. It is not yet a general raw-FITS-to-catalog execution path.
2. An experimental QC-first hierarchical detector is being developed separately and is **not** integrated into this benchmark pipeline.
3. The external-validation stage currently depends on the included **2010-12-12** reference data and is not yet a date-agnostic validation service.
4. The independent catalog provides **candidate-level measurements**. Family identity, front-height continuity, width, and speed need further physical and catalog-based validation.
5. Some historical scripts and references to broader local datasets may exist elsewhere in the repository; the quick start above is the supported reproduction path for the frozen benchmark.
6. The currently pinned environment has not yet been validated through an entirely fresh dependency installation on each target operating system.

## Legacy COR2 preprocessing workflows

The repository also contains historical research workflows for Beacon and science-image preprocessing, including Z-score running difference, CMAD, and GCS-oriented processing. These are **separate from the frozen benchmark CLI** described above. Their presence does not mean that raw FITS ingestion is already connected to the modular `cme_pipeline` stages.
