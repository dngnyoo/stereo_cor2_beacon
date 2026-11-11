# STEREO-A COR2 CME Tracking and Segmentation Pipeline

A reproducible research pipeline for tracking **coronal mass ejection (CME) candidates** in STEREO-A COR2 data, recovering candidate structures, building families, producing pixel-level segmentation masks, and comparing results with independent CME catalogs.

**Start here:** [QUICK_START.md](QUICK_START.md) — clone, install, run, inspect outputs, and verify the frozen benchmark.

## Available versions

| Branch | Pipeline | What it produces |
| --- | --- | --- |
| **`main` (default)** | 7 stages: tracking → recovery → association → birth gate → size gate → segmentation → validation | Tracking and segmentation arrays, summary tables, run report, external-catalog comparisons |
| [`feature/independent-cme-catalog`](https://github.com/dngnyoo/stereo_cor2_beacon/tree/feature/independent-cme-catalog) | 8 stages: adds **CME Products** between segmentation and validation | All baseline outputs **plus** independent candidate catalog CSVs, family-labeled scientific PNGs, and GIF |

> The CME Products feature is on a **development branch**. Its code and documentation are available, but it has **not yet been merged into `main` or fully verified end-to-end in a fresh environment**. The seven-stage baseline has been reproduced from a fresh GitHub clone.

## Quick start: baseline on `main`

Run the following from a Linux shell with Python 3.12 available:

```bash
git clone https://github.com/dngnyoo/stereo_cor2_beacon.git
cd stereo_cor2_beacon

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m cme_pipeline run --config configs/baseline_rule_b.yaml --force
```

The selected 2010-12-12 frozen benchmark inputs are included in the repository; there is **no separate raw-image download step** for this reproducible benchmark.

Outputs are written to `runs/baseline_rule_b/`. Run configuration: [`configs/baseline_rule_b.yaml`](configs/baseline_rule_b.yaml).

To generate **candidate event catalogs and visualizations**, check out the feature branch **before installing and running**:

```bash
git switch feature/independent-cme-catalog
python -m cme_pipeline run --config configs/baseline_rule_b.yaml --force
```

For precise output paths, test commands, cache regeneration, and troubleshooting, follow [QUICK_START.md](QUICK_START.md).

## Pipeline overview

```text
Frozen COR2 detection inputs
   |
   v
1. Tracking (E1 baseline; alternative methods supported)
   |
2. Signed dynamic PA recovery
   |
3. Family association (Rule B baseline)
   |
4. Family birth-radius gate
   |
5. Family size gate
   |
6. Island-first pixel segmentation
   |
   +-- [feature branch only] 7. CME Products: catalog + visualization
   |
7. External catalog validation (8 on the feature branch)
```

**Important scope restriction:** this is a reproducible **frozen-detection** benchmark. It uses precomputed detection / running-difference inputs, not a fully integrated pipeline from arbitrary raw FITS files.

## Reproduced baseline results

The following metrics were obtained in a fresh clone of the `main` branch on **2026-11-11**, using an existing Python virtual environment:

| Quantity | Result |
| --- | ---: |
| Observation pairs | 32 |
| E1 tracks | 219 |
| Candidate families | 185 |
| Admitted families | 15 |
| Admitted member tracks | 37 |
| Cumulative recovered patches | 5,174 |
| Cumulative union-mask pixels | 50,665 |
| SEEDS angular IoU | 0.5242 |
| CACTus (all CME entries) angular IoU | 0.2715 |
| CACTus (CME 0024) angular IoU | 0.7414 |
| MVCC/CORSET angular IoU | 0.4032 |
| Göttingen angular IoU | 0.5547 |

These are **angular reference-catalog comparison metrics**, not general pixelwise IoUs. A fresh environment installation and the updated eight-stage feature still require independent verification.

## Principal outputs

On **`main`**, see:

- `runs/baseline_rule_b/arrays/final_pixel_segmentation_union.npy` — benchmark union pixel masks, shape `(32, 256, 256)`.
- `runs/baseline_rule_b/arrays/family_pixel_masks.npz` — candidate Family-specific masks (possible overlap).
- `runs/baseline_rule_b/stages/segmentation/` — frame-level and Family-level segmentation statistics.
- `runs/baseline_rule_b/report.md` — run summary and validation comparison.
- `runs/baseline_rule_b/manifest.json` and `pipeline.log` — run provenance and execution messages.

On **`feature/independent-cme-catalog`**, the additional CME Products Stage generates:

- `tables/cme_event_catalog.csv` and `tables/cme_frame_measurements.csv`
- `visualizations/segmentation_overlays/pair_XXX_YYYYMMDDTHHMMSSZ_families.png`
- `visualizations/family_tracking/` — per-Family peak observation images
- `visualizations/height_time/` — exploratory height-versus-time graphs
- `visualizations/cme_animation.gif`

For the 2010 benchmark, the scientific overlays use north-up COR2 coordinates, fixed RD display `[-600, +600]` DN with `coolwarm`, black invalid pixels, solar reference geometry, **admitted Family labels** (`F19` for `FAM_T19`), and actual Pair/UTC annotations.

**Catalog interpretation:** an admitted candidate Family is not necessarily one independently confirmed physical CME. Angular extent, leading-front proxy, and plane-of-sky speed are exploratory mask-based measurements and should be inspected against per-frame observations before scientific use.

## Documentation and configuration

- [QUICK_START.md](QUICK_START.md) — installation, both branches, complete output paths, verification and troubleshooting
- [`cme_pipeline/README.md`](cme_pipeline/README.md) — modular seven-stage baseline architecture and extension guide on `main`
- [Tracking architecture](cme_pipeline/TRACKING_ARCHITECTURE.md) — tracking contracts and variants
- [Baseline configuration](configs/baseline_rule_b.yaml)
- [A5b + identity comparison configuration](configs/tracking_a5b_identity.yaml)

To see cache status: `python -m cme_pipeline status --config configs/baseline_rule_b.yaml`.

To rerun the segmentation and later stages: `python -m cme_pipeline run --config configs/baseline_rule_b.yaml --from-stage segmentation`.

On the feature branch, CME Products can be regenerated using `--from-stage products` while reusing preceding stage caches.

## Additional research scripts

The repository retains historical Beacon/science preprocessing and experimental research scripts (including `download.py` and the four standalone preprocessing styles). These are **separate workflows** from the frozen benchmark CLI described above. Do not run `download.py` as a prerequisite for reproducing the included benchmark.

## Current limitations

- Frozen benchmark inputs are bundled, but not a general raw FITS → detection → catalog production workflow.
- The feature-branch eight-stage pipeline and its visualization output need end-to-end and fresh-environment verification before promotion to `main`.
- Independent Family catalogs are candidate-level outputs; physical CME identity, front continuity, speeds, and angular widths need additional validation.
- External validation currently uses reference material specific to the frozen 2010-12-12 benchmark.
