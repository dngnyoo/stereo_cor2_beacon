# STEREO-A COR2 CME Tracking and Segmentation Pipeline

A reproducible research pipeline for tracking **coronal mass ejection (CME) candidates** in STEREO-A COR2 data, recovering candidate structures, building families, producing pixel-level segmentation masks, and comparing results with independent CME catalogs.

**Start here:** [QUICK_START.md](QUICK_START.md) — clone, install, run, inspect catalogs and visualizations, and verify the frozen benchmark.

## Current release

The default **`main` branch now contains the complete eight-stage pipeline**, including an independent candidate-event catalog and standardized scientific visualizations. The earlier seven-stage benchmark metrics remain the reference regression target. The previous development branch is retained for provenance, but is no longer required to produce catalogs or PNG/GIF output.

**Validation status:** the seven-stage benchmark was reproduced from a clean GitHub clone on 2026-11-11. The newly merged eight-stage `main` code still requires a fresh-clone end-to-end run and updated tests before its outputs can be considered regression-verified.

## Quick start: generate the full CME results on `main`

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

The default command now produces segmentation masks, independent candidate catalog CSVs, diagnostic PNG images, a GIF animation, and external validation results. No branch switch is needed.

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
7. CME Products: independent catalog + scientific visualization
   |
8. External catalog validation
```

**Important scope restriction:** this is a reproducible **frozen-detection** benchmark. It uses precomputed detection / running-difference inputs, not a fully integrated pipeline from arbitrary raw FITS files.

## Reproduced baseline results

The following metrics were obtained in a fresh clone of the previous seven-stage `main` branch on **2026-11-11**, using an existing Python virtual environment, on the previous seven-stage baseline (expected unchanged by the new Products stage):

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

These are **angular reference-catalog comparison metrics**, not general pixelwise IoUs. A fresh environment installation and the merged eight-stage pipeline still require independent verification.

## Principal outputs

The current **`main`** baseline produces:

- `runs/baseline_rule_b/arrays/final_pixel_segmentation_union.npy` — benchmark union pixel masks, shape `(32, 256, 256)`.
- `runs/baseline_rule_b/arrays/family_pixel_masks.npz` — candidate Family-specific masks (possible overlap).
- `runs/baseline_rule_b/stages/segmentation/` — frame-level and Family-level segmentation statistics.
- `runs/baseline_rule_b/report.md` — run summary and validation comparison.
- `runs/baseline_rule_b/manifest.json` and `pipeline.log` — run provenance and execution messages.

The **CME Products Stage** additionally generates:

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

CME Products can be regenerated on `main` using `--from-stage products` while reusing preceding stage caches.

## Additional research scripts

The repository retains historical Beacon/science preprocessing and experimental research scripts (including `download.py` and the four standalone preprocessing styles). These are **separate workflows** from the frozen benchmark CLI described above. Do not run `download.py` as a prerequisite for reproducing the included benchmark.

## Current limitations

- Frozen benchmark inputs are bundled, but not a general raw FITS → detection → catalog production workflow.
- The merged eight-stage pipeline and its visualization output still need end-to-end regression verification and a fresh-environment installation test.
- Independent Family catalogs are candidate-level outputs; physical CME identity, front continuity, speeds, and angular widths need additional validation.
- External validation currently uses reference material specific to the frozen 2010-12-12 benchmark.
