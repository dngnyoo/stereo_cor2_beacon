# STEREO-A COR2 CME Tracking and Segmentation Pipeline

A reproducible research pipeline for tracking **coronal mass ejection (CME) candidates** in STEREO-A COR2 data, recovering candidate structures, building families, producing pixel-level segmentation masks, and comparing results with independent CME catalogs.

**Start here:** [QUICK_START.md](QUICK_START.md) — clone, install dependencies, download/reuse NASA FITS, and run the complete COR2-A raw-to-catalog pipeline.

## Current release and reproducibility

The integrated `run-data` command supports downloaded or local raw COR2 d7
FITS, North-up RD, and previously computed C0 detections. It independently
recomputes E1 tracks and executes Recovery, Association, Birth/Size Gates,
Segmentation, and CME Products. The original Frozen benchmark is retained.

**Verified on November 12, 2026:** a fresh local checkout reprocessed the
same December 2010 source FITS into 63 RD pairs, 201 E1 tracks, 15 admitted
families, and 45,504 segmented pixels. Its final 31×256×256 mask was
bitwise identical to the prior general 63-pair run (0 changed pixels).
The original Frozen 60-pair regression also passed separately.
External catalog validation remains specific to explicit Frozen regression.
Uninterrupted remote-download-to-final execution and generalization to other
dates are not yet certified.

### Official COR2-A calibration

The repository includes the verified `Calibration/cor2A_mask.fts`
(2048×2048, 16,781,760 bytes). No manual mask installation is needed.
The pipeline checks its SHA-256 before calibration:

`7cde9474fa9d1fe93d9a6ff08464598d5d2250b423b00be449ca24b2ec7ca606`

Previously generated results using `cor2_mask.fts` are archived historical
comparisons, **not** the official COR2-A baseline. All new raw-FITS
preprocessing defaults to COR2-A. Other packaged calibration FITS are
retained for archival/legacy preprocessing compatibility. Explicit
`--mask-path` is available for deliberate controlled comparisons.

See [Integrated C0/E1 pipeline](docs/INTEGRATED_C0_E1_PIPELINE.md)
for complete CLI usage and limitations.

### Shared raw FITS and one directory per run

`run-data --source download` now saves each NASA archive/date under
`data/raw/cor2/{science,beacon}/YYYYMMDD/` and reuses each completed,
SHA-256-verified date automatically. New dates accumulate alongside existing
dates. No manual `--input-dir` is required for downloaded raw data.

Each new run writes acquisition metadata, North-up RD, C0 detection, E1 tracking,
and final CME products inside **one** `runs/<run-name>/` tree. Results from
previous layouts are not moved or deleted automatically. See
[Raw data and run layout](docs/RUN_DATA_LAYOUT.md) for exact paths, safe
legacy raw migration, and cleanup guidance. Interrupted v2 runs support
`--resume` with the same arguments and unchanged config/mask, reusing
completed stages while rejecting partial artifacts. This new resume path
still requires an execution regression on MELONHOUSE.


## Quick start: raw FITS to CME catalog

General E2E processing uses the new `run-data` CLI and the separately installed
official `Calibration/cor2A_mask.fts` (SHA-256 above):

```bash
git clone https://github.com/dngnyoo/stereo_cor2_beacon.git
cd stereo_cor2_beacon
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Install the real COR2-A static mask from your verified calibration source.
cp ~/stereo_cor2/handover/calibration/cor2A_mask.fts Calibration/cor2A_mask.fts
sha256sum Calibration/cor2A_mask.fts

python -m cme_pipeline run-data \
  --config configs/baseline_rule_b.yaml \
  --date 20101212 --run-name 20101212_cor2a_e2e_v1 \
  --source download --warmup-days 1
```

Raw archive files accumulate in `data/raw/cor2/{science,beacon}/YYYYMMDD/`.
When the experimenter selects a previously downloaded day, its verified files
are reused instead of downloaded again; new days are added to the same tree.
An individual experiment's metadata, RD, C0, E1 and final masks/catalogs
are stored together under `runs/<run-name>/`. The run name must be unique.

For the historical frozen-detection benchmark (no raw download):
`python -m cme_pipeline run --config configs/baseline_rule_b.yaml --force`.
The old-mask Frozen and official COR2-A general runs are **different scientific
conditions**; don't expect matching pixel totals.

See [QUICK_START.md](QUICK_START.md) for the complete commands, local tests,
`--resume`, migration, mask comparison and troubleshooting.

## Pipeline overview

```text
Shared NASA STEREO-A COR2 Science/Beacon FITS (data/raw/cor2)
  -> d7 30-minute cadence selection (DATE-OBS; gaps not bridged)
  -> No-MSB calibration + COR2-A mask + North-up RD
  -> C0 detection (24h causal threshold warmup)
  -> independent E1 component-first tracking
  -> Signed-PA recovery / family association / birth & size gates
  -> pixel segmentation / candidate CME catalog and visualizations
  -> run manifest + reproducibility summary
```

General runs omit the historical external catalog validator because it
assumes a fixed 32-frame Frozen sequence; the legacy benchmark retains
that validation path.

## Reproduced baseline results

The following metrics were obtained in a fresh clone of the original seven-stage baseline on **2026-11-11**, using an existing Python virtual environment. They are expected to remain unchanged by the new Products stage:

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

These are **angular reference-catalog comparison metrics**, not general pixelwise IoUs. The separately validated general 63-pair pipeline used a different input sequence and is not directly comparable on pixel counts.

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

- [QUICK_START.md](QUICK_START.md) — installation, complete output paths, verification and troubleshooting
- [`cme_pipeline/README.md`](cme_pipeline/README.md) — modular eight-stage architecture and extension guide
- [Tracking architecture](cme_pipeline/TRACKING_ARCHITECTURE.md) — tracking contracts and variants
- [Baseline configuration](configs/baseline_rule_b.yaml)
- [A5b + identity comparison configuration](configs/tracking_a5b_identity.yaml)

To see cache status: `python -m cme_pipeline status --config configs/baseline_rule_b.yaml`.

To rerun the segmentation and later stages: `python -m cme_pipeline run --config configs/baseline_rule_b.yaml --from-stage segmentation`.

CME Products can be regenerated on `main` using `--from-stage products` while reusing preceding stage caches.

## Additional research scripts

The repository retains historical Beacon/science preprocessing and experimental research scripts (including `download.py` and the four standalone preprocessing styles). These are **separate workflows** from the frozen benchmark CLI described above. Do not run `download.py` as a prerequisite for reproducing the included benchmark.

## Current limitations

- The general raw FITS → CME catalog pipeline has reproduced the old-mask
  2010-12-12 63-pair result at the pixel level in a fresh checkout.
  The new shared-data/run-layout, resume and official COR2-A baseline still
  require local execution/regression verification.
- The official COR2-A 17MB binary must be installed separately until a
  verified asset distribution is published; the tracked empty placeholder
  must not be interpreted as a valid mask.
- Historical external validation applies to the explicit Frozen
  2010-12-12 sequence only.
- Other dates and physical CME interpretations require scientific validation:
  candidate families need not be unique physical eruptions, and general
  E1 geometry still retains historical radial/occulter scale assumptions.
- Full cross-machine portability of manifests containing absolute FITS
  source paths has not yet been certified; keep old acquisition paths while
  migrating, or regenerate manifests from shared raw sources.
