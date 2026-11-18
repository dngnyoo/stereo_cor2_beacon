# COR2 standardized pipeline (design contract, migration in progress)

This document defines the target architecture; existing legacy commands remain unchanged until parity tests pass.

## Nine stages

| # | Stage | Substeps | Contract outputs |
| --- | --- | --- | --- |
| 01 | preprocessing | Acquisition, cadence selection, North-up RD | rd, valid_mask, manifest, geometry, source_provenance |
| 02 | detection | D4, T24 thresholds, QC | patch_activation, component_labels, frame_qc, detection_scores |
| 03 | tracking | Pre-size, A5b, strict continuity | track_states, track_ownership, tracking_provenance |
| 04 | association | Identity mapping | family_membership, association_decisions |
| 05 | gates | Birth, consecutive size-qualified persistence | gate_decisions, admitted_family_ids, qualified_pair_ranges |
| 06 | recovery | Region recovery | recovered_patch_masks, recovery_metrics |
| 07 | segmentation | Pixel masks | pixel_masks, segmentation_metrics |
| 08 | catalog | CME events, height-time, kinematics | events, height_time, kinematics |
| 09 | validation | External catalog matching | catalog_matches, evaluation_metrics |

The authoritative stage identifiers and required artifact names live in `cme_pipeline/core/standard_stages.py`.

## Output layout

```text
runs/<run-id>/
  config_snapshot.json
  run_manifest.json
  01_preprocessing/{metadata.json,data/,diagnostics/}
  02_detection/{metadata.json,data/,diagnostics/}
  03_tracking/{metadata.json,data/,diagnostics/}
  04_association/{metadata.json,data/,diagnostics/}
  05_gates/{metadata.json,data/,diagnostics/}
  06_recovery/{metadata.json,data/,diagnostics/}
  07_segmentation/{metadata.json,data/,diagnostics/}
  08_catalog/{metadata.json,data/,diagnostics/}
  09_validation/{metadata.json,data/,diagnostics/}
```

Stage directories exist only for executed stages. Visualization outputs are not scientific stages and live under that stage's diagnostics directory.

## Execution rules

1. New `run --config ... --until ...` creates one new folder. Never automatically borrow outputs from other runs.
2. Explicit `run --run <existing> --until ...` may continue only with the stored immutable configuration and verified upstream artifacts.
3. Explicit `--input-run` and `--from` may create a new child run using pinned, verified parent inputs; record provenance and parent artifact checksums.
4. No mutation of completed science artifacts. Do not infer success from folder existence. Refuse inconsistent config, schema, timestamps, patch geometry, or hashes.
5. Visualization `off|diagnostic|all` defaults to off. CLI `--visualize` overrides per-stage config, which overrides shared config. Visualization changes do not invalidate scientific outputs.
6. No deletion or movement of legacy runs until equivalence is validated.

## Baseline scientific settings

20240503: Absolute D4 P95(abs(Z)), Gaussian sigma=1, causal T24 median+3 scaled MAD, outer P95 P04 floor with absolute min 0.5; P16 spatial normalized weighted tile variance cutoff 1.5; pre-size 8-connected >=8 patches; A5b outward front tolerance 4px, max_missed_frames=1, terminate_on_qc=true; Identity association; first observed minimum radius <=30px; size-qualified >=10 patches for >=3 consecutive OBSERVED frames. Outward split is **excluded**.

## Implementation migration gates

- Inventory existing module paths, imports and stage entrypoints; classify algorithm vs diagnostic vs experiment vs utility.
- Add adapters matching the artifact schema; link named algorithms through explicit registries.
- Introduce single-YAML validation and `run --until`, `--dry-run` only after wiring existing algorithms.
- Preserve old imports and CLI entrypoints as compatibility wrappers during migration.
- Cross-check old/new detection arrays, QC statuses, owned masks, tracks, family admissions and catalogs, with deterministic fixtures.
- Archive old `runs/` directories only after results and provenance are verified.

**Status:** Contract and metadata primitives landed on the refactor branch. The new orchestrator, migration, and full local parity tests are NOT yet complete.

## Phase 2: executable read-only preflight (implemented)

The new `plan` subcommand is available on the refactor branch:

```bash
python -m cme_pipeline plan --config configs/standard_baseline.example.yaml --until gates
python -m cme_pipeline plan --config configs/standard_baseline.example.yaml --until tracking --visualize diagnostic
python -m unittest discover -s tests -p "test_standard_*.py"
```

Preflight validates top-level stage names and visualization values, then prints
ordered stage IDs, algorithms, contract input/output keys and effective
visualization modes. It creates **no run** and executes **no science computation**.
Legacy `python -m cme_pipeline run` is intentionally unchanged and
does **not** yet implement the new `--until` interface.

Next: implement stage-specific adapters, contract validation and isolated
`runs/<run-id>/<number>_<stage>` output for stages 01–05. Then compare a
fresh baseline run against the frozen historical outputs before enabling a
new standard execution entrypoint.

## Phase 3: isolated scientific runner (integration tests pending)

A new command is introduced, separate from the legacy run command:

~~~bash
python -m cme_pipeline run-standard --config configs/standard_baseline.example.yaml --until gates --run-name 20240503_standard_trial_001
~~~

Before running, edit input.preprocessed_dir in the YAML to point to an existing validated No-MSB North-up RD directory; set input.spatial_metrics_csv explicitly.

To discover the preprocessing path recorded by an existing D4 detection, run:

~~~bash
python -c 'import json; print(json.load(open("runs/20240503_d4_validation/detection_metadata.json"))["input_dir"])'
~~~

Recommended first smoke test:

~~~bash
python -m unittest discover -s tests -p 'test_standard_*.py'
python -m cme_pipeline run-standard --config configs/standard_baseline.example.yaml --until preprocessing --run-name 20240503_standard_smoke_001
~~~

New run names must be unique. This first runner imports an explicitly
selected existing preprocessing output into the new 01_preprocessing/data/
directory; FITS acquisition and cadence selection from scratch are pending.

It reuses the existing D4 detector, optional normalized P16 spatial QC
transform, registered A5b tracker with pre-size, Identity Association, Birth
Gate and consecutive Size-qualified Persistence Gate. It writes each stage's
artifacts and their SHA-256 checksums under one run folder.

LIMITATIONS: This implementation is unverified on MELONHOUSE; it is not a
certified replacement for the legacy baseline. It supports standard-run only
through stage 05, only explicit preprocessed RD input, no resume, and no
automatic cross-run discovery. Non-off visualization fails explicitly
until independent diagnostic renderers are integrated. Legacy entrypoints
remain unchanged. Compare exact patch arrays, QC statuses, ownership masks
and admitted Family IDs before trusting scientific equivalence.

## Phase 4: explicit input-run branching (new, requires local integration test)

The standard-run CLI now accepts --input-run. When supplied with a completed
standard Detection Run, it reads the verified stage-02 artifacts without
re-running or copying Detection and creates a NEW run containing only the
executed stages (03 Tracking, 04 Association and/or 05 Gates).

Example after a completed standard Detection Run:

~~~bash
python -m cme_pipeline run-standard --config configs/standard_baseline.example.yaml --input-run runs/20240503_standard_detection_001 --until tracking --run-name 20240503_child_a5b_001
python -m cme_pipeline run-standard --config configs/standard_baseline.example.yaml --input-run runs/20240503_standard_detection_001 --until gates --run-name 20240503_child_gates_001
~~~

To compare Tracking variants, duplicate the baseline YAML, change
tracking.method to a3 or a4, and provide a distinct new output --run-name.

Rules:
- Exactly one explicit parent Run; no global discovery, resume, or overwrite.
- Parent must have completed stage 02, its referenced contract files must
  match recorded SHA-256 hashes, and its date and Detection configuration
  must match the new config.
- Child manifest records parent path, selected stage, source revision and
  pinned parent artifact checksums. The underlying Detection folder is
  read-only; the child does not copy it into the child run.
- Parent must be kept intact for descendants to remain reproducible.
- Current adapter accepts Detection parent only; parent Tracking or Gates
  branching is not yet implemented. CLI --until is supported only to Gates.
- Visualization other than off still requires a separate renderer.
- Baseline science parity for a full 01–05 run was confirmed by MELONHOUSE:
  old/new D4 arrays and QC, A5b ownership/states, and Gate summary matched.
  **The verified current Gate summary admitted 12 of 31 families**, not 9.
- Input-run branching itself still needs local integration testing and
  old/new ownership and Gate parity validation.

Smoke tests:
~~~bash
python -m unittest discover -s tests -p 'test_standard_*.py'
~~~

## Phase 5: explicit Tracking parent -> Association/Gates (pending MELONHOUSE test)

From a successful child Tracking Run, create a new run with only stages 04 and 05:

~~~bash
python -m cme_pipeline run-standard \
  --config configs/standard_baseline.example.yaml \
  --input-run runs/20240503_child_a5b_001 \
  --until gates \
  --run-name 20240503_child_gates_001
~~~

The CLI selects the highest completed stage in the explicitly chosen
input Run (Tracking before Detection). No resume and no upstream recomputation.
Parent Tracking contract files are SHA-256 verified, the exact Detection
ancestor is independently verified, and pre-size Tracking input arrays are
reloaded rather than inferred. The child saves only 04_association and
05_gates, with provenance pointing to both ancestors.

Different Gate thresholds can be used in separate new YAML configurations;
the Tracking settings must remain identical to the parent.

This code has not yet passed a live MELONHOUSE regression. Compare its
05_gates/data/tables/04_full_day_family_summary.csv against the verified
20240503_standard_gates_001 result (12 admitted families) before relying on
the new input-run path for science.

## Phase 6: Association Run -> Gates-only Run (needs live parity check)

A completed Association Run can now be used as the explicit input. No
Detection, Tracking, or Association computation occurs in the new child.

~~~bash
python -m cme_pipeline run-standard \
  --config configs/standard_baseline.example.yaml \
  --input-run runs/20240503_child_association_001 \
  --until gates \
  --run-name 20240503_association_to_gates_001
~~~

The new Run should have only 05_gates plus its manifest and config snapshot.
Its parent Association membership CSV is verified against recorded SHA-256;
Tracking and Detection ancestor stages are also verified. Saved Identity
Membership rows must match every observed Tracking pair/track assignment.

The original Association family IDs are retained. Birth eligibility is
re-evaluated from the saved Tracking birth_r_min_px using the *child* Gate
threshold, enabling experiments on birth radius without rerunning Association.
The saved anchor flags are checked against the original Association Run's
threshold to detect inconsistent source artifacts.

Expected baseline parity: 31 candidate families and 12 admitted; compare
the child 05_gates/data/tables/04_full_day_family_summary.csv to the verified
20240503_standard_gates_001 file, including individual family IDs.

**Important:** This implementation and its regression tests are committed,
but the new route has not been run or certified on MELONHOUSE yet.

## Phase 7: Stage 06 independent Recovery (pending local scientific validation)

The new standard Recovery adapter accepts a pure Gates-only Run with an
explicit Association parent and a verified Tracking/Detection ancestor. It
does not recompute Gates or any upstream scientific stages.

An explicit physical inner occulter radius in pixels is required. The
legacy Signed Dynamic PA implementation defaults to 16.326 px, but this
must be confirmed appropriate for the actual observation geometry before use.
In your LOCAL copy of configs/standard_baseline.example.yaml add:

~~~yaml
recovery:
  method: signed_dynamic_pa
  inner_radius_px: 16.326  # Confirm instrument and selected data geometry
  radial_window_px: 13.35
  pa_margin_deg: 5.0
  gaussian_sigma: 1.0
~~~

Execute a new run:

~~~bash
python -m cme_pipeline run-standard \
  --config configs/standard_baseline.example.yaml \
  --input-run runs/20240503_association_to_gates_001 \
  --until recovery \
  --run-name 20240503_recovery_001
~~~

The child writes only 06_recovery with union recovered patch masks,
family-specific NPZ and a JSON metric summary; Gate decisions remain read-only
from the parent. The child manifest pins the Gate stage outputs by SHA-256 and
identifies the input, Tracking and Detection ancestors.

Initial scope deliberately excludes: a Gates Run generated by a combined
01-05 or Tracking-to-Gates execution, unverified alternative recovery methods,
unverified Gaussian sigma variants, automatic resume, and visualization modes
other than off. These limitations are explicit instead of silently choosing
a different scientific path.

The Stage 06 implementation has not been validated on MELONHOUSE. First run:

~~~bash
python -m unittest discover -s tests -p 'test_standard_*.py'
~~~

Compare recovered patch masks to an independently verified original Recovery
reference with the *same* Gate family IDs, physical radius and recovery
parameters. Only identical inputs/parameters justify bitwise comparisons.

## Phase 8: Stage 07 independent Island-First Segmentation (local validation pending)

A completed *standard* Recovery Run can be an explicit input. No upstream
Recovery, Gates, Association, Tracking or Detection algorithm is repeated.

~~~bash
python -m cme_pipeline run-standard \
  --config configs/20240503_recovery.yaml \
  --input-run runs/20240503_recovery_001 \
  --until segmentation \
  --run-name 20240503_segmentation_001
~~~

The default Island-First parameters in the existing implementation are
min_island_area_px=16, max_hole_fill_px=16, occulter_r_min_px=22.0,
fov_r_max_px=122.5. To override them for a *new* experiment, set
segmentation.min_island_area_px, segmentation.max_hole_fill_px,
segmentation.occulter_r_min_px and segmentation.fov_r_max_px in YAML.

The new run contains **07_segmentation** and its metadata, including:

- arrays/final_pixel_segmentation_union.npy: (N,256,256) Boolean
- arrays/family_pixel_masks.npz: mask_<FAM_ID> for all admitted families
- tables/06_full_day_framewise_pixel_summary.csv
- tables/07_full_day_family_pixel_statistics.csv
- segmentation_summary.json: total segmented pixels, family count and provenance

Validation includes SHA-256 for all Recovery stage contract artifacts, full
Gates/Association/Tracking/Detection ancestry, strict Union equals Family-mask
union, matching frame counts, admission Family keys and saved parameters.

Original segmentation's overlapping-family policy is preserved: its global
pixel mask is segmented once, and pixels are assigned to each intersecting
Family ROI *without enforcing exclusive ownership*. The current input
Recovery has 2,748 multi-Family overlapping patch positions. No overlap
arbitration has been added.

After testing:

~~~bash
python -m unittest discover -s tests -p 'test_standard_*.py'
~~~

For a scientific parity check compare the new Stage 07 masks and tables
against the original Island-First execution run using the same saved Recovery
and segmentation parameters. Comparing only total segmented pixels is not
sufficient. This branch has not yet been executed locally against real data.

## Phase 9: Stage 08 image-plane CME Catalog (MELONHOUSE test pending)

A completed explicit Stage 07 Segmentation Run now feeds an independently
created Stage 08 Catalog Run. No previous stage is recomputed and no optional
visualizations/animations are rendered. The canonical
cme_pipeline.stages.segmentation.cme_catalog.build_cme_catalog() is reused.

**Geometry prerequisite:** Unlike pixel segmentation, the catalog expresses
front heights in R_sun and speeds in km/s. The current Tracking loader has
geometry.rsun_px=NaN. Set catalog.rsun_px to a *verified* solar radius
measured in native 256x256 pixels for these observations (not a guessed value).
The builder uses the published default solar radius 695700 km for the
height-to-km conversion.

Example (replace the placeholder only after checking FITS/WCS scale):

~~~yaml
catalog:
  method: event_kinematics
  rsun_px: <VERIFIED_SOLAR_RADIUS_IN_256PX_IMAGE>
  solar_radius_km: 695700.0
~~~

Create an alternate YAML config from configs/20240503_recovery.yaml, add
the catalog section and invoke:

~~~bash
python -m cme_pipeline run-standard \
  --config configs/20240503_catalog.yaml \
  --input-run runs/20240503_segmentation_001 \
  --until catalog \
  --run-name 20240503_catalog_001
~~~

The new 08_catalog Run writes tables/cme_event_catalog.csv
(EVENT_FIELDS), tables/cme_frame_measurements.csv (FRAME_FIELDS),
tables/cme_kinematics.csv (event-based speed view), and
catalog_summary.json. The manifest records hashes and references the exact
Segmentation parent Run. Catalog event speeds remain *exploratory* linear
fits of per-frame leading height vs elapsed time, using the canonical code's
speed_fit_r2 and speed_quality fields. They are not ground truth.

Strict preflight verifies the original Segmentation contract SHA-256, its
Recovery/Gates/Tracking/Detection chain, saved global and family pixel masks,
framewise pixel counts and family stats. Overlapping Family masks are
preserved; no new exclusivity or merging logic is introduced.

First run the regression tests, then the real Catalog run, and inspect
event speeds and fit quality. Stage 08 code is committed but local live-data
execution is still pending.

## Phase 10: Stage 09 isolated SEEDS-only external Validation (local test pending)

This Stage 09 adapter uses the existing *official GMU SEEDS COR2-A* monthly
download, detail download, detail-to-frame parser, and
`scripts.validate_seeds.validate_seeds` one-to-one temporal angular IoU
event matcher. It does **not** run the legacy frozen 2010 external-catalog
validation algorithm or automatically substitute 2010 reference data.
CACTus, Göttingen and other catalogs are deferred.

Prepare the independently supplied official SEEDS reference:

~~~bash
python -m scripts.fetch_seeds_monthly --date 20240503
python -m scripts.fetch_seeds_details --date 20240503
~~~

The scripts save raw source files and normalized frame measurements at
`data/catalogs/seeds/20240503/seeds_frames.csv`. If these files are already
available, skip downloads. If monthly source assets already exist but details
do not, run only the details downloader. If the SEEDS provider has no events
or the source format changes, stop and investigate; never synthesize frames.

Create `configs/20240503_seeds_validation.yaml` by copying the verified
`configs/20240503_catalog.yaml` (and retaining its exact Catalog/Segmentation/
Recovery/Gates configuration). Add:

~~~yaml
validation:
  method: seeds
  catalogs: [SEEDS]
  seeds_frames_csv: data/catalogs/seeds/20240503/seeds_frames.csv
  max_time_delta_sec: 900
  min_mean_iou: 0.1
  min_common_frames: 1
~~~

Then execute a new Run:

~~~bash
python -m cme_pipeline run-standard \
  --config configs/20240503_seeds_validation.yaml \
  --input-run runs/20240503_catalog_001 \
  --until validation \
  --run-name 20240503_seeds_validation_001
~~~

The resulting `09_validation/data/tables/` contains event_pair_scores.csv
and event_status.csv; `09_validation/data/summary.json` contains the matching
settings, reference SHA-256, source paths and conditional SEEDS metrics.

This validation is post-hoc: SEEDS is a non-exhaustive external *reference*,
not a ground-truth segmentation mask. Match metrics are conditional on
evaluable observations; unmatched catalog detections outside the reference
window remain OUTSIDE_EVALUATION. The parser does not invent missing SEEDS
frames or interpolate trajectories. The recorded source reference must
contain official SEEDS_COR2A_20240503T... event identifiers. Stage 09 loads
the original Stage 08 catalog from a verified completed Run, and checks its
Stage 07 through 01 ancestor chain. It never recomputes any upstream stages.

Run the standard regression tests plus existing SEEDS parser/matcher tests:

~~~bash
python -m unittest discover -s tests -p 'test_standard_*.py'
python -m pytest -q tests/test_parse_seeds_details.py tests/test_seeds_validation.py
~~~

Implementation is committed; real 2024 data execution remains to be tested.

### 2024 SEEDS monthly unwrapped PA compatibility

Observed original May 2024 GMU monthly records include PA=399 on line 31
and PA=374 on line 32. The original validator rejected all PA>360
incorrectly for these three-digit source-native identifiers. The revised
monthly parser preserves these raw values and original filename/event-ID
components (P399, P374), admitting the three-digit range 0..999 while
retaining positive WIDTH <=360 and positive observation count validation.
Circular angular comparison in the existing geometry utilities handles
normalization modulo 360. Do **not** rewrite source PA in filenames, because
detail-file retrieval must preserve the exact official source identifiers.

If monthly raw data were already downloaded by a run that failed during
parsing, **do not rerun** fetch_seeds_monthly without a deliberate overwrite.
Instead parse the cached raw monthly.txt directly:

~~~bash
python -m scripts.parse_seeds_monthly --date 20240503
python -m scripts.fetch_seeds_details --date 20240503
~~~

If the next source row raises a separate schema/geometry error, inspect
that original row instead of silently skipping or rewriting scientific data.

## Stage 01: raw FITS local-first acquisition (2024-05-10 trial)

The standard runner now supports direct local/remote **Raw FITS to new
North-up No-MSB RD** processing inside Stage 01. Existing preprocessed
imports remain supported as a distinct legacy experiment option.

Set `input.raw_dir` to reuse original local FITS files without modifying or
copying them to another source directory. If this key is omitted, Stage 01
uses the existing official dual-archive download implementation. An explicit
but missing local path is an error, **not** permission to download silently.
Both modes use `prepare_source` for d7 frame selection and
`run_no_msb_north_up` for actual RD processing. Stage 01 saves
`input_manifest.json`, the standard RD/WCS outputs and checksum-verified
stage metadata inside the same new run directory.

~~~yaml
experiment:
  date: "20240510"
input:
  raw_dir: /home/aababc/stereo_cor2/DATA_1M/STEREO_DATA/STA/d7
  warmup_days: 2
preprocessing:
  method: dual_archive_rd
~~~

To download instead, omit `raw_dir`; optionally set `input.raw_root`
for download cache location. `input.preprocessed_dir` is mutually exclusive
with `raw_dir` and raw preparation options.

~~~bash
python -m cme_pipeline run-standard \
  --config configs/20240510_full.yaml \
  --until preprocessing \
  --run-name 20240510_stage01_trial
~~~

This new acquisition capability does not yet remove the separate integrated
runner's 01–05 `--until` limit or enable stage-specific `--visualize all`.
Those remain separate implementation work. Tests mock the science functions;
real May 10 Stage 01 must be tested on the user's raw local files.
