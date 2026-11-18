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
