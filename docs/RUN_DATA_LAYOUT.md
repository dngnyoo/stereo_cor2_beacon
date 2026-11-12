# COR2 raw data and run layout (v2)

## Core policy

Raw NASA observations are shared by **date and archive**, not by run name.
The first complete acquisition of one UTC date writes:

```text
data/raw/cor2/
  science/YYYYMMDD/*.fts
  science/YYYYMMDD/.download_complete.json
  beacon/YYYYMMDD/*.fts
  beacon/YYYYMMDD/.download_complete.json
```

Both archives are retained; Science L0 d7 is preferred where available,
Beacon d7 is the fallback. Byte-identical cross-archive files are retained
rather than deleted so that a future run does not redownload them.
The completion marker stores file sizes and SHA-256. On a second run, fully
verified days are reused **without any HTTP request**. A missing date is
downloaded into its matching directory. Missing or checksum-mismatched files
in a previously completed day are refreshed atomically, without overwriting
unrelated cached FITS. A legacy directory without a completion marker is
refreshed from a listing once, after which its existing nonempty files are
reused and a marker is written.

Each `run-data` run owns all of its output below a single name:

```text
runs/<run_name>/
  run_request.json
  acquisition/
    input_manifest.json
    manifests/raw_inventory.csv
    manifests/selected_d7_sources.csv
    preprocessed/
      01_rd_manifest.csv
      north_up_RD_DNs_per_second.npy
      north_up_pair_valid_masks.npy
      preprocessing_metadata.json
      target_wcs_header.txt
  detection/
    detection_metadata.json
    frame_detection_summary.csv
    patch_state_p04.npy
    patch_component_labels.npy
  tracking/
    03_patch_e1_track_states.csv
    01_patch_component_geometries.csv
    02_patch_e1_candidate_evaluations.csv
    summary.json
  stages/
  tables/
  arrays/
  visualizations/
  .cache/
  pipeline.log
  report.md
  manifest.json
  integrated_run_summary.json
```

If `--preprocessed-dir` or `--detection-dir` is used, the precomputed inputs
remain external; only stages computed by the new run are placed under it.
Explicit `--acquisition-dir` is supported for advanced use and can override
the default in-run acquisition folder. A completed run is never overwritten.
`prepare-data --source download --product d7` uses the same shared
`data/raw/cor2/` tree while keeping its own chosen output directory.

## Reproducibility

The historic verified 63-pair replay remains intact in prior run
directories. The layout migration does **not** overwrite those results
or silently rewrite absolute paths in their input manifests. Do not delete
old acquisition directories until references and results have been backed up.

The official calibration input is `Calibration/cor2A_mask.fts`, now included
in Git. Its SHA-256 is verified before raw preprocessing. No manual copy or
installation is required for new clones. The old-mask 63-pair run is retained
as a historical comparison, not the official COR2-A scientific baseline.

## Safe migration of existing 1.2GB raw data

From the checkout root, first preview:

```bash
python scripts/migrate_shared_raw.py \
  --from-raw runs/acquisition_20101212_raw_e2e_20101212/raw
```

If the preview is correct, migrate while preserving the old paths:

```bash
python scripts/migrate_shared_raw.py \
  --from-raw runs/acquisition_20101212_raw_e2e_20101212/raw \
  --apply --hardlink
```

`--hardlink` saves space on the same filesystem; the old raw directory must
remain untouched while older manifests refer to it. Omit `--hardlink` to
create separate verified copies instead. The migration refuses any
different-content existing destination, never removes the source, and
verifies SHA-256 for every newly created file.

## Cleanup

- **Keep:** `data/raw/cor2`, verified run manifests, configuration, key tables,
  final masks, products and validation artifacts.
- **Optional after independent verification:** obsolete failed runs,
  duplicated CSV exports and disposable stage caches.
- **Never automatically delete:** raw files, unique experiments, historical
  frozen references, or files still referenced by recorded manifests.

## Interrupted run recovery

Use `--resume` with **the same** `run-data` arguments, run name, configuration
content and mask bytes. The persisted `run_request.json` locks these
inputs. It reuses complete acquisition manifests, RD, C0, and E1 stage files
and re-runs the canonical downstream stages when the final report is absent.
An incomplete stage directory is rejected instead of being overwritten.
If `integrated_run_summary.json` exists, the completed run is reused without
work. Runs from the legacy layout lack `run_request.json` and must be
replayed under a new name.

```bash
python -m cme_pipeline run-data \
  --config configs/baseline_rule_b.yaml \
  --date 20101212 --run-name 20101212_cor2a_baseline \
  --source download --warmup-days 1 --resume
```

Resume of a partially completed stage is fail-closed; inspect that stage,
back it up and create a new run name rather than deleting uncertain data.
Completed shared-date raw acquisition is reused even with different run names.

The 2010 general 63-pair pre-mask-switch pixel-identical regression is
distinct from new official COR2-A results. The layout/resume changes have
not yet been scientifically replayed on MELONHOUSE; run the provided offline
tests and one 63-pair regression before treating this as certified.
