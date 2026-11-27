# Stage 09 external catalog comparison

Use an existing completed Stage 08 Run via `run-standard --input-run`.
Legacy method: seeds is unchanged (including automatic reference preparation in full runs).
For independent multi-catalog evaluation, replace the validation section in a copy
of the Stage 08 config snapshot with:

```yaml
validation:
  method: external_catalogs
  catalogs: [SEEDS, CACTus, MVCC]
  seeds_frames_csv: data/catalogs/seeds/20101212/seeds_frames.csv
  cactus_events_csv: data/catalogs/cactus/normalized/20101212/cactus_events.csv
  mvcc_frames_csv: data/catalogs/mvcc/normalized/20101212/mvcc_frames.csv
  mvcc_events_csv: data/catalogs/mvcc/normalized/20101212/mvcc_events.csv
  max_time_delta_sec: 900
  min_mean_iou: 0.1
  min_common_frames: 1
  cactus_min_iou: 0
  mvcc_max_time_delta_sec: 60
  mvcc_min_mean_iou: 0.1
```

Paths resolve from the current working directory. Prepare references with the
existing fetch/parse scripts beforehand. Missing references are recorded as
REFERENCE_UNAVAILABLE; malformed available references fail closed.
MVCC published coverage is March 2007–September 2014.
MVCC observations come from the verified segmentation ancestor, not detection rows.
No upstream science is rerun in explicit input mode. Each catalog has independent
tables and a summary; catalog_matches.csv contains assigned matches only with
their metric names. Never pool metrics. Explicit Stage 09 and full runs support `off`, `diagnostic`, and `all` visualization modes (see below).

## Optional automatic acquisition

```yaml
validation:
  method: external_catalogs
  catalogs: [SEEDS, CACTus, MVCC]
  acquisition:
    mode: auto
    reuse_existing: true
    cache_root: data/catalogs
  visualization:
    mode: "off"
```

Omit explicit reference paths to use the cache root. Existing normalized references
are reused and hashed in evaluation provenance. Explicit paths take precedence and
are never silently replaced. Missing files are fetched and parsed with the existing
source clients; incomplete normalized directories fail closed. HTTP 404/410 is
recorded as unavailable; network failures and other HTTP errors are recorded as
ACQUISITION_FAILED. Schema errors fail closed. MVCC coverage is checked before any
network request. Acquisition and visualization are independent options; acquisition alone does not render overlays.

## Integrated visualization

For method external_catalogs, explicit Stage 09 and full runs now support
`--visualize diagnostic` (up to six active frames per catalog) and `--visualize all`
(all active frames plus GIF). Outputs are in the Stage 09 child Run under
09_validation/visualizations and are hashed in stage metadata. The renderer reads
verified preprocessing and segmentation ancestors and uses the same catalog.rsun_px
as Stage 08, not a new per-frame FITS radius. SEEDS display uses nearest reference
per RD frame (not necessarily the evaluator's reverse correspondence); CACTus
radii are display guides; MVCC uses saved alignment. Optional validation key
`mvcc_assume_ht_rsun_deg: true` enables HT curves under the documented unit assumption.
Unavailable catalogs have no fabricated overlay. Existing standalone scripts remain.

## Evaluation definitions and interpretation

| Catalog | Metric | Temporal correspondence |
| --- | --- | --- |
| SEEDS | Mean frame angular IoU | Nearest detection frame per reference observation, up to 900 seconds by default |
| CACTus | Liftoff-window angular sector union IoU | Union of detected frame sectors inside the inclusive reference t0 to t0 + dt0 window |
| MVCC | Observation-aligned mean frame angular IoU | Nearest independent RD observation within 60 seconds by default; no observation excluded, observed but no detection scored zero |

Each evaluator independently assigns events one-to-one. CACTus dt0 denotes
liftoff duration, not complete visibility duration; its display radii are
not measured heights. Angular IoU is not pixel-mask IoU. Means are conditional
on assigned matches and cannot establish completeness or detection precision
on their own. Do not pool the three metrics or interpret unavailable references
as zero. MVCC manual/unmeasured events are preserved but not quantitatively
evaluated by the current CORSET parser. HT curves are optional and explicitly
assume Rsun and degrees; morphology does not require that assumption.

## Reproduction from a verified Stage 08 parent

From the external-catalog worktree, set PARENT to a completed Stage 08 Run.
Create a new configuration from its snapshot, preserving all upstream settings.
The following example enables cache-based acquisition and all overlays without
requiring per-catalog CSV paths.

```bash
PARENT=/absolute/path/to/stage08_run
export PARENT
python - <<'PY'
from pathlib import Path
import json, os, yaml
parent = Path(os.environ["PARENT"]).resolve()
cfg = json.loads((parent / "config_snapshot.json").read_text())
cfg["validation"] = {
    "method": "external_catalogs",
    "catalogs": ["SEEDS", "CACTus", "MVCC"],
    "acquisition": {
        "mode": "auto", "reuse_existing": True,
        "cache_root": str(Path("data/catalogs").resolve()),
    },
    "max_time_delta_sec": 900,
    "min_mean_iou": 0.1,
    "min_common_frames": 1,
    "cactus_min_iou": 0,
    "mvcc_max_time_delta_sec": 60,
    "mvcc_min_mean_iou": 0.1,
    "mvcc_assume_ht_rsun_deg": False,
    "visualization": {"mode": "all"},
}
dest = Path("experiments/external_catalog_research/configs")
dest.mkdir(parents=True, exist_ok=True)
with (dest / "external_replay.yaml").open("x") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
python -m cme_pipeline run-standard \
    --config experiments/external_catalog_research/configs/external_replay.yaml \
    --input-run "$PARENT" --until validation --visualize all \
    --run-name external_replay_001
```

Use unique output names. For a fresh full run, verify required upstream
settings separately, including recovery.inner_radius_px and, for imported RD,
catalog.rsun_px. A Stage 08 replay does not recompute detection, tracking,
recovery, or segmentation. Runtime caches, generated configs and local
research output are excluded from Git; retain them separately for exact replay.

## Recorded integration evidence (2026-11-26)

Code revision tested: 66e2eb2e17b641a7e358641cae4b1b0a9f4cc013.
Protected baseline: e1bce5f34d3493912b2385f4c17f1d0fc78c68f0.

These are operator-reported execution results, not a statistically designed
performance benchmark.

| Evaluation date | SEEDS matched / mean IoU | CACTus matched / mean IoU | MVCC matched / mean IoU |
| --- | ---: | ---: | ---: |
| 2010-12-12 | 4 / 0.371846102969451 | 3 / 0.4789220628830084 | 1 / 0.30833920379060165 |
| 2024-05-03 | 3 / 0.4355679743393593 | 2 / 0.5039331196599659 | unavailable |
| 2024-05-10 | 4 / 0.412302659892032 | 3 / 0.6566351336299561 | unavailable |

Local Run identifiers:

- 20240503_external_validation_001: existing strict Stage 08 from 20240503_full_002.
- 20240510_external_validation_001 and 20240510_external_visualization_001:
  existing strict Stage 08 from 20240510_full_001; numerical results and matched
  CSV identical before/after rendering; artifact verification passed.
- 20101212_external_e2e_001: imported 63-pair legacy RD, generated D4/A5b stages
  01–05, stopped on missing recovery.inner_radius_px. Resumed with separate
  20101212_external_recovery_001, 20101212_external_segmentation_001,
  20101212_external_catalog_001 and 20101212_external_validation_001.
  Stage 09 reused normalized reference caches with no explicit CSV paths.
  All three visualizations were inspected by the operator and artifact
  verification passed. The top-level initial full Run remains a failed
  Run record; completion belongs to the resumed child chain.

The 2010 experiment used one-day legacy RD input (first 32 pairs marked warmup),
D4 detection rather than the historical C0 detection, recovery.inner_radius_px
16.326 (existing standard baseline parameter, not a new physical measurement),
and catalog.rsun_px 8.440511373511905 derived from the evaluation-date FITS
RSUN and angular pixel scale. The optional MVCC HT overlay was enabled.
Do not compare it to legacy C0 scores as a same-input algorithm regression.

Acquisition test on 2010-12-12: three normalized cache reuses, two raw-only
parses (CACTus/MVCC), and three empty-cache official downloads succeeded.
MVCC monthly JSON and three event detail files reproduced existing SHA256
values; parser output was 5 events, 1 evaluable CORSET event, 22 morphology
frames and 1552 edge points. Empty-cache download was verified separately
from the integrated Stage 09 replay.

## Full test-suite status

Operator command: `python -m pytest -q` at revision 66e2eb2.

Result: 254 passed, 7 failed, 4 errors, 12 subtests passed, 122 warnings.
No external-catalog test appeared in the failure/error list.

| Existing test file | Count | Observed blocker |
| --- | ---: | --- |
| test_association_and_gates.py | 4 errors | Missing cached tracking output |
| test_configured_run_data.py | 1 failure | Expected stage list omits recovery_apply |
| test_segmentation_reproducibility.py | 2 failures | Missing historical reference arrays |
| test_tracking_integration.py | 4 failures | Missing historical reference arrays |

Diff inspection found no changes to these four tests, core, or stages from
main, and recovery_apply already exists on main. This supports classifying
the failures as existing test/environment issues; a same-environment full
main test run has not been performed. Do not report an all-green suite or
claim exact historical array reproducibility. No tests were skipped, altered,
or weakened to obtain these results. Address legacy fixture management and
registry expectations separately from this external catalog feature.

## Next research step

Keep the external evaluator isolated from Stage 01–08 scientific decisions.
For algorithm comparison, predeclare the date set, reference snapshots,
matching thresholds, observation coverage, and upstream configuration.
Evaluate completed Stage 08 products under the same protocol and retain
per-catalog results, unmatched events and unavailable states. The batch evaluation driver is documented below. Expanded date sampling and controlled multi-algorithm comparisons remain future work.

## Batch Stage 09 driver (experimental)

```yaml
catalogs: [SEEDS, CACTus, MVCC]
reference_cache: data/catalogs
runs:
  - algorithm: baseline
    date: "20101212"
    input_run: /absolute/path/to/completed/stage08
```

Save as a local YAML plan and execute
`python -m scripts.batch_external_validation --plan PLAN.yaml --output-dir runs/batch_001`.
Relative paths resolve from the working directory. Optional thresholds mapping
accepts the six existing scoring parameters; defaults match Stage 09.
Each algorithm/date label must be unique. The driver validates parent ancestry,
prepares references once per date, copies and hashes normalized scoring inputs,
and runs unchanged Stage 09 with explicit pinned references and visualization off.
Unavailable/acquisition-error catalogs are recorded in the batch table without
rerunning acquisition per algorithm. Output directory must not exist.
The batch writes reference and batch manifests, independent Stage 09 child Runs,
comparison_summary.csv, matched_events.csv, and comparison.png (separate panels).
Nonzero exit means at least one input, acquisition, reference or evaluation error.
A missing reference or no match is not a measured zero. Observation-manifest
hashes help flag unequal observation inputs but are not sufficient to certify
identical data or a controlled algorithm comparison. Use existing Stage 09
standalone execution for detailed overlays. No upstream stage is recomputed.
This driver is not yet validated against real multi-algorithm Stage 08 products.

### Batch replay validation (2026-11-26)

Tested revision: b8f1fd589df2558ccbc517c9e2dac5eeffad1a2b.
Operator-reported result: 5 batch-driver tests passed in 1.88 seconds.
Actual local Run: runs/batch_validation_001, using the three Stage 08 inputs
listed in the integration evidence above. All seven available catalog means
reproduced the earlier individual Stage 09 results to six displayed decimal
places, and matched-event counts agreed. Both 2024 MVCC entries remained
REFERENCE_UNAVAILABLE, with null scores and counts. Exact byte-for-byte
comparison and independent multi-algorithm scientific comparison have not
been established by this replay.

The integration candidate must preserve the legacy SEEDS path and the
scientific Stage 01–08 implementations. The previously recorded eleven
legacy test failures/errors remain disclosed; a fresh integration test
and a baseline control test in comparable fixture environments are required
before declaring regression equivalence. No legacy test is to be weakened
or skipped merely to obtain a green result.
