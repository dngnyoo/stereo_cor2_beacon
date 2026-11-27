# Stage 09 external catalog comparison

Use an existing completed Stage 08 Run via run-standard --input-run.
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
their metric names. Never pool metrics. Full-run visualization provides a summary
plot; detailed overlay/GIF tools remain standalone and are not invoked automatically.
Explicit input execution uses --visualize off.
