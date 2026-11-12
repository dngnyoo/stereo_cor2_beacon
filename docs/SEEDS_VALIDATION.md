# SEEDS-only external catalog comparison (initial offline version)

This is a **post-hoc evaluation**, separate from C0 detection, E1 tracking and
segmentation. It consumes an already completed run's independently generated
`tables/cme_frame_measurements.csv`.

Run from the repository root:

```bash
python -m scripts.validate_seeds \
  --run-dir runs/clean_cor2a_verify
```

Outputs under `runs/clean_cor2a_verify/validation/seeds/`:

- `event_pair_scores.csv`: each possible SEEDS-to-detection event pair,
  mean per-frame circular angular IoU and radial error; matched assignment flag.
- `event_status.csv`: matched, unmatched SEEDS, unmatched detection, or
  detection outside evaluation timestamps.
- `summary.json`: event-level conditional match rates and configuration.

The default reference is the historical 2010-12-12
`validation/frozen/20101212/experiments/04_d7_external_catalog_validation/results/04_event_scores/seeds_frame_scores.csv`.
Only the **SEEDS source measurements** (`parent_event_id`, `seeds_timestamp`,
`seeds_bpa`, `seeds_width`, `seeds_lead`, `is_evaluable`) are read.
The older candidate association and matching score columns are **ignored**.

For another date use `--seeds-csv /path/to/independently-prepared-SEEDS.csv`
with the same source columns. This first version does not download/parse
the upstream SEEDS catalog automatically; source CSV provenance must be
documented separately.

Default matching uses the nearest observation within 900 s, at least one
angularly overlapping frame, mean IoU >= 0.1, and a globally one-to-one
maximum-weight event assignment. These are *initial operational thresholds*
that should be sensitivity-tested before reporting scientific conclusions.

**Interpretation:** SEEDS may omit events and the historical reference marks
some observation gaps as non-evaluable. Report conditional precision, recall
and F1 as *SEEDS agreement*, **not** independently verified CME detection
accuracy. Detections outside SEEDS-evaluable timestamps are reported separately
and excluded from conditional precision. Width-based angular IoU is not
pixel-level segmentation IoU. Speed is not evaluated in this first version.

Run tests:
```bash
python -m pytest -q tests/test_seeds_validation.py
```

## Visual review: overlay SEEDS on existing segmentation PNGs

This is the quick **image-only** review path. It does NOT rerun detection,
load 3D mask arrays, or calculate IoU; it leaves all original PNGs untouched.

```bash
python -m scripts.visualize_seeds_overlay \
  --run-dir runs/clean_cor2a_verify \
  --gif
```

Input: `<run>/visualizations/segmentation_overlays/pair_###_YYYYMMDDTHHMMSSZ_families.png`

Outputs:
- `<run>/validation/seeds/frames/pair_###_..._seeds.png`
- `<run>/validation/seeds/seeds_overlay.gif` (optional `--gif`)

Magenta translucent sectors come from SEEDS central PA, angular width,
and `seeds_height` (radius converted using `--rsun-px`), using only
evaluable source observations within the default ±900-second time window.

**Not a true SEEDS binary pixel mask**: the drawn wedge is an approximate
angular/height region intended for visual inspection. The source PNG already
contains the pipeline's segmentation contours and labels. Existing PNGs use a
fixed 864×840 layout (7.2×7 in at 120 dpi) with a centered square axes region.
The overlay checks this layout before drawing. It uses historical geometry
defaults: `--sx 126.618 --sy 127.503 --rsun-px 8.163 --inner-px 22
--outer-limit-px 120`; adjust them explicitly if your run's geometry differs.
Wrong orientation or geometry can produce a visually misleading overlay.
This utility uses source timestamps encoded in PNG filenames rather than
historical frame offsets.

```bash
python -m pytest -q tests/test_seeds_png_overlay.py
```

## Fetch SEEDS observations from an HTTPS CSV source

The NASA STEREO event portal links to the GMU SEEDS catalog, but a
verified direct COR2-A bulk CSV download URL **has not been established**.
Do **not** infer an endpoint or silently substitute LASCO/C2 data.
This collector supports source URLs supplied by the user or a confirmed
archive; it is **not** yet an unattended date-only official-site downloader.

After identifying a verified *COR2-A frame-measurement CSV* endpoint:

```bash
python -m scripts.collect_seeds \
  --date 20101212 \
  --source-url 'https://VERIFIED-ARCHIVE/path/to/seeds_frame_measurements.csv'
```

For verified year/day-organized archives, replace `--source-url` with
`--url-template 'https://VERIFIED-ARCHIVE/{year}/{month}/{date}.csv'`.
The tool downloads up to 20 MiB over HTTPS, validates the known frame-level
SEEDS reference schema, filters by UTC date, and saves:

- `data/catalogs/seeds/YYYYMMDD/original_source.csv`
- `data/catalogs/seeds/YYYYMMDD/seeds_frame_reference.csv`
- `data/catalogs/seeds/YYYYMMDD/source_provenance.json`
  (URL, UTC download timestamp, original SHA-256).

The output can then be passed to either:

```bash
python -m scripts.validate_seeds --run-dir runs/clean_cor2a_verify \
  --seeds-csv data/catalogs/seeds/20101212/seeds_frame_reference.csv
python -m scripts.visualize_seeds_overlay --run-dir runs/clean_cor2a_verify \
  --seeds-csv data/catalogs/seeds/20101212/seeds_frame_reference.csv --gif
```

Important: a generic event-only SEEDS table is *not* interchangeable
with the per-frame PA/width/height measurements this pipeline requires.
The collector refuses unknown schemas and does not synthesize missing
per-frame measurements. An upstream event-only catalog requires its own
documented, scientifically justified conversion method before use.

```bash
python -m pytest -q tests/test_collect_seeds.py
```

## Official GMU SEEDS STEREO-A/COR2 monthly TXT source

The confirmed source directory pattern is:

`http://spaceweather.gmu.edu/seeds/secchi/detection_cor2/a/YYYY/MM/`

with `monthly.txt` (event records), `daily.txt` (daily detections)
and `daily_images.txt` (daily image counts). These are distinct datasets.

```bash
python -m scripts.fetch_seeds_monthly --date 20101212
```

This writes the exact original bytes (without trying to convert them)
under `data/catalogs/seeds/raw/2010/12/` together with
`source_provenance.json` recording SHA-256 and the actual download URL.
HTTP is supported here because the supplied official endpoint is HTTP;
do not send authentication secrets to it.

**Remaining step:** the upstream GMU host was not reachable from the
development environment, so the actual row/column layout in
`monthly.txt` has not yet been verified. We deliberately **do not**
interpret positional columns, fabricate per-frame heights, or classify
daily counts as CME events. After downloading locally, inspect:

```bash
sed -n '1,35p' data/catalogs/seeds/raw/2010/12/monthly.txt
```

Use a real sample to build a tested official TXT parser and then connect
it to the separate event-level and frame-level comparison stages.
For now, `scripts.collect_seeds` remains for an already normalized
frame-measurement CSV provided by a verified source; `fetch_seeds_monthly`
is the direct official-TXT downloader.

```bash
python -m pytest -q tests/test_fetch_seeds_monthly.py
```

## Parser verified against authentic 2010-12-12 monthly rows

Actual `monthly.txt` rows supplied from the GMU COR2-A archive have the
whitespace-delimited structure:

```text
2010/12/12 04:54:00   8  91 138   499     5.7
```

Known event-level columns used by the parser: timestamp UTC, observation
count, central position angle, angular width, and speed. The trailing
numeric/placeholder field is preserved as unverified auxiliary metadata
and is NOT used to calculate any CME metric. The parser constructs the
same stable event identifiers already present in the frozen reference
(e.g. `SEEDS_COR2A_20101212T045400_W138_V0499_P091`).

Because this monthly file contains **event summaries**, it does not
provide individual image timestamps, per-frame varying angular width, or
height. The validated event CSV is not a drop-in replacement for the
curated frame-measurement table and cannot truthfully produce the existing
sector-per-frame overlay by itself.

The normal downloader now also parses the selected date:

```bash
python -m scripts.fetch_seeds_monthly --date 20101212
```

If the raw monthly file has already been fetched, parse it **without
redownloading**:

```bash
python -m scripts.parse_seeds_monthly --date 20101212 \
  --input data/catalogs/seeds/raw/2010/12/monthly.txt \
  --output data/catalogs/seeds/raw/2010/12/seeds_events_20101212.csv
```

The output of the downloader is
`data/catalogs/seeds/raw/YYYY/MM/seeds_events_YYYYMMDD.csv`;
the standalone parser also allows a custom output path.

Unit test:
```bash
python -m pytest -q tests/test_parse_seeds_monthly.py
```
