# 2010-12-12 COR2-A External Catalog Verification

This directory contains reference records for SEEDS and CACTus catalogs for the 2010-12-12 observation window.

## Deliverables

- `2010-12-12_SEEDS_COR2A_events.csv`: 1 row per event. Contains all header fields, stable event IDs, source filenames, and provenance.
- `2010-12-12_SEEDS_COR2A_frames.csv`: 1 row per measurement frame. Contains original measurements, parent event IDs, and source filenames.
- `2010-12-12_CACTus_COR2A_events.csv`: 8 CME events and 1 Flow event from the CACTus SECCHI-A/COR2 catalog where `t0` falls on 2010-12-12.

Both CSV files are encoded in UTF-8 with standard comma separation.

## Event IDs and Provenance

- `event_id` is formatted as `SEEDS_COR2A_YYYYMMDDThhmmss_Wwww_Vvvvv_Pppp`, generated deterministically from date, time, angular width, speed, and principal position angle tokens.
- `provenance` records the upstream SEEDS detail source, origin paths, and SHA-256 integrity checksums.

## Fields and Definitions

- Event CSV: `DATE_CREATED`, `DATE_OBS`, `TIME_OBS`, `DETECTOR`, `OBSERVER`, `IMAGE_TYPE`, `NUM_FRAMES`, `ONSET1`, `ONSET2`, `ONSET2_RSUN`, `CEN_PA`, `WIDTH`, `SPEED`, `ACCEL`, `REMARK`.
- Frame CSV: `DATE`, `TIME`, `HMLEAD`, `HMHEIGHT`, `HMFOLLOW`, `TEL`, `PASTART`, `PAEND`, `BPA`.
- Following standard SEEDS conventions: `HMLEAD`, `HMHEIGHT`, `HMFOLLOW`, `ONSET2_RSUN` are in solar radii ($R_\odot$); `CEN_PA`, `WIDTH`, `PASTART`, `PAEND`, `BPA` are in degrees ($^\circ$); `SPEED` is in km/s; and `ACCEL` is in m/s$^2$.

## Verification Results

- Events: Exactly 6 events, all with `DETECTOR=cor2A` and `DATE_OBS=2010/12/12`.
- Frames: Exactly 30 measurement frames across the 6 events.
- All frames successfully mapped to parent event IDs without duplication.
- Confirmed bit-level consistency against official monthly SEEDS catalog archives.
