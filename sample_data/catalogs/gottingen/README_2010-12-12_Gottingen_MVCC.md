# 2010-12-12 Göttingen Best-of and MVCC / CORSET Verification

This document records the provenance and verification metadata for the Göttingen Best-of catalog and the MVCC/CORSET multi-viewpoint CME database for 2010-12-12.

## Scope & Preservation Principles

- The records here represent external reference datasets and are maintained independently without speculative manual merging.
- Göttingen and MVCC datasets serve as secondary observational references rather than complete automated ground-truth catalogs.
- All CSV files are UTF-8 encoded with standard comma separation and explicit column headers. Missing numeric indicators (`-999`, `None`, `NULL`, `*`) are standardized to null in analysis columns while preserved in `*_raw` provenance columns.

## Key Catalogs

| CSV File | Rows | Description |
|:---|---:|:---|
| `2010-12-12_Gottingen_BestOf.csv` | 2 | Göttingen Best-of events 566a and 566b |
| `2010-12-12_MVCC_all_observations.csv` | 10 | COR2-A and COR2-B multi-viewpoint records |
| `2010-12-12_MVCC_COR2A_events.csv` | 5 | Subset filtered for STEREO-A COR2 observations |

### Göttingen Best-of
- Event `566a` is fully confirmed across all 17 telemetry fields.
- Record `566b` serves as a placeholder record (`record_status=placeholder_not_separate_confirmed_CME`) and is not counted as a separate confirmed CME event.

### MVCC / CORSET Observations
- `stable_event_id`: Canonical event identifier (`MVCC_<event_id>`).
- `event_type`: Upstream morphology classification (`F=Flux Rope`, `O=Other`, `P=Preceding CME`).
- `CPA`, `width`: Central position angle and angular width in degrees.
- `Vrad`, `Vexp`: Radial propagation velocity and lateral expansion velocity in km/s.
- `Mproj`, `Mdeproj`: Apparent projected and deprojected mass estimates in $10^{15}\text{ g}$.

