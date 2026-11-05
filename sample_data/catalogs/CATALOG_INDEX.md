# External CME Catalog Master Inventory & Provenance Index (2010-12-12)

**Target Date**: 2010-12-12 UTC  
**Target Location**: `sample_data/catalogs/`  

> [!NOTE]
> This index records the physical paths, record counts, core fields, and provenance metadata of the external CME catalog reference datasets for 2010-12-12.

---

## 1. Catalog Summary

| Catalog | Instrument | Key Filename | Valid Records | Scientific Scope & Role |
| :--- | :---: | :--- | :---: | :--- |
| **SEEDS** | COR2-A | `2010-12-12_SEEDS_COR2A_events.csv` | 6 events | Automated segmentation CME event summary (CPA, Width, Speed) |
| **SEEDS** | COR2-A | `2010-12-12_SEEDS_COR2A_frames.csv` | 30 frames | Frame-by-frame angular sectors for 6 events (PASTART, PAEND, BPA) |
| **CACTus** | COR2-A | `2010-12-12_CACTus_COR2A_events.csv` | 9 events | Optical-flow-based detections (8 CMEs + 1 Flow) |
| **MVCC** | COR2-A | `2010-12-12_MVCC_COR2A_events.csv` | 5 events | Multi-viewpoint CORSET event-level summary (CPA, Width, Vrad) |
| **MVCC** | COR2-A | `2010-12-12_MVCC_CORSET_height_time_COR2A.csv` | 1,552 records | Precision height-time radial trajectory tracking data |
| **Göttingen** | STEREO-A | `2010-12-12_Gottingen_BestOf.csv` | 2 events | Best-of manual inspection catalog (1 confirmed + 1 incomplete) |

---

## 2. File Manifest

| Catalog | Filename | Rows / Size | Purpose & Key Fields |
| :--- | :--- | :---: | :--- |
| **CACTUS** | `2010-12-12_CACTus_COR2A_events.csv` | 9 rows | CACTus COR2-A detections: 8 CME events + 1 Flow event |
| **GOTTINGEN** | `2010-12-12_Gottingen_BestOf.csv` | 2 rows | Göttingen Best-of CME catalog records (2 events) |
| **GOTTINGEN** | `2010-12-12_Gottingen_MVCC_bundle.zip` | 295.1 KB | Archive of original raw tables and assets |
| **GOTTINGEN** | `2010-12-12_Gottingen_MVCC_source_manifest.csv` | 12 rows | Provenance manifest of original Göttingen/MVCC source assets |
| **GOTTINGEN** | `2010-12-12_Gottingen_MVCC_validation.json` | JSON meta | Structured JSON schema validation report |
| **MVCC** | `2010-12-12_MVCC_Balmaceda_deprojection.csv` | 1 row | Balmaceda method 3D mass deprojection comparison |
| **MVCC** | `2010-12-12_MVCC_COR2A_events.csv` | 5 rows | MVCC COR2-A event records (5 confirmed CME events) |
| **MVCC** | `2010-12-12_MVCC_CORSET_fit_parameters_COR2A.csv` | 3 rows | CORSET COR2-A kinematic trajectory polynomial fit parameters |
| **MVCC** | `2010-12-12_MVCC_CORSET_headers_COR2A.csv` | 29 rows | CORSET COR2-A FITS observation header metadata |
| **MVCC** | `2010-12-12_MVCC_CORSET_height_time_COR2A.csv` | 1552 rows | CORSET COR2-A detailed height-time radial track measurements |
| **MVCC** | `2010-12-12_MVCC_CORSET_mass_lat_height_deprojection_COR2A.csv` | 21 rows | Deprojected mass with latitude and height correction (COR2-A) |
| **MVCC** | `2010-12-12_MVCC_CORSET_morphology_COR2A.csv` | 22 rows | CORSET COR2-A morphology time series (PA, width, area) |
| **SEEDS** | `2010-12-12_SEEDS_COR2A_events.csv` | 6 rows | SEEDS automated segmentation catalog events |
| **SEEDS** | `2010-12-12_SEEDS_COR2A_frames.csv` | 30 rows | SEEDS frame-level sector boundaries and BPA |
