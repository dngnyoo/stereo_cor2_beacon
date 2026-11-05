# STEREO-A COR2 Near-Real-Time CME Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, high-performance scientific pipeline for automated near-real-time (NRT) detection, temporal tracking, fine-scale morphology recovery, family association, pixel segmentation, and external catalog cross-validation of Coronal Mass Ejections (CMEs) from STEREO-A COR2 Beacon (d7) observations.

---

## 1. Quick Start

### 1.1 Environment Setup
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

### 1.2 One-Click Verification Demo
Run the end-to-end pipeline on the built-in benchmark dataset to verify numerical and scientific reproducibility:
```bash
python run_demo.py
```

### 1.3 Unit and Bit-Exact Reproducibility Tests
```bash
python -m unittest discover tests/
```
> Verifies that all 21 unit and regression tests pass with 100% bit-exact array matching.

---

## 2. Pipeline Architecture (7 Core Stages)

The pipeline consists of 7 modular, interchangeable stages executing in strict causal sequence:

```text
[Raw Beacon FITS]
       │
       ▼  (preprocessing/run_preprocessing.py)
   1. EXPTIME normalization, bias subtraction, SEBIP decompression
   2. WCS North-Up solar-centered common grid (256×256) reprojection
   3. 30-minute running difference (RD = t_end - t_start)
   4. Causal 24-hour rolling baseline & multi-scale patch activity
       │
       ▼  [Science Arrays: RD, Valid Mask, Manifest]
       │
[Stage 1: E1TrackingStage]
       │  (4×4 patch-grid-based 1:1 temporal tracking)
       ▼
[Stage 2: SignedPARecoveryStage]
       │  (CME position angle corridor signed fine-scale recovery)
       ▼
[Stage 3: PureDOFamilyAssociationStage]
       │  (Multi-track direct overlap [DO] CME family consolidation)
       ▼
[Stage 4: FamilyBirthRadiusGateStage]
       │  (Family-level solar root anchor filtering: r_min <= 30.0 px)
       ▼
[Stage 5: FamilySizeGateStage]
       │  (5-patch minimum size thresholding & envelope union)
       ▼
[Stage 6: IslandFirstSegmentationStage]
       │  (Stage 26 sub-patch pixel segmentation: Z > 0, Area >= 16 px)
       ▼
[Stage 7: ExternalValidationStage]
          (Quantitative validation against SEEDS, CACTus, MVCC/CORSET, Göttingen)
```

---

## 3. Directory Layout

```text
.
├── README.md                      # Main project documentation and quickstart
├── HANDOVER_GUIDE.md              # Detailed scientific principles, coordinate conventions, and tuning guide
├── requirements.txt               # Required Python dependencies
├── download_cor2.py               # NASA STEREO Science Center data downloader
├── run_demo.py                    # One-click verification demo script
│
├── calibration/                   # COR2 calibration assets (cor2_mask.fts, sebip coefficients)
│
├── preprocessing/                 # [Preprocessing Package] Raw FITS -> Science Arrays
│   ├── calibration.py             # Exposure, bias, and SEBIP decompression routines
│   ├── reproject.py               # WCS North-Up solar-centered reprojection
│   ├── detector.py                # 24h rolling baseline and Z-score standardization
│   └── run_preprocessing.py       # Standalone CLI preprocessing orchestrator
│
├── cme_pipeline/                  # [CME Pipeline Package] Object-oriented modular architecture
│   ├── core/                      # Pipeline orchestrator, Stage base class, Config, Cache, Context
│   ├── stages/                    # 7 modular stage implementations
│   │   ├── tracking/              # E1 1:1 temporal patch tracking
│   │   ├── recovery/              # Dynamic PA signed recovery
│   │   ├── association/           # Pure DO family association
│   │   ├── gates/                 # Birth Radius and Size gates
│   │   ├── segmentation/          # Island-First pixel segmentation
│   │   └── validation/            # External catalog cross-validation
│   └── utils/                     # S^1 circular geometry, hash fingerprinting, I/O helpers
│
├── configs/                       # YAML pipeline configuration files
│   └── baseline_rule_b.yaml       # Locked canonical baseline configuration
│
├── sample_data/                   # Built-in benchmark dataset (~20 MB)
│   ├── benchmark_20101212/        # Preprocessed science arrays and frame manifest (2010-12-12)
│   └── catalogs/                  # SEEDS, CACTus, MVCC/CORSET, Göttingen reference catalogs
│
└── tests/                         # Unit tests and bit-exact science array reproducibility checks
```

---

## 4. Usage

### 4.1 Downloading Raw Data from NASA Archive
To download observations for a target date from NASA STEREO Science Center:
```bash
# Download Beacon (d7) FITS files for a single date:
python download_cor2.py --date 20240510 --type beacon --output-dir ./raw_data

# Download a date range:
python download_cor2.py --start-date 20240501 --end-date 20240505 --type beacon
```

### 4.2 Preprocessing Raw FITS Files
To process raw FITS observations into standardized science arrays:
```bash
python -m preprocessing.run_preprocessing \
    --input-dir ./raw_data/20240510/ \
    --output-dir ./preprocessed/20240510/ \
    --date-str 20240510
```

### 4.3 Running the Pipeline via CLI
```bash
# Execute pipeline with default configuration
python -m cme_pipeline run --config configs/baseline_rule_b.yaml

# Re-run starting from a specific stage (reuses cached upstream outputs)
python -m cme_pipeline run --config configs/baseline_rule_b.yaml --from-stage association

# Inspect current stage cache status and fingerprints
python -m cme_pipeline status --config configs/baseline_rule_b.yaml
```

---

## 5. Benchmark Reference Metrics (2010-12-12 Benchmark)

Official reference metrics produced by `baseline_rule_b.yaml`:

| Metric | Reference Value | Verification Status |
|:---|:---:|:---:|
| **Tracking Input Tracks** | 219 tracks (Gate OFF) | Exact match |
| **Solar Root Anchors** | 33 tracks ($r_{\min} \le 30.0\text{ px}$) | Exact match |
| **Admitted Families** | 15 families (37 tracks) | Exact match |
| **Cumulative Recovered Patches** | **5,174 patches** | 100% Bit-exact match |
| **Final Active CME Pixels** | **50,665 pixels** | 100% Bit-exact match |
| **SEEDS Event Angular IoU** | **0.5242** (Radial Error: 1.74 $R_\odot$) | Exact match |
| **CACTus CME 0024 IoU** | **0.7414** | Exact match |
| **MVCC / CORSET IoU** | **0.4032** (Radial Error: 1.18 $R_\odot$) | Exact match |
| **Göttingen Event IoU** | **0.5547** | Exact match |

---

## 6. Documentation & License

For scientific hypotheses, parameter tuning, coordinate conventions, and architectural details, refer to [`HANDOVER_GUIDE.md`](HANDOVER_GUIDE.md).
Licensed under the MIT License.
