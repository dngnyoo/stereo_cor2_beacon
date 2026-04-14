# STEREO/COR2 Data Processing Pipeline

This project is a comprehensive pipeline for processing solar corona data from the STEREO-A/COR2 instrument. It supports four distinct processing styles, ranging from rapid monitoring to high-precision scientific analysis, utilizing **Z-score Running Difference (RD)** to enhance the visibility of Coronal Mass Ejections (CMEs).

## Project Overview
The primary goal is to extract clean solar event signals by mitigating instrumental noise and background fluctuations. By standardizing images (Z-score) before calculating differences, the pipeline ensures robust detection of plasma motion regardless of varying exposure levels or sensor conditions.

## Directory Structure
```text
.
├── common_utils.py       # Shared utilities (FITS loading, Z-score logic, paths)
├── download.py           # Data crawler from NASA server
├── 1_beacon_gcs.py       # Style 1: Fast Beacon Monitoring (Raw Z-score RD)
├── 2_beacon_cmad.py      # Style 2: Physical Beacon Analysis (pB Synthesis + CMAD)
├── 3_science_hv.py       # Style 3: Science Helioviewer-style (Simple RD)
├── 4_science_cmad.py     # Style 4: High-precision Science (Gold Standard)
├── functions.py          # External SCC library for basic FITS handling
├── requirements.txt      # Python dependencies
└── DATA/                 # Directory where raw data is stored (created by download.py)
```

## Core Technology: Z-score Running Difference
Traditional running difference (t+1 - t) is highly sensitive to global brightness shifts caused by sensor temperature or exposure timing. 

Our pipeline implements **Z-score Standardization** for each frame before subtraction. This approach highlights structural changes in units of standard deviation, effectively suppressing background noise and artifacts.

---

## Installation and Setup

### 1. Environment Setup
It is recommended to use a Python virtual environment:
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Acquisition
Since solar datasets are large, you must download the specific data for your target date:
```bash
python download.py
# Enter the date (e.g., 20240510) when prompted. 
# It will automatically populate the ./DATA folder with the required structure.
```

---

## Execution Guide

### Style 1: Beacon GCS (Fast)
Quick-look tool for real-time monitoring using compressed Beacon data.
```bash
python 1_beacon_gcs.py
```

### Style 2: Beacon CMAD (Physical)
Applies CMAD background subtraction and pB synthesis to Beacon data for scientific validity.
```bash
python 2_beacon_cmad.py
```

### Style 3: Science Helioviewer (Simple)
Processes high-resolution science data using simple subtraction for comparison with public web viewers.
```bash
python 3_science_hv.py
```

### Style 4: Science CMAD (High-Precision)
The "Gold Standard" pipeline using full L1/L2 calibration, CMAD subtraction, and Z-score RD on high-res data.
```bash
python 4_science_cmad.py
```

---

## Output
All processed results are saved in the `./RESULTS/` directory, categorized by style. Each FITS file includes updated header metadata (VMIN, VMAX, HISTORY) for immediate visualization.