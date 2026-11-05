# STEREO-A COR2 CME Pipeline Technical Handover Guide

This document provides a comprehensive scientific and engineering guide for successors, collaborators, and researchers taking over this codebase.

---

## 1. Core Scientific Principles & Design Philosophy

### 1.1 Near-Real-Time Beacon (d7) Operational Constraint
- The primary goal of this pipeline is **automated, near-real-time (NRT) early detection and tracking of CMEs using compressed Beacon (d7) telemetry** for operational space weather forecasting.
- Retrospectively available high-resolution Science (d4) imagery must **never** be inserted into the Beacon time series or used to interpolate cadence gaps during NRT evaluation.
- Actual Beacon telemetry timestamps and irregular observation cadences ($\Delta t$) must be strictly preserved.
- When telemetry interruptions exceed 75 minutes ($\Delta t > 4,500\text{ s}$), track termination is enforced to maintain strict physical causality.

### 1.2 WCS Solar Geometry & Position Angle (PA) Conventions
- All observations are reprojected onto a standardized North-Up solar-centered grid ($256 \times 256$ pixels):
  - Solar Center: $(SX, SY) \approx (126.618, 127.503)$
  - Scale: $1 R_\odot \approx 8.163\text{ px}$ ($\approx 117.6''/\text{px}$)
  - Inner Occulter: $r \approx 22.0\text{ px}$ ($\approx 2.7 R_\odot$)
  - Outer FOV: $r \approx 122.5\text{ px}$ ($\approx 15.0 R_\odot$)
- CME Position Angle (PA) Convention:
  $$\text{PA} = \left(\text{degrees}\left(\text{atan2}(-\text{HPLN}, \text{HPLT})\right)\right) \pmod{360^\circ}$$
  - $0^\circ$: Solar North
  - $90^\circ$: Solar East
  - $180^\circ$: Solar South
  - $270^\circ$: Solar West
- When computing angular differences, spans, or mean directions, always use the circular geometry module (`cme_pipeline/utils/geometry.py`) to correctly handle the $0^\circ/360^\circ$ periodic boundary ($S^1$).

---

## 2. Seven Core Pipeline Stages: Methodological Details

### Stage 1: E1 1:1 Temporal Tracking (`E1TrackingStage`)
- Evaluates $4 \times 4$ patch activity against a causal 24-hour rolling robust threshold ($T_{24} = \text{median} + 3.0 \times 1.4826 \times \text{MAD}$).
- Establishes 1:1 temporal associations between consecutive frames based on spatial proximity and angular consistency.
- **Critical Policy**: Individual tracks are **not** filtered upfront by solar birth radius (Gate OFF maintains all 219 tracks). Premature filtering at the track level inadvertently discards valid fragmented flanks of massive CMEs.

### Stage 2: Dynamic PA Signed Recovery (`SignedPARecoveryStage`)
- Constructs an adaptive propagation corridor ($\pm 5.0^\circ$) centered along the observed trajectory of each active track.
- Recovers fine-scale positive difference features ($Z > 0$) using 8-connectivity within the corridor.
- Prevents noise explosion while recovering subtle CME substructures that fall below coarse detection thresholds.

### Stage 3: Pure DO Family Association (`PureDOFamilyAssociationStage`)
- Massive physical CMEs often split into multiple tracks across coarse patch grids.
- Identifies spatial overlap (Direct Overlap, DO) or immediate 8-adjacency between tracks within the same observation frame to build an association graph.
- Merges connected components into cohesive physical **CME Families**.
- Threshold: `do_threshold = 0.05` ($\ge 5\%$ area overlap or adjacent contact).

### Stage 4: Family Birth Radius Gate (`FamilyBirthRadiusGateStage`)
- Evaluates solar root anchor physics at the **Family level**, rather than individual tracks.
- A family is validated as a true solar-originating CME if at least one member track originated within $r_{\min} \le 30.0\text{ px}$ ($\le 3.67 R_\odot$).
- Transient external noise clusters and coronal background fluctuations are rejected at this stage.

### Stage 5: Family Size Gate (`FamilySizeGateStage`)
- Enforces an event-level persistence constraint: requires a minimum cumulative detected area of 5 patches (`min_detected_patches = 5`) across the family lifetime.
- Consolidates all recovered spatial envelopes of admitted member tracks.

### Stage 6: IslandFirst Pixel Segmentation (`IslandFirstSegmentationStage`)
- Extracts true positive plasma enhancements ($Z > 0$) within the multi-track patch recovery envelope.
- Applies morphological cleaning: rejects isolated noise islands smaller than 16 pixels (`min_island_area_px = 16`) and fills interior voids $\le 16\text{ px}$.
- Yields the final binary CME segmentation masks.

### Stage 7: External Catalog Validation (`ExternalValidationStage`)
- Benchmarks segmented CME masks against four major international CME catalogs:
  - **SEEDS**: Angular IoU and radial leading-edge distance error ($R_\odot$)
  - **CACTus**: Overall angular IoU and principal CME 0024 IoU
  - **MVCC / CORSET**: Multi-view 3D reconstructed boundary IoU and radial error
  - **Göttingen**: Catalog IoU alignment

---

## 3. Configuration & Parameter Tuning Guide

Key parameters in `configs/baseline_rule_b.yaml`:

| Parameter | Default | Scientific Purpose & Tuning Guidance |
|:---|:---:|:---|
| `birth_gate.max_birth_radius_px` | `30.0` | Solar root anchor radius ($\approx 3.67 R_\odot$). Increase if early emergence occurs further out in FOV. |
| `size_gate.min_detected_patches` | `5` | Minimum family patch count. Prevents single-frame transient noise from triggering detections. |
| `association.do_threshold` | `0.05` | Minimum overlap ratio for family merging. Lower values encourage wider family consolidation. |
| `segmentation.min_island_area_px` | `16` | Minimum connected island area. Filters out single-pixel background pepper noise. |
| `recovery.pa_margin_deg` | `5.0` | Angular corridor margin around CPA. Increase to $7.5^\circ$ for extremely wide or expanding CMEs. |

---

## 4. Environment & Successor Checklist

1. **Python Environment**: Python 3.10, 3.11, or 3.12 recommended.
2. **Calibration Assets**: Verify `calibration/cor2_mask.fts` exists.
3. **Self-Consistency Verification**:
   ```bash
   python run_demo.py
   python -m unittest discover tests/
   ```
   Both commands must pass with zero errors on any newly deployed machine.
4. **Pushing to a New Repository**:
   - Navigate into `handover/` and initialize a fresh git repository:
     ```bash
     cd handover
     git init
     git add .
     git commit -m "Initial commit: Modular STEREO-A COR2 CME pipeline"
     git branch -M main
     git remote add origin <NEW_REMOTE_URL>
     git push -u origin main
     ```
