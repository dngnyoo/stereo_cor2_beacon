# Stereo_COR2 — Modular Pipeline Refactoring Final Report

**Project Identifier**: `cme_pipeline`  
**Target Dataset**: 2010-12-12 STEREO-A COR2 Beacon d7 (Pairs 29–60, 32 full observation pairs)  
**Baseline Pipeline**: Baseline Rule B (E1 Tracking + Signed Dynamic PA Recovery + Stage 32B-2 Rule B Direct Overlap + Family Birth Radius Gate + Size Gate + Island-First Segmentation + External Validation)  

---

## 1. Executive Summary

The 7 core scientific processing stages of the Stereo_COR2 project (Detection/Tracking, Fine Recovery, Family Association, Solar Birth Gate, Size Gate, Sub-Patch Pixel Segmentation, and External Catalog Cross-Validation) have been fully refactored from legacy scripts into an **independently executable, class-based modular package (`cme_pipeline/`)**.

### Key Achievements
1. **100% Bit-Exact Scientific Reproducibility**:
   - Cumulative Recovered Patches: **5,174 patches** (`np.array_equal == True`, diff count = 0)
   - Cumulative Segmented Pixels: **50,665 pixels** (`np.array_equal == True`, diff count = 0)
   - Admitted Families (15) and Admitted Tracks (37): 100% exact match
   - External Validation Metrics: SEEDS (0.5242, 1.7413 $R_\odot$), CACTus All (0.2715), CACTus CME 0024 (0.7414), MVCC (0.4032, 1.1843 $R_\odot$), Göttingen (0.5547) identical to 4 decimal places
2. **Independent Stage Interfaces & Explicit Typed Data Contracts**:
   - Subclassed from `PipelineStage` abstract base class.
   - Dataclass-based I/O data models (`TrackingOutput`, `RecoveryOutput`, `AssociationOutput`, `BirthGateOutput`, `SizeGateOutput`, `SegmentationOutput`, `ValidationOutput`) eliminating global state mutations.
3. **Stage-Level Local File Caching & Incremental Execution (`--from-stage`)**:
   - Deterministic SHA-256 fingerprint caching based on algorithm version, configuration parameters, and input data hashes.
   - Instant restore within 0.28 seconds on cache hits.
   - `--from-stage` allows selective downstream re-runs.
4. **Isolated Experimentation & Comprehensive Unit Testing**:
   - Full isolation in `runs/<run_name>/` preserving previous results.
   - Comprehensive unit and regression test suite passing with 0 failures, 0 errors.

---

## 2. Benchmark Verification Matrix

| Pipeline Stage | Scientific Quantity | Reference Value | Modular Output | Match Status |
|:---|:---|:---:|:---:|:---:|
| **Tracking** | Total Input Tracks (Gate OFF) | 219 tracks | 219 tracks | Exact Match |
| **Tracking** | Solar Root Anchors ($r \le 30$) | 33 tracks | 33 tracks | Exact Match |
| **Association** | Candidate Family Groups | 185 groups | 185 groups | Exact Match |
| **Association** | Association Edges (DO $\ge 0.05$) | 222 edges | 222 edges | Exact Match |
| **Birth Gate** | Admitted Solar Families | 23 families | 23 families | Exact Match |
| **Birth Gate** | Rejected Non-Solar Families | 162 families | 162 families | Exact Match |
| **Size Gate** | Admitted Persistent Families | 15 families | 15 families | Exact Match |
| **Size Gate** | Admitted Member Tracks | 37 tracks | 37 tracks | Exact Match |
| **Size Gate** | Cumulative Recovered Patches | **5,174 patches** | **5,174 patches** | **Bit-Exact (100%)** |
| **Segmentation** | Cumulative Segmented Pixels | **50,665 pixels** | **50,665 pixels** | **Bit-Exact (100%)** |
| **Validation** | SEEDS Event Angular IoU | 0.5242 | 0.5242 | Exact Match |
| **Validation** | CACTus CME 0024 Angular IoU | 0.7414 | 0.7414 | Exact Match |
| **Validation** | MVCC / CORSET Angular IoU | 0.4032 | 0.4032 | Exact Match |
| **Validation** | Göttingen Best-Of Angular IoU | 0.5547 | 0.5547 | Exact Match |

---

## 3. Conclusion

The refactored `cme_pipeline` provides clean modularity, deterministic caching, and extensible stage contracts while guaranteeing zero divergence from the verified scientific baseline.

