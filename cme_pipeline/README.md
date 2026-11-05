# CME Pipeline: Modular Architecture & Developer Guide

`cme_pipeline` is an object-oriented, modular scientific pipeline package for automated detection, temporal tracking, fine-scale morphology recovery, family association, pixel segmentation, and external catalog cross-validation of Coronal Mass Ejections (CMEs) from STEREO-A COR2 observations.

It preserves 100% bit-exact numerical reproducibility against baseline benchmark experiments while allowing researchers to seamlessly swap, ablate, or extend individual pipeline stages.

---

## 1. Pipeline Architecture

The pipeline consists of 7 modular stages executing in strict causal order:

```text
[1. E1TrackingStage]
       │  (Detection & 1:1 Temporal Tracking)
       ▼
[2. SignedPARecoveryStage]
       │  (Dynamic PA Signed Patch Recovery)
       ▼
[3. PureDOFamilyAssociationStage]
       │  (Direct Overlap [DO] Multi-Track Association)
       ▼
[4. FamilyBirthRadiusGateStage]
       │  (Family-level Solar Root Anchor Filter, r_min <= 30 px)
       ▼
[5. FamilySizeGateStage]
       │  (Family-level 5-Patch Size Filter & Recovery Union)
       ▼
[6. IslandFirstSegmentationStage]
       │  (Stage 26 Sub-Patch Pixel Segmentation: Z > 0, Area >= 16 px)
       ▼
[7. ExternalValidationStage]
          (Quantitative validation against SEEDS, CACTus, MVCC, Göttingen)
```

> [!IMPORTANT]
> **Causal Architecture Principle**: Individual tracks are not prematurely pruned by birth radius. All 219 Gate-OFF tracks undergo spatial Family Association first; Birth Gate and Size Gate filters are then evaluated at the consolidated Family level.

### Package Directory Structure
```text
cme_pipeline/
├── __init__.py
├── __main__.py               # CLI entrypoint (run, status)
├── core/
│   ├── config.py             # YAML configuration data models
│   ├── context.py            # Runtime environment, directory isolation, logger
│   ├── cache.py              # Local file-based StageCache manager
│   ├── stage.py              # PipelineStage abstract base class & Typed Dataclasses
│   └── pipeline.py           # Sequential pipeline orchestrator
├── stages/
│   ├── tracking/             # E1TrackingStage / AdaptiveROITrackingStage
│   ├── recovery/             # SignedPARecoveryStage
│   ├── association/          # PureDOFamilyAssociationStage
│   ├── gates/                # FamilyBirthRadiusGateStage, FamilySizeGateStage
│   ├── segmentation/         # IslandFirstSegmentationStage
│   └── validation/           # ExternalValidationStage
├── experiments/
│   ├── runner.py             # Python programmatic experiment runner
│   └── comparison.py         # Inter-run array and metric regression comparisons
└── utils/
    ├── geometry.py           # S^1 circular geometry, minimal enclosing arc, WCS rasters
    └── io.py                 # Hash fingerprinting, YAML and array I/O helpers
```

---

## 2. Stage Data Contracts

Every stage subclasses `PipelineStage` and enforces strictly typed dataclass inputs and outputs:

| Stage Name | Input Contract | Output Contract | Key Deliverables |
|:---|:---|:---|:---|
| `tracking` | `None` (Config-based loader) | `TrackingOutput` | `manifest_df`, `patch_component_labels` (32, 64, 64), `track_states_df` (219 tracks), `geometry` |
| `recovery` | `TrackingOutput` | `RecoveryOutput` | `recovery_func` (Dynamic PA recovery operator) |
| `association` | `TrackingOutput` / `RecoveryOutput` | `AssociationOutput` | `track_to_family`, `family_members`, `edges_df` (222 candidate edges), `track_membership_df` |
| `birth_gate` | `AssociationOutput` | `BirthGateOutput` | `birth_gated_family_ids` (23 admitted families), `rejected_family_reasons` |
| `size_gate` | `BirthGateOutput` | `SizeGateOutput` | `admitted_family_ids` (15 families), `admitted_track_ids` (37 tracks), `recovered_patch_masks` (**5,174 patches**) |
| `segmentation`| `SizeGateOutput` | `SegmentationOutput` | `final_pixel_segmentation_union` (**50,665 pixels**), `framewise_pixel_summary_df` |
| `validation` | `SegmentationOutput` | `ValidationOutput` | `comparison_metrics_df` (Catalog agreement metrics across 4 catalogs) |

---

## 3. How to Run

### 3.1 CLI Execution
```bash
python -m cme_pipeline run --config configs/baseline_rule_b.yaml
```

### 3.2 Resuming from a Specific Stage (`--from-stage`)
Reuses cached outputs for upstream stages and recomputes only the target stage and downstream stages:
```bash
python -m cme_pipeline run --config configs/baseline_rule_b.yaml --from-stage association
```

### 3.3 Checking Cache Validity (`status`)
Inspects stage cache fingerprints, timestamps, and validity:
```bash
python -m cme_pipeline status --config configs/baseline_rule_b.yaml
```
