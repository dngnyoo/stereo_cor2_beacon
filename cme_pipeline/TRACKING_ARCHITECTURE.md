# CME Pipeline — Unified Tracking Architecture & Extension Guide

This document defines the standardized architecture of the `cme_pipeline` package integrating **E1**, **A3**, **A4**, and **A5b** tracking methods into an extensible, reproducible 7-stage CME detection and segmentation pipeline.

---

## 1. End-to-End Stage Flow

```text
[Frozen Detection Loader]  (cme_pipeline/stages/detection/frozen_detection.py)
  ├─ Shared WCS GeometryContext, 32-frame Gaussian smoothed cache, thresholds
  ├─ patch_activation_grid4      : (N, 64, 64) bool
  └─ raw_component_labels        : (N, 64, 64) int32 (pure 8-connected components)
         │
         ▼
[1. Tracking Stage]  (Dispatched via cme_pipeline/stages/tracking/registry.py)
  ├─ Methods: 'e1' | 'a3' | 'a4' | 'a5b' (or legacy 'a0', 'a5', 'birth_anchored')
  ├─ patch_component_labels      : (N, 64, 64) int32 (spatial detection component IDs)
  ├─ track_ownership_masks       : (N, 64, 64) int32 (mutually exclusive Track IDs)
  ├─ track_states_df             : Track-level kinematics, PA bounds, front/state decoupling
  ├─ component_states_df         : 1:Many per-component breakdown (A4/A5b multi-component)
  └─ leading_front_states_df     : Decoupled leading front observation/prediction logs
         │
         ▼ (Normalized via adapter.adapt_tracking_output)
[2. Signed Dynamic PA Recovery]  (cme_pipeline/stages/recovery/signed_pa_recovery.py)
  └─ Uses get_track_observed_mask (track_ownership_masks == tid) strictly on OBSERVED frames
         │
         ▼
[3. Family Association Stage]  (Rule B: pure_do | Direct 1:1: identity | legacy)
  ├─ 'identity' (Recommended for A4/A5b): FAM_T{tid:02d} -> [tid]
  └─ 'pure_do'  (Required for E1 Baseline): Direct Overlap parent-child linkage
         │
         ▼
[4. Birth Radius Gate] -> [5. Size Gate (>= 5 patches) + Per-Family Recovery]
  └─ Preserves natural PA-corridor overlaps between adjacent families
         │
         ▼
[6. Island-First Pixel Segmentation] -> [7. External Catalog Validation]
  ├─ final_pixel_segmentation_union : (N, 256, 256) bool
  ├─ family_pixel_masks             : Dict[str, (N, 256, 256) bool]
  └─ build_track_level_masks(...)   : (T, N, H, W) per-track tensors & overlap stats
```

---

## 2. Core Design Principles & Semantic Contracts

### 2.1 Strictly Separated Identifiers (`Component ID` vs `Track ID`)
Previous experimental iterations (`Exp 03`–`Exp 07` pilots) stored `track_id` inside `patch_component_labels` and mapped `component_id = track_id` so downstream stages could run. That conflation has been eliminated in the core architecture:

| Artifact / Field | Exact Semantic Meaning |
| :--- | :--- |
| **`patch_component_labels`** | Raw **8-connected spatial detection component labels** (`0` = background, `1..K` = connected blobs of `patch_activation_grid4` per frame). Identical across all tracking methods for the same detection input. |
| **`track_ownership_masks`** | Mutually exclusive **Track ID assignment** (`0` = unclaimed, `1..T` = active Track ID). Aliased via property `.ownership_masks` for backwards compatibility with legacy scripts/tests. |
| **`component_states_df`** | Per-component measurements (`component_id`, `track_id`, `num_patches`, `r_min`, `r_front`, `cpa`, `is_leading_front`) preserving 1:Many components under one `track_id`. |
| **`track_states_df`** | Standardized track timeline including decoupled `state` (`OBSERVED` / `PREDICTED_ONLY`) vs `leading_front_state` (`OBSERVED` / `PREDICTED`), `birth_r_min_px`, `pa_min`, `pa_max`, `radial_front`, `num_components`. |

Downstream stages (`PureDOFamilyAssociationStage`, `AdaptiveROIAssociationStage`, `FamilySizeGateStage`, `SignedPARecoveryStage`) resolve a track's active patches through the canonical helper:
```python
from cme_pipeline.stages.tracking.adapter import get_track_observed_mask
```
which queries `(track_ownership_masks[f_idx] == tid)`, guaranteeing exact isolation even when a track holds multiple disconnected components or when A5b splits a single detection component between two competing tracks.

### 2.2 Predicted-Only & Decoupled Leading Front Safety
- When `state == 'PREDICTED_ONLY'`, `num_patches == 0`, `track_ownership_masks == tid` contains 0 patches, `pa_min`/`pa_max` are `NaN`, and `SignedPARecoveryStage` skips that frame cleanly.
- When `state == 'OBSERVED'` and `leading_front_state == 'PREDICTED'` (e.g., an inner CME root fragment is observed after the outer front fades), recovery seeds strictly originate from the observed inner root patches (`track_ownership_masks == tid`), never from synthetic pixel coordinates.

### 2.3 Preservation of Per-Track Recovery Overlaps
When multiple admitted tracks/families undergo `Signed Dynamic PA Recovery` independently, adjacent corridors (within `±5.0°` margins) can share border patches/pixels (e.g., 7,285 shared pixels on A5b, 2010-12-12). **These overlaps are intentionally preserved intact** in `family_pixel_masks` and `build_track_level_masks(...)` rather than arbitrarily dropped or reassigned to a single winner.

---

## 3. Registered Tracking & Association Algorithms

Configured via YAML or `PipelineConfig`:

| Canonical Method | Aliases | Description | Key Validated Parameters |
| :--- | :--- | :--- | :--- |
| **`e1`** | `baseline` | Original 1:1 baseline tracking (Exp 11, 219 tracks) | `patch_size: 4`, `max_cadence_gap_sec: 4500.0`, `alpha: 0.0` |
| **`a3`** | `component_first` | Component-first 1:1 adaptive matching | `outward_gate_enabled`, `outward_gate_tolerance_px`, `ema_alpha: 0.1` |
| **`a4`** | `multi_component` | 1:Many disconnected components per track + front decoupling | `outward_gate_enabled`, `outward_gate_tolerance_px`, `ema_alpha: 0.1` |
| **`a5b`** | `pa_width_weighted_cpa`, `weighted_cpa` | Preferred PA-width-normalized CPA Voronoi boundary tracking | Above + `min_pa_width_deg: 8.0` |

### Official Reference YAML Configurations (`configs/`)
- [`configs/tracking_a5b_identity.yaml`](../configs/tracking_a5b_identity.yaml): **Recommended configuration** (`method: a5b`, `outward_gate_enabled: true`, `outward_gate_tolerance_px: 6.0`, `min_pa_width_deg: 8.0`, `ema_alpha: 0.1`, `association: identity`) — yields 20 tracks, 6,549 recovered patches, 59,899 union pixels, 7,285 shared pixels.
- [`configs/tracking_a5b_rule_b.yaml`](../configs/tracking_a5b_rule_b.yaml): A5b with `pure_do` (Rule B), bitwise-identical to Identity on 2010-12-12.
- [`configs/tracking_a4_identity.yaml`](../configs/tracking_a4_identity.yaml): Validated Exp 04 A4 configuration (18 tracks).
- [`configs/tracking_a3_fixed_identity.yaml`](../configs/tracking_a3_fixed_identity.yaml): Validated Exp 04 A3-fixed configuration (50 tracks).
- [`configs/tracking_e1_rule_b.yaml`](../configs/tracking_e1_rule_b.yaml): E1 baseline reproducibility (219 tracks, 5,174 recovered patches, 50,665 pixels).

### Example Minimal Configuration
```yaml
run_name: my_a5b_run
date_str: 20101212
repo_root: /home/aababc/stereo_cor2
output_base_dir: runs

tracking:
  method: a5b
  parameters:
    min_pa_width_deg: 8.0
    outward_gate_enabled: true
    outward_gate_tolerance_px: 6.0
    ema_alpha: 0.1

association:
  method: identity
```

> **Strict Parameter Validation**: Any unknown parameter placed under `tracking.parameters` (or a method-specific parameter passed to an algorithm that does not support it, such as `min_pa_width_deg` on `e1` or `a4`) raises an explicit `ValueError` rather than being silently ignored.

---

## 4. Provenance & Reproducibility (`manifest.json`)

Every pipeline execution writes a complete provenance record to `<run_dir>/manifest.json`:
- `provenance.tracking_algorithm` and `tracking_stage_class` (with version)
- `provenance.effective_tracking_parameters` (resolved dict including defaults)
- `provenance.association_algorithm` and `association_stage_class`
- `provenance.code_revision` (`git_commit` SHA and `is_dirty`)
- `provenance.input_dataset` (source manifest/arrays paths and counts)

---

## 5. Guide: Adding a New Tracking Algorithm (e.g., A6 / A7)

To integrate a future tracking algorithm into the CME pipeline without duplicating downstream logic:

1. **Create your Stage class** under `cme_pipeline/stages/tracking/my_a6_tracking.py` inheriting from `PipelineStage`:
   - Set `name: str = "tracking"` and semantic `version: str = "6.0.0"`.
   - Accept `inputs: Optional[DetectionOutput]` in `run(...)`; if `inputs is None`, load shared detection inputs via:
     ```python
     from cme_pipeline.stages.detection.frozen_detection import load_frozen_detection_inputs
     det_out = inputs if isinstance(inputs, DetectionOutput) else load_frozen_detection_inputs(config, context)
     ```
   - Validate and resolve your parameters via:
     ```python
     effective_params = resolve_and_validate_tracking_parameters("a6", config.tracking)
     ```
   - Include `effective_params` in `compute_fingerprint(...)`.
   - Populate `patch_component_labels=det_out.raw_component_labels.copy()` (raw detection blobs) and `track_ownership_masks` (`(N, 64, 64) int32` mutually exclusive Track IDs).
   - Keep predicted-only rows strictly at `0` patches owned.
   - Populate `component_states_df` and `leading_front_states_df`, wrap the output with `adapt_tracking_output(out)` and return it.

2. **Register the method** in `cme_pipeline/stages/tracking/registry.py` (inside `_ensure_defaults_registered()` or dynamically via `register_tracking_method(...)`):
   ```python
   register_tracking_method(
       TrackingMethodSpec(
           canonical_name="a6",
           stage_cls=MyA6TrackingStage,
           aliases=("my_a6_method",),
           supported_parameters=frozenset({"ema_alpha", "outward_gate_enabled", "my_new_threshold"}),
           default_parameters={"ema_alpha": 0.1, "outward_gate_enabled": True, "my_new_threshold": 0.5},
           description="A6 Tracking Method...",
       )
   )
   ```

3. **Run immediately** via CLI or Python using `tracking: method: a6` and `association: method: identity` (or `pure_do`) without changing `pipeline.py`, `recovery`, `association`, `gates`, `segmentation`, or `validation`.

---

## 6. Backwards Compatibility with Historical Experiments (03–07)

- Legacy experiment runner scripts in `cme_pipeline/experiments/` (`run_component_first_a3_experiment.py`, `run_multicomponent_a4_experiment.py`, `run_06_a5b_experiment.py`, `run_07_*`) and historical outputs in `runs/` are preserved untouched.
- `cme_pipeline/stages/tracking/a5b_pipeline_adapter.py` remains as a deprecated wrapper delegating to `adapt_tracking_output` while preserving legacy `component_id = track_id` semantics exclusively for untouched Experiment 07 runner scripts.
- Ongoing `one_month_pipeline/` and `one_month_preprocessing_detection_survey/` workflows are strictly isolated and unaffected.

