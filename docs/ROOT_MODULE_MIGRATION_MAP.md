# Root-module migration map (initial review)

Generated from the 99-script inventory and direct inspection of key entrypoints.
This is a **planning map**, not evidence that files have been moved.
The 99-file inventory reported 37 already under stages, 24 needing manual
classification, 13 experiments/runners, 13 diagnostics, and 12 shared modules.
No AST parse errors were reported.

| Current root module | Proposed target/role | Migration notes |
| --- | --- | --- |
| `cadence_selection.py` | `stages/preprocessing/cadence_selection.py` | Used by live_detection and live_tracking_input; preserve import shim |
| `data_preparation.py` | `stages/preprocessing/acquisition.py` | FITS inventory/download; referenced by dual_archive/configured_run_data |
| `dual_archive.py` | `stages/preprocessing/dual_archive.py` | Acquisition/source preference; preserve source provenance |
| `north_up_preprocessing.py` | `stages/preprocessing/north_up.py` | No-MSB calibration, WCS, RD; protect scientific parity |
| `live_detection.py` | `stages/detection/live_detection.py` | D4/C3/C0 and thresholds; adapter needed |
| `p16_qc_tracking_input.py` | `stages/detection/spatial_qc.py` | QC output creation; join with detection contract |
| `live_tracking_input.py` | `stages/tracking/detection_adapter.py` | Shared loader from QC detection to tracker; preserve unavailable semantics |
| `configured_run_data.py` | `orchestration/legacy_configured_run.py` | Runner, not a scientific stage |
| `live_full_pipeline.py` | `orchestration/legacy_live_full.py` | Existing benchmark orchestration; preserve execution |
| `gate_experiment.py` | `orchestration/legacy_gates.py` | Runs association+birth+size; not an algorithm implementation |
| `run_recovery_data.py` | `orchestration/legacy_recovery.py` | Saved-gates to recovery adapter |
| `outward_split.py` | `experiments/outward_split.py` | Ablation only, never baseline |
| `detection_comparison.py` | `diagnostics/detection_comparison.py` | Comparison |
| `detection_diagnostics.py` | `diagnostics/detection.py` | Visualization |
| `gate_comparison.py` | `diagnostics/gate_comparison.py` | Comparison |
| `gate_only_visualization.py` | `diagnostics/gate_only.py` | Visualization |
| `gate_visualization.py` | `diagnostics/gates.py` | Visualization |
| `gates_only_experiment.py` | `experiments/gates_only.py` | Historical experiment |
| `inner_outer_diagnostics.py` | `diagnostics/inner_outer.py` | Diagnostic |
| `p16_count_adjusted_concentration.py` | `experiments/p16_count_adjusted_concentration.py` | Experimental QC metric |
| `p16_normalized_qc_review.py` | `diagnostics/p16_normalized_qc_review.py` | Review |
| `p16_spatial_diagnostics.py` | `diagnostics/p16_spatial.py` | P16 metrics calculator; may become a detection QC substep |
| `p16_spatial_qc_review.py` | `diagnostics/p16_spatial_qc_review.py` | Review |
| `p16_spatial_review.py` | `diagnostics/p16_spatial_review.py` | Review |
| `p16_three_variances.py` | `experiments/p16_three_variances.py` | Metric comparison |
| `rd_diagnostics.py` | `diagnostics/rd.py` | Visualization |
| `tracking_comparison.py` | `diagnostics/tracking_comparison.py` | Comparison |
| `tracking_diagnostics.py` | `diagnostics/tracking.py` | Visualization |
| `script_inventory.py` | `tools/script_inventory.py` | Development tool |
| `stage_info.py` | `tools/stage_info.py` | Registry/contract inspection |
| `__main__.py` | Retain root CLI | Stable entrypoint |
| `__init__.py` | Retain package root | Package metadata |

## Migration safety requirements

1. Keep the current root module as a thin compatibility wrapper when moving source.
2. Search and update all imports and runtime string-based entrypoints.
3. Do not turn visualization or one-off experiments into pluggable science algorithms.
4. Preserve No-MSB numerical outputs, D4 absolute scores, QC observation status, A5b ownership masks, and final gate decisions.
5. Run test suite and old-vs-new science artifact parity checks *before* deleting any old path.
6. Do not modify or reorganize old `runs/` during migration.

## Next engineering milestone

Implement the new orchestrator as a thin set of explicit adapters for steps 01–05;
run a fresh standalone nine-stage-layout run up to gates, and compare with the
saved Strict Continuity baseline. Do not yet advertise `run --until` as
operational. Remaining stages 06–09 are separate integration checkpoints.
