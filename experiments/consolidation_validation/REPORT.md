# HTM / Gaussian / RET Research Consolidation

## Scope

Consolidation of four research branches into experiment/ret.

Validated integration commit: 96f5854

Archived research branches:

- archive/htm-pixel-temporal
- archive/gaussian-evidence-tracking
- archive/radial-evidence-tracking
- archive/ret-valley-symmetric

The consolidation preserves the original research history,
implementations, configurations, reports, and stored results.

## Regression Validation

### Core Pipeline

- 62 regression tests passed.
- 122 NumPy All-NaN warnings were also observed in
  the original baseline integration tests.
- No tracking algorithm or threshold changes were made.

### Gaussian Evidence Tracking

- All 40 fixed-seed synthetic comparisons matched.
- Original synthetic outcome: 35 PASS, 5 FAIL.
- Integrated synthetic outcome: 35 PASS, 5 FAIL.
- All four primary real-data tracking configurations matched
  the original numerical results within tolerance.
- Execution time was excluded from numerical comparisons.

### Radial Evidence Tracking

- All 40 fixed-seed synthetic comparisons matched.
- All four primary real-data configurations preserved
  the original numerical values.
- RET-1, RET-2, and RET-3 expose six additional
  valley-boundary diagnostic columns.
- No original columns or values were lost.

### RET-V1

- Full 63-pair experiment executed successfully.
- All 44 compared CSV, PNG, and GIF artifacts matched.
- Image comparisons used decoded pixels.
- One historical discrepancy-audit CSV was preserved;
  independent regeneration of that file was not established.
- Both prominence and symmetric-valley modes remain available.

### HTM H0-H2

- All 31 evaluation frames processed successfully.
- All five pilot and sensitivity CSV comparisons matched.
- Temperature, evidence formulation, temporal decay,
  and radial speed sweeps completed successfully.

### HTM H3-H4

- All seven tracking configurations completed successfully.
- Baseline track states and all six intervention CSVs matched.
- All original non-runtime summary metrics were preserved.
- All original valid CPA values were preserved.
- 1,488 previously missing CPA field values were populated
  in the reproduced outputs.
- No original valid CPA values were changed or lost.
- The archived and integrated HTM implementations were identical.

## Path Portability

Five RET research scripts were updated to resolve the
repository root from their own file locations.

All five scripts passed syntax validation.
Manifest and track inspection scripts were successfully
executed from outside the repository directory.

## Limitations

- External catalog validation produced zero angular IoU
  in the isolated HTM execution.
- Reference assets were subsequently restored, but catalog
  validation was not rerun after restoration.
- The original HTM summary did not contain seeds_mean_iou.
- Historical HTM CSV generation provenance could not be
  established independently from the archived source commit.
- Gaussian and RET real-data regression covered the primary
  configurations, not every historical sensitivity configuration.
- Full numerical equivalence is limited to the evaluated
  datasets, configurations, and synthetic random seed.
- Historical algorithm limitations are retained intentionally.

## Conclusion

No consolidation-induced algorithm regression was identified
in the completed comparisons.

Original results and reproduced validation outputs are
preserved separately. Future algorithm improvements must
be conducted as distinct research changes.
