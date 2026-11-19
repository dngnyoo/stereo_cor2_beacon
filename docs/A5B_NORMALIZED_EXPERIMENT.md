# A5b-N Candidate Score Experiment

This is an **experimental variant** (`tracking.method: a5b_n`) and **does not
change** `a5b` / `v07_pa_width_weighted_cpa.py`.

## Isolated change

At Step 2, candidate scores are recomputed from the **same original signals**:

- `O`: original direct/adjacent overlap score, clipped to [0,1]
- `A`: original angular similarity (already in [0,1])
- `K`: original outward-consistency heuristic (already in [0,1])
- `F`: current front radius / verified geometry FOV max radius, clipped [0,1]

For direct/adjacent contact:
`S = (O * A * K * F) ** (1/4)`.

For PA/radial-corridor-only candidates (no contact):
`S = (A * K * F) ** (1/3)`.
Excluding O in this case is **intentional**, since zero contact is allowed
by baseline matching; including O would force zero match priority.

No manually assigned score weights and no absolute score threshold.
**The scores are ranking heuristics, not probabilities**.
Although [0,1] bounded, the original overlap / front definitions are not
statistically calibrated.

## Deliberately unchanged

Candidate eligibility; 8-connected masks; Step 3 PA-width-normalized contested
ownership; Step 4 continuity and priority ordering; birth and termination;
outward gate; EMA; and downstream stages.

**Important:** The modified Step 2 score can change selection priority only
where competitors require a score-based decision; direct overlapping-track
contests are still divided by the original PA-width rule. Do not infer that
all ownership is now product-optimized.

## Reproduce

```bash
python -m unittest tests.test_a5b_normalized_scoring
python -m cme_pipeline run-standard \
  --config configs/20101212_a5b_normalized.yaml \
  --until validation --visualize all \
  --run-name exp_20101212_a5b_n_001
```

Compare against `configs/20101212_full.yaml` on the same data and independent
run names. A5b-NR (redefined spatial / temporal / detection evidence) is
**not implemented here** and should be tested separately after this ablation.
