# A5b-NR Redesigned Normalized Multiplicative Tracking Experiment

This is an **experimental variant** implementing two tracking methods:
- `tracking.method: a5b_nr1` (Phase 1: Continuous normalized evidence scoring with geometric Voronoi partitioning)
- `tracking.method: a5b_nr2` (Phase 2: Continuous normalized evidence scoring with score-aware Voronoi arbitration)

Neither variant modifies the legacy `a5b` (`v07_pa_width_weighted_cpa.py`) or `a5b_n` (`v08_normalized_a5b.py`) implementations.

## Theoretical Motivation

The predecessor experiment A5b-N revealed:
1. **Multiplicative Zero-Collapse**: 29/60 (48.3%) of candidates collapsed to zero due to rigid step-function thresholds in angular consistency ($|\Delta\text{CPA}| \ge 35^\circ$) and kinematic consistency ($r_{\text{curr}} \le r_{\text{prev}}$).
2. **Clipping Saturation**: 14/60 (23.3%) of overlap scores exceeded 1.0 (saturating at 1.0) due to unnormalized dilated patch unions.
3. **Contested Ownership Decoupling**: Candidate scores were purely used for iteration ordering; contested component ownership was decided by purely geometric PA-width Voronoi distance, making final masks insensitive to tracking confidence.

## Mathematical Redesign

All candidate scores are continuous, strictly bounded in $(0, 1]$, and non-saturating:

$$S_{NR} = (C \cdot A \cdot K \cdot D)^{1/4} \in (0, 1]$$

### 1. Spatial Continuity ($C$)
- **Direct Overlap**: Symmetric Sørensen–Dice coefficient $C = \frac{2|P_c \cap P_t|}{|P_c| + |P_t|} \in (0, 1]$.
- **Boundary Adjacency**: Fractional perimeter contact length $C = \operatorname{clip}(0.20 + 0.30 \cdot \frac{L_{\text{contact}}}{L_{\text{perim}}}, 0.20, 0.50)$.
- **Non-Contact Spatial Proximity**: Gaussian spatial distance kernel $C = \exp(-d_{\text{min}}^2 / 2\sigma_s^2)$ with $\sigma_s = 50\text{ px}$.
- Floor: $\epsilon_C = 0.01$.

### 2. Angular Consistency ($A$)
- Circular angular difference: $\Delta\text{CPA} = \operatorname{circ\_diff}(\text{CPA}_c, \text{CPA}_t) \in [0^\circ, 180^\circ]$.
- Circular Gaussian distribution: $A = \exp(-\Delta\text{CPA}^2 / 2\sigma_{\text{PA}}^2)$ where $\sigma_{\text{PA}} = \max(15.0^\circ, W_t / 2)$.
- Floor: $\epsilon_A = 0.01$.

### 3. Kinematic Consistency ($K$)
- Causal expansion velocity $\hat{v}_r$ estimated from median of recent causal observation intervals, clamped to $[100, 2500]\text{ km/s}$ ($[0.005, 0.125]\,R_\odot/\text{min}$). Defaults to $450\text{ km/s}$ if $< 2$ history points.
- Predicted front: $\hat{r}_t = r_{\text{prev}} + \hat{v}_r \cdot \Delta t$.
- Residual: $\Delta r = r_c - \hat{r}_t$.
- Kernel: $K = \exp(-(\Delta r)^2 / 2\sigma_r^2)$ where $\sigma_r = \max(20\text{ px}, 0.15 \cdot r_{\text{prev}})$. Inward motion is smoothly damped rather than hard-zeroed.
- Floor: $\epsilon_K = 0.01$.

### 4. Detection Confidence ($D$)
- Mean D4 patch z-score excess above detection threshold: $E = \frac{1}{|P_c|} \sum_{p \in P_c} \max(0, z_p - z_{\text{thresh}})$.
- Exponential saturation: $D = 1 - \exp(-E / \sigma_D)$ with $\sigma_D = 2.0$.
- Floor: $\epsilon_D = 0.05$.

## Phase 2: Score-Aware Voronoi Arbitration (`a5b_nr2`)

For contested components claimed by multiple tracks, individual patches $p$ are arbitrated using score-modulated effective angular distance:

$$d_{\text{eff}}(p, i) = \frac{|\operatorname{circ\_diff}(\text{PA}_p, \text{CPA}_i)|}{W_i \cdot \left(S_{NR, i}\right)^\beta}, \quad \beta = 0.5$$

Tie-breaker: higher $S_{NR, i}$, earlier birth frame, lower Track ID.

## Reproduction & Verification

```bash
# Run unit and tracker test suites
pytest tests/test_a5b_nr_scoring.py tests/test_a5b_nr_tracker.py

# Run standard NR1 pipeline
python -m cme_pipeline run-standard \
  --config configs/20101212_a5b_nr1.yaml \
  --run-name exp_20101212_a5b_nr1

# Run score-aware NR2 pipeline
python -m cme_pipeline run-standard \
  --config configs/20101212_a5b_nr2.yaml \
  --run-name exp_20101212_a5b_nr2

# Evaluate comparative metrics
python scripts/compare_b0_b1_nr1_nr2.py
python scripts/evaluate_frozen_catalog_nr.py
```
