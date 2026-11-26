# RET-V1: Symmetric Valley-Based Angular Lobe Boundary Experiment Report

**Date:** 2026-11-26  
**Author:** Antigravity / CME Research Pipeline  
**Target Sequence:** 2010-12-12, STEREO-A COR2 (63 image pairs, 00:24 UTC to 23:54 UTC)  
**Parent Branch:** `experiment/radial-evidence-tracking` (`73f2b02`)  
**Active Experimental Branch:** `experiment/ret-valley-symmetric`  
**Commit Hash:** `0e27075`  
**Worktree:** `/home/aababc/stereo_cor2_retv`  

---

## 1. Executive Summary & Key Results

Radial Evidence Tracking (**RET**) was introduced to discover and track Coronal Mass Ejections (CMEs) directly from polar-accumulated spatial evidence rays without relying on fragmented connected components. However, on the 2010-12-12 STEREO-A COR2 benchmark sequence, baseline **RET-B0** experienced a catastrophic tracking failure on the primary northeast CME:
- **Track 42 Failure in Baseline (RET-B0):** In Pair 36 (06:54 UTC), prominence-based lobe width estimation with 1.5× expansion dilated the angular lobe to the maximum cap of 120.0° ([340°, 100°]). Integrating over this excessively broad sector captured diffuse northern coronal structure, artificially inflating the estimated radial front from 38.08 px to 62.20 px (\(v_r = +48.24\text{ px/hr}\)). In subsequent frames (Pairs 37 and 38), the candidate front settled back to its true physical boundary (52.15 px and 59.19 px), triggering inward velocity gate violations (\(v_r = -20.10\text{ px/hr}\) and \(-6.03\text{ px/hr} < -6.0\text{ px/hr}\)). Track 42 was terminated after just 4 frames (1.5 hours), leaving the visually prominent CME structure orphaned for the remainder of the transit (\(r > 48\text{ px}\) birth limit).

**RET-V1 (Symmetric Valley-Based Lobe)** replaces the prominence-based boundary width with an evidence-driven, circular valley detection algorithm that bounds each angular lobe using the nearest inter-peak valley distance \(h = \min(d_L, d_R)\) and defines a symmetric lobe width \(W = 2h\).

### Core Quantitative Achievements
1. **Spike Elimination in Pair 36:** In RET-V1, detected valleys at \(\theta_L = 11.0^\circ\) (\(d_L = 29.0^\circ\)) and \(\theta_R = 70.0^\circ\) (\(d_R = 30.0^\circ\)) define a tightly focused symmetric lobe of width \(W = 58.0^\circ\) (\([11.0^\circ, 69.0^\circ]\)). This bounded integration eliminates northern background contamination, yielding an observed front radius of **\(r_{\text{front}} = 43.10\text{ px}\)** (a physically consistent advance of \(+6.02\text{ px}\), \(v_r = +12.06\text{ px/hr}\)). The spurious 62.2 px spike is completely eliminated.
2. **Full Physical Lifecycle Continuity:** Under RET-V1, the 46° CME (Track 41) is tracked **continuously across 16 consecutive image pairs (Pairs 33 to 48, 05:24 UTC to 12:54 UTC, total duration 7.5 hours)** from initial near-Sun detection at \(r = 34.06\text{ px}\) to clean exit at the outer field-of-view limit (\(r = 122.50\text{ px}\)), with a steady mean radial propagation speed of \(11.79\text{ px/hr}\) (\(\approx 300\text{ km/s}\)) and mean CPA jitter of \(5.11^\circ\).
3. **Full-Sequence Performance Gains:** Across all 63 frames, mean angular lobe width decreased by 21.4% (from 42.24° to 33.22°), and mean motion prediction residual improved by 21.0% (from 17.89 px to 14.13 px).
4. **Complete Synthetic & Regression Pass:** All 10 dedicated synthetic unit tests in `tests/test_ret_v1_valley.py` pass (10/10), and all 10 standard synthetic benchmark scenarios in `synthetic_suite.py` pass across all variants (40/40 test cells, 100% regression pass rate).

---

## 2. Root Cause Analysis of Baseline Failure (RET-B0)

In RET-B0, candidate angular lobes were extracted using `scipy.signal.peak_widths` evaluated at 70% relative height, scaled by an empirical expansion factor of 1.5×, and clamped to [8°, 120°]:

```python
# Baseline RET-B0 Lobe Width Estimation
widths, _, left_ips, right_ips = peak_widths(a_ext, peaks, rel_height=0.70)
lobe_width = min(max_lobe_width_deg, max(min_lobe_width_deg, width_deg * 1.5))
```

### The Pair 36 Failure Mechanism
In Pair 36 (06:54 UTC), the angular profile exhibited multiple peaks. The primary CME peak was centered at \(\mathrm{PA} \approx 40^\circ\). Due to broad shoulders and diffuse elevated background toward the North (\(330^\circ - 360^\circ\)), the 70% relative height contour extended over a wide angular span. Multiplying by 1.5× saturated the maximum width clamp:
\[
W_{\mathrm{B0}} = 120.0^\circ \quad \implies \quad \mathrm{PA} \in [340.0^\circ, 100.0^\circ]
\]

When computing the radial evidence profile \(\bar{E}(r)\) by integrating radial rays across \([340^\circ, 100^\circ]\), the integration incorporated diffuse streamer activity and northern transients situated at \(r \approx 60 - 70\text{ px}\). Consequently, the supported radial front \(r_{\text{front}}\) leaped from 38.08 px (Pair 35) to 62.20 px (Pair 36), implying an instantaneous velocity of \(+48.24\text{ px/hr}\) (\(\approx 1200\text{ km/s}\)).

In Pair 37 (07:24 UTC), the northern diffuse background subsided, and the angular lobe width contracted to 65.48°, yielding a physical front at 52.15 px. However, the tracker evaluated:
\[
dr = 52.15 - 62.20 = -10.05\text{ px} \quad \implies \quad v_r = \frac{-10.05\text{ px}}{0.5\text{ hr}} = -20.10\text{ px/hr}
\]
Because \(v_r < -6.0\text{ px/hr}\) (the inward velocity threshold `min_radial_velocity_px_hr`), the candidate was rejected as an unphysical inward collapse. Track 42 missed the frame (\(M=1\)).

In Pair 38 (07:54 UTC), Candidate 1 had \(r = 59.19\text{ px}\). Comparing against Track 42's last observation (62.20 px), \(dr = -3.01\text{ px}\). As detailed in Section 7, time-step discrepancies in velocity calculation produced \(v_r = -6.03\text{ px/hr} < -6.0\text{ px/hr}\), causing a second inward gate rejection. With \(M = 2 \ge \text{max\_consecutive\_misses}\), Track 42 was permanently terminated.

Because subsequent candidate fronts were already at \(r \approx 66 - 122\text{ px}\), none could initiate a new track under the near-Sun birth limit (\(r \le 48.0\text{ px}\)). The primary CME became completely untracked.

---

## 3. RET-V1 Architecture & Implementation

### 3.1 Algorithm Formulation
RET-V1 replaces heuristic relative-height expansion with a symmetric, valley-bounded formulation:

1. **Circular Valley Finding:** Let \(\theta_p\) be the peak position angle of candidate \(k\), and let \(\theta_{p-1}\) and \(\theta_{p+1}\) be the cyclically adjacent detected peak angles in \([0^\circ, 360^\circ)\).
2. **Search Intervals:** 
   \[
   \mathcal{I}_L = (\theta_{p-1}, \theta_p), \quad \mathcal{I}_R = (\theta_p, \theta_{p+1})
   \]
   Within each interval along the smoothed angular evidence array \(A_{\mathrm{smooth}}(\theta)\), locate the minimum evidence values:
   \[
   v_L = \arg\min_{\theta \in \mathcal{I}_L} A_{\mathrm{smooth}}(\theta), \quad v_R = \arg\min_{\theta \in \mathcal{I}_R} A_{\mathrm{smooth}}(\theta)
   \]
3. **Circular Distances:**
   \[
   d_L = (\theta_p - \theta_L) \pmod{360^\circ}, \quad d_R = (\theta_R - \theta_p) \pmod{360^\circ}
   \]
4. **Symmetric Nearest-Side Half-Width:**
   \[
   h = \min(d_L, d_R)
   \]
   \[
   W = 2h
   \]
   \[
   \mathrm{PA}_{\mathrm{start}} = (\theta_p - h) \pmod{360^\circ}, \quad \mathrm{PA}_{\mathrm{end}} = (\theta_p + h) \pmod{360^\circ}
   \]
5. **Robustness & Fallbacks:**
   - *Tie-Breaking:* If multiple bins attain the minimum, the bin closest to \(\theta_p\) is selected.
   - *Single Peak / Edge Cases (\(K < 2\) or adjacent peaks):* Falls back to prominence-based width with 0.75× scaling.
   - *Narrow-Lobe Rejection:* If \(W < 8.0^\circ\) (`min_lobe_width_deg`), the candidate is rejected as noise.

### 3.2 Exact Code Implementation & Line References
The algorithm is implemented in two core files on branch `experiment/ret-valley-symmetric` (commit `0e27075`):

1. **`experiments/ret_tracking/ret_evidence_engine.py`**:
   - Lines 85–94: Extended `RadialCandidate` dataclass with diagnostic valley fields (`left_valley_pa_deg`, `right_valley_pa_deg`, `left_valley_dist_deg`, `right_valley_dist_deg`, `selected_half_width_deg`, `fallback_used`).
   - Line 231: Updated signature of `detect_angular_candidates(..., lobe_width_mode: str = "prominence")`.
   - Lines 285–365: Implemented circular valley detection, symmetric half-width computation, tie-breaking, narrow-lobe rejection, and single-peak fallback.
   - Lines 375–410: Integrated symmetric bounds into candidate integration and radial profile extraction.

2. **`experiments/ret_tracking/ret_tracker.py`**:
   - Line 40: Added `lobe_width_mode: str = "prominence"` to `RETConfig`.
   - Lines 68–72: Configured `RET-B0` to enforce `lobe_width_mode = "prominence"` and `RET-V1` to enforce `lobe_width_mode = "valley_symmetric"`.
   - Line 272: Passed `self.config.lobe_width_mode` to `detect_angular_candidates()`.
   - Lines 98–104 & 555–562: Extended `RETTrackObservation` and `to_dataframe()` with valley diagnostic fields.

---

## 4. Pair 36 Deep Dive & Spike Elimination Proof

In Pair 36 (2010-12-12 06:54 UTC), the angular evidence profile features a dominant peak at \(\theta_p = 40.0^\circ\), flanked by a northern peak at \(\mathrm{PA} \approx 0.0^\circ\) and an eastern peak at \(\mathrm{PA} \approx 104.0^\circ\).

### Pair 36 Lobe Extraction Comparison
| Parameter | Baseline (RET-B0) | RET-V1 (Symmetric Valley) | Physical Meaning |
|---|---|---|---|
| **Peak Position Angle (\(\theta_p\))** | 40.0° | 40.0° | CME core direction |
| **Left Valley Position (\(\theta_L\))** | N/A (prominence) | 11.0° | Inter-peak minimum toward North |
| **Right Valley Position (\(\theta_R\))** | N/A (prominence) | 70.0° | Inter-peak minimum toward East |
| **Left Valley Distance (\(d_L\))** | N/A | 29.0° | (40.0° - 11.0°) |
| **Right Valley Distance (\(d_R\))** | N/A | 30.0° | (70.0° - 40.0°) |
| **Selected Half-Width (\(h\))** | 60.0° (clamped) | **29.0°** | \(\min(29.0^\circ, 30.0^\circ)\) |
| **Lobe Width (\(W\))** | **120.0°** | **58.0°** | Tightly bounded lobe |
| **Angular Integration Sector** | [340.0°, 100.0°] | [11.0°, 69.0°] | Strictly bounds CME flank |
| **Radial Peak Radius (\(r_{\text{peak}}\))** | 45.11 px | 36.07 px | Concentrated inner core |
| **Observed Front Radius (\(r_{\text{front}}\))** | **62.20 px** | **43.10 px** | **Spike eliminated (+6.02 px vs +24.12 px)** |
| **Instantaneous Velocity (\(v_r\))** | +48.24 px/hr | +12.06 px/hr | Physically consistent propagation |
| **Predicted Front Residual** | +22.11 px | +4.52 px | 79.6% lower prediction error |

![Pair 36 Radial Profile Comparison](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/pair_036_radial_profile_comparison.png)

### Proof of Spike Elimination
In RET-B0, integrating across [340°, 100°] pulled in radial rays between 340° and 11° where faint northern coronal features had elevated evidence at \(r \approx 62\text{ px}\). 
In RET-V1, the northern boundary is cleanly severed at \(\theta_L = 11.0^\circ\). Within [11.0°, 69.0°], the radial profile decays monotonically beyond \(r = 43.10\text{ px}\), completely eliminating the non-physical front leap.

---

## 5. Continuous CME Tracking Across Pairs 33–48

Under RET-V1, Track 41 maintains unbroken association across 16 consecutive frames (05:24 UTC to 12:54 UTC), tracking the CME from near-Sun birth to field-of-view exit.

### Candidate & Track Evolution Table (Pairs 33–48)
| Pair | UTC Time | Cand ID | Peak PA | Lobe Width | Front Radius | \(v_r\) (px/hr) | Track ID | Track Status | Association Outcome |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **33** | 05:24 | 1 | 37.0° | 46.0° | 34.06 px | +0.00 | **41** | tentative | new_track_birth |
| **34** | 05:54 | 2 | 37.0° | 50.0° | 34.06 px | +0.00 | **41** | tentative | matched_existing_track |
| **35** | 06:24 | 2 | 37.0° | 50.0° | 37.08 px | +6.03 | **41** | **confirmed** | matched_existing_track |
| **36** | 06:54 | 1 | 40.0° | 58.0° | 43.10 px | +12.06 | **41** | confirmed | matched_existing_track |
| **37** | 07:24 | 1 | 48.0° | 62.0° | 52.15 px | +18.09 | **41** | confirmed | matched_existing_track |
| **38** | 07:54 | 1 | 50.0° | 46.0° | 59.19 px | +14.07 | **41** | confirmed | matched_existing_track |
| **39** | 08:24 | 1 | 51.0° | 44.0° | 66.22 px | +14.07 | **41** | confirmed | matched_existing_track |
| **40** | 08:54 | 1 | 51.0° | 46.0° | 74.26 px | +16.08 | **41** | confirmed | matched_existing_track |
| **41** | 09:24 | 1 | 51.0° | 60.0° | 84.31 px | +20.10 | **41** | confirmed | matched_existing_track |
| **42** | 09:54 | 1 | 51.0° | 52.0° | 94.36 px | +20.10 | **41** | confirmed | matched_existing_track |
| **43** | 10:24 | 1 | 52.0° | 48.0° | 100.39 px | +12.06 | **41** | confirmed | matched_existing_track |
| **44** | 10:54 | 1 | 53.0° | 36.0° | 108.43 px | +16.08 | **41** | confirmed | matched_existing_track |
| **45** | 11:24 | 1 | 53.0° | 28.0° | 113.46 px | +10.05 | **41** | confirmed | matched_existing_track |
| **46** | 11:54 | 1 | 53.0° | 30.0° | 122.50 px | +18.09 | **41** | confirmed | matched_existing_track |
| **47** | 12:24 | 1 | 53.0° | 24.0° | 122.50 px | +0.00 | **41** | confirmed | matched_existing_track |
| **48** | 12:54 | 1 | 53.0° | 26.0° | 122.50 px | +0.00 | **41** | confirmed | outer_fov_exit |

![CME 46 deg Front Trajectory](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/cme_46deg_front_trajectory.png)
![CME 46 deg Lobe Width Trajectory](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/cme_46deg_lobe_width_trajectory.png)

In stark contrast to RET-B0 where the CME was lost at Pair 38 (\(r = 59.2\text{ px}\)), RET-V1 tracks the entire transit through the outer corona.

---

## 6. Assessment of Other CME Structures

To verify that RET-V1 does not degrade tracking on other coronal features, we evaluated the concurrent lobes at \(\approx 107^\circ\) (East limb streamer / CME) and \(\approx 276^\circ\) (West limb activity):

1. **East Limb CME / Streamer (\(\mathrm{CPA} \approx 104^\circ - 107^\circ\)):**
   - In RET-B0: Tracked as Track 41 from Pair 33 to Pair 36 (\(r_{\text{front}} = 40.1\text{ px} \to 72.3\text{ px}\)), with mean lobe width \(\approx 61^\circ\).
   - In RET-V1: Tracked as Track 40 from Pair 33 to Pair 36 (\(r_{\text{front}} = 40.1\text{ px} \to 72.3\text{ px}\)), with clean inter-peak valleys at \(\theta_L = 70.0^\circ\) and \(\theta_R = 208.0^\circ\). Trajectory and velocity were identical (\(v_r \approx 28.1\text{ px/hr}\)).
2. **West Limb Activity (\(\mathrm{CPA} \approx 278^\circ\)):**
   - In RET-B0: Tracked as Track 43 in Pair 36 (\(r_{\text{front}} = 42.1\text{ px}\), width 34.4°).
   - In RET-V1: Tracked as Track 42 in Pair 36 (\(r_{\text{front}} = 38.1\text{ px}\), width 82.0°, bounded by valleys at 222° and 319°). The track persisted stably without fragmentation.
3. **Northern Streamer Transient (\(\mathrm{CPA} \approx 0^\circ\)):**
   - Correctly segmented as an independent candidate (Candidate 5 in RET-V1, width 22.0°, bounded by valleys at 319° and 11°). It was properly decoupled from the northeast CME.

---

## 7. Velocity Gate Discrepancy & \(M > 0\) Velocity Doubling Effect

Section 9 of the evaluation investigated the velocity calculation in `ret_tracker.py` to assess whether time interval discrepancies contributed to the baseline failure:

### Source Code Inspection (`ret_tracker.py`, lines 218–222 & 373)
```python
# Frame-level cadence calculation:
dt_hours = 0.5
if self.previous_timestamp is not None and timestamp is not None:
    dt_sec = (timestamp - self.previous_timestamp).total_seconds()
    dt_hours = max(0.01, dt_sec / 3600.0)
...
# Association velocity evaluation:
dr = cand.r_front_px - last_obs.r_front_px
vr_instant = dr / dt_hours
```

### The Inward Velocity Doubling Mechanism
- `dt_hours` is computed as the elapsed time between the **current frame** and the **immediately preceding frame** (\(\Delta t_{\text{frame}} \approx 0.5\text{ hr}\)).
- When a track has missed observations (\(M > 0\)), `last_obs` was recorded \(M + 1\) frames ago (\(\Delta t_{\text{actual}} = (M + 1) \cdot \Delta t_{\text{frame}}\)).
- However, the code divides \(dr\) by \(\Delta t_{\text{frame}}\) rather than \(\Delta t_{\text{actual}}\).
- **Pair 38 Impact on Track 42:**
  - Track 42 was missed in Pair 37. Its last observation was in Pair 36 at \(r = 62.24\text{ px}\).
  - In Pair 38, Candidate 1 had \(r = 59.23\text{ px}\).
  - The actual physical displacement was \(dr = -3.01\text{ px}\) over \(\Delta t_{\text{actual}} = 1.0\text{ hr}\).
  - The true physical velocity was:
    \[
    v_{r,\text{actual}} = \frac{-3.01\text{ px}}{1.0\text{ hr}} = -3.01\text{ px/hr}
    \]
  - Since the inward gate is \(v_r \ge -6.0\text{ px/hr}\), **\(-3.01\text{ px/hr}\) would have passed the inward gate!**
  - However, because the tracker divided by \(\Delta t_{\text{frame}} = 0.5\text{ hr}\), it computed:
    \[
    v_{r,\text{tracker}} = \frac{-3.01\text{ px}}{0.5\text{ hr}} = -6.02\text{ px/hr}
    \]
  - Because \(-6.02\text{ px/hr} < -6.00\text{ px/hr}\), the candidate was rejected by a margin of 0.02 px/hr, causing immediate permanent termination (\(M = 2\)).

While RET-V1 completely prevents this condition by eliminating the preceding spike at Pair 36, this finding identifies an architectural bug in the baseline association logic that should be patched in future iterations by computing `dt = (timestamp - last_obs.timestamp).total_seconds() / 3600.0`.

---

## 8. Synthetic Unit Tests & Regression Validation

To ensure mathematical correctness and eliminate regression risks, RET-V1 was subjected to rigorous unit and synthetic suite testing:

### 8.1 Dedicated RET-V1 Unit Tests (`tests/test_ret_v1_valley.py`)
All 10 unit test specifications outlined in Section 11 were implemented and verified:
1. `test_symmetric_peaks_equal_valleys`: Symmetric peaks with equidistant valleys yield identical half-widths (\(h = 20.0^\circ, W = 40.0^\circ\)). **PASS**
2. `test_asymmetric_peaks_smaller_distance`: Asymmetric valleys (\(d_L = 30^\circ, d_R = 15^\circ\)) correctly select \(h = 15.0^\circ, W = 30.0^\circ\). **PASS**
3. `test_valley_finding_across_0_deg`: Peaks at 350° and 30° correctly handle circular wrap-around (0° / 360°). **PASS**
4. `test_flat_valley_plateau_tie_breaking`: Plateaus with multiple identical minima consistently break ties toward the peak PA. **PASS**
5. `test_single_peak_fallback`: Single peak profile (\(K=1\)) successfully activates fallback with `fallback_used=True`. **PASS**
6. `test_narrow_lobe_rejection`: Narrow peaks with \(W < 8.0^\circ\) are rejected as unphysical candidates. **PASS**
7. `test_adjacent_peaks_non_overlapping_interiors`: Adjacent peaks produce symmetric lobes whose interiors never cross into adjacent peak centers. **PASS**
8. `test_baseline_prominence_mode_preservation`: Baseline prominence mode reproduces exact baseline widths and properties. **PASS**
9. `test_candidate_valley_diagnostics_populated`: Extended valley metadata fields are properly populated on all candidates. **PASS**
10. `test_ret_tracker_integration_valley_mode`: `RETTracker` with `lobe_width_mode="valley_symmetric"` executes without errors. **PASS**

### 8.2 Standard 10-Scenario Synthetic Benchmark (`synthetic_suite.py`)
All 10 standard synthetic benchmark scenarios were executed across all variants (RET-0 through RET-3):
- Single Outward Arc, Two Distinct CPAs, Similar CPA Different Radii, Distinct Speeds, Angular Expansion, Curved Propagation, Low Evidence Contrast, Noise Spikes, Signal Dropouts, Irregular Cadence.
- **Result:** **40/40 test cells PASSED (100% pass rate)**. No regressions were introduced.

---

## 9. Full-Sequence Quantitative Comparison Table

The following table summarizes tracker performance across the complete 63-pair 2010-12-12 STEREO-A COR2 observation sequence:

| Performance Metric | Baseline (RET-B0) | RET-V1 (Symmetric Valley) | Relative Change |
|---|:---:|:---:|:---:|
| **Total Detected Candidates** | 504 | 610 | +21.0% (cleaner lobe separation) |
| **Total Initiated Tracks** | 74 | 75 | +1.4% |
| **Confirmed Tracks (\(\ge 3\) obs)** | 41 | 42 | +2.4% |
| **Mean Track Duration (frames)** | 3.44 | **4.07** | **+18.3%** |
| **Maximum Track Duration (frames)** | 12 | **16** | **+33.3%** |
| **Mean CPA Jitter (\(^\circ\))** | 5.26° | **5.11°** | **-2.9%** (higher stability) |
| **Mean Radial Velocity (\(\text{px/hr}\))** | 31.63 | 30.54 | -3.4% |
| **Mean Outward Consistency** | 0.951 | 0.918 | Stable (\(\pm 3.5\%\)) |
| **Mean Prediction Residual (\(\text{px}\))** | 17.89 | **14.13** | **-21.0% (superior accuracy)** |
| **Mean Lobe Width (\(^\circ\))** | 42.24° | **33.22°** | **-21.4% (tighter angular bounds)** |
| **Primary CME (46°) Track ID** | Track 42 | **Track 41** | Maintained |
| **Primary CME Duration (frames)** | 4 | **16** | **+300% (full FOV transit)** |
| **Primary CME Duration (hours)** | 1.5 hr | **7.5 hr** | **+400%** |
| **Primary CME Final Status** | Terminated (miss=2) | **Outer FOV Exit (122.5 px)** | **Full Recovery** |
| **Primary CME Start Radius** | 34.1 px | 34.1 px | Identical birth |
| **Primary CME End Radius** | 62.2 px | **122.5 px** | Continuous propagation |
| **Primary CME Mean Speed** | 18.76 px/hr | 11.79 px/hr | \(\approx 300\text{ km/s}\) (standard CME) |

---

## 10. Limitations & Future Directions (RET-V2)

While RET-V1 successfully resolves the angular expansion failure and enables full CME tracking, two secondary characteristics warrant future development:

1. **Symmetric Assumption on Asymmetric Structures:**
   - Real CMEs often exhibit asymmetric flanks where one side borders empty corona while the other borders an active streamer.
   - Forcing \(h = \min(d_L, d_R)\) ensures the lobe never encroaches on neighboring structures, but it may slightly under-integrate the open flank.
   - *RET-V2 Recommendation:* Implement asymmetric integration bounds \([\theta_p - d_L, \theta_p + d_R]\) weighted by local evidence gradients.
2. **Elapsed Time in Velocity Gating:**
   - As documented in Section 7, `ret_tracker.py` should explicitly derive `dt` from `(timestamp - track.last_observation.timestamp)` to prevent velocity inflation when observations are missed (\(M > 0\)).
3. **Adaptive Mid-FOV Track Relinking:**
   - In cases of severe occultation or prolonged data dropouts (\(M > 2\)), candidates appearing at \(r > 48\text{ px}\) cannot currently initiate tracks. Implementing a coasting Kalman state with relaxed radial birth gates for high-velocity mid-FOV candidates would further increase resilience.

---

## 11. Provenance, Repository State, & Generated Deliverables

### Repository & Branch Metadata
- **Repository:** `dngnyoo/stereo_cor2_beacon`
- **Worktree:** `/home/aababc/stereo_cor2_retv`
- **Branch:** `experiment/ret-valley-symmetric`
- **Parent Commit:** `73f2b02` (`feat(ret): implement independent radial evidence tracking...`)
- **Active Commit:** `0e27075` (`feat(ret): implement RET-V1 symmetric valley-based lobe boundary estimation...`)

### Deliverable Paths (Forward-Slash File Links)
1. **Diagnostic Comparison Images (Pairs 35–46):**
   - [compare_pair_035.png](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/compare_pair_035.png)
   - [compare_pair_036.png](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/compare_pair_036.png)
   - [compare_pair_037.png](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/compare_pair_037.png)
   - [compare_pair_038.png](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/compare_pair_038.png)
2. **Dedicated Diagnostic Plots:**
   - [pair_036_radial_profile_comparison.png](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/pair_036_radial_profile_comparison.png)
   - [cme_46deg_front_trajectory.png](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/cme_46deg_front_trajectory.png)
   - [cme_46deg_lobe_width_trajectory.png](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/cme_46deg_lobe_width_trajectory.png)
3. **Animated Visualizations:**
   - [ret_v1_full_sequence.gif](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/ret_v1_full_sequence.gif) (Full 63-pair sequence animation)
   - [ret_v1_pairs35_46_comparison.gif](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/ret_v1_pairs35_46_comparison.gif) (Side-by-side Pairs 35–46 animation)
4. **Metrics & Candidate Diagnostics CSVs:**
   - [full_sequence_comparison.csv](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/full_sequence_comparison.csv)
   - [pair35_46_candidate_diagnostics.csv](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/pair35_46_candidate_diagnostics.csv)
