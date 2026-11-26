# Technical Investigation Report: Pair 39 Visualization Inconsistencies & Corrective Resolution

**Date:** 2026-11-26  
**Sequence:** 2010-12-12, STEREO-A COR2, Pair 39 (08:24 UTC, Frame Index 38)  
**Branch:** `experiment/ret-valley-symmetric`  
**Parent Commit:** `f4169cc`  
**Worktree:** `/home/aababc/stereo_cor2_retv`  

---

## 1. Executive Summary & Root Cause Confirmation

A rigorous investigation was conducted into the two visual anomalies observed in `compare_pair_039.png` from the initial RET-V1 evaluation:
1. **Track 45 anomaly:** Displayed in the RET-V1 tracking overlay at $\mathrm{CPA} \approx 230^\circ$, $r_{\text{front}} = 107\text{ px}$, even though no candidate existed near $230^\circ$ in Pair 39's angular evidence profile.
2. **Track 41 anomaly:** Displayed in Pair 39 at $\mathrm{CPA} \approx 53^\circ$, $r_{\text{front}} = 122\text{ px}$, despite the evaluation report stating that the CME front reaches $122.5\text{ px}$ at Pair 48.

### Confirmed Root Cause
The tracking algorithm, candidate detection, and Hungarian bipartite association are **100% numerically correct and completely unaffected**. 

The root cause was isolated entirely to a **visualization reference bug on shared mutable track objects**:
- In `experiments/ret_tracking/ret_tracker.py` (lines 316–327), `FrameRETResult.active_confirmed_tracks` stored references to the live `RETTrack` objects in the tracker's internal dictionary `self.tracks`.
- In `experiments/ret_tracking/run_ret_v1_experiment.py`, the image rendering functions (`plot_side_by_side_comparison`, `plot_single_variant_panel`, and `generate_animated_gifs`) were executed in a post-processing loop **after the full 63-frame tracking sequence had completed**.
- The overlay rendering code accessed each track's state via:
  ```python
  for trk in res.active_confirmed_tracks:
      obs = trk.latest
  ```
- Because `trk.latest` is a property returning `self.observations[-1]`, reading `trk.latest` post-simulation retrieved the track's **terminal observation at the end of its multi-hour lifespan**, rather than its observation at that historical frame!
  - **Track 41:** In Pair 39 (08:24 UTC), Track 41 was matched to Candidate 1 at **$\mathrm{CPA} = 48.45^\circ$, $r_{\text{front}} = 66.22\text{ px}$**. However, Track 41 continued propagating through Pair 48 (12:54 UTC), where its terminal observation reached $\mathrm{CPA} = 53.33^\circ$, $r_{\text{front}} = 122.50\text{ px}$. `trk.latest` mistakenly plotted its Pair 48 terminal state into Pair 39 (+56.28 px error).
  - **Track 45:** In Pair 39 (08:24 UTC), Track 45 was matched to Candidate 4 at **$\mathrm{CPA} = 198.18^\circ$, $r_{\text{front}} = 32.05\text{ px}$**. Track 45 continued propagating southwest across 14 frames until Pair 51 (14:24 UTC), where its terminal observation reached $\mathrm{CPA} = 230.11^\circ$, $r_{\text{front}} = 107.43\text{ px}$. `trk.latest` mistakenly plotted its Pair 51 terminal state into Pair 39 (+31.94° CPA and +75.38 px error).

The quantitative evaluation (`pair35_46_candidate_diagnostics.csv` and `full_sequence_comparison.csv`) and trajectory plots (`cme_46deg_front_trajectory.png`) were **unaffected** because they explicitly filtered observations by `obs.frame_idx == idx`.

---

## 2. Pair 39 Current Angular Candidates (Runtime State)

In Pair 39 (frame index 38, 2010-12-12 08:24:24 UTC), the RET-V1 evidence engine detected exactly 5 angular candidates:

| Candidate ID | Peak PA | Weighted CPA | PA Start | PA End | Lobe Width | Radial Peak ($r_{\text{peak}}$) | Front Radius ($r_{\text{front}}$) | Peak Evidence | Total Lobe Evidence |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | $51.0^\circ$ | **$48.45^\circ$** | $29.0^\circ$ | $73.0^\circ$ | $44.0^\circ$ | $37.08\text{ px}$ | **$66.22\text{ px}$** | 179.02 | 3908.33 |
| **2** | $106.0^\circ$ | **$105.44^\circ$** | $73.0^\circ$ | $139.0^\circ$ | $66.0^\circ$ | $58.18\text{ px}$ | **$95.36\text{ px}$** | 40.17 | 1766.91 |
| **3** | $278.0^\circ$ | **$278.92^\circ$** | $251.0^\circ$ | $305.0^\circ$ | $54.0^\circ$ | $42.10\text{ px}$ | **$42.10\text{ px}$** | 17.33 | 422.13 |
| **4** | $199.0^\circ$ | **$198.18^\circ$** | $173.0^\circ$ | $225.0^\circ$ | $52.0^\circ$ | $27.02\text{ px}$ | **$32.05\text{ px}$** | 19.28 | 698.00 |
| **5** | $323.0^\circ$ | **$323.09^\circ$** | $316.0^\circ$ | $330.0^\circ$ | $14.0^\circ$ | $77.28\text{ px}$ | **$77.28\text{ px}$** | 6.73 | 65.69 |

- **Candidate 1:** Corresponds to the primary expanding northeast CME structure ($r_{\text{front}} = 66.22\text{ px}$).
- **Candidate 2:** Corresponds to the East limb streamer / CME structure ($r_{\text{front}} = 95.36\text{ px}$).
- **Candidate 3:** Corresponds to West limb coronal activity ($r_{\text{front}} = 42.10\text{ px}$).
- **Candidate 4:** Corresponds to the nascent Southwest propagating structure ($r_{\text{front}} = 32.05\text{ px}$).
- **Candidate 5:** A faint northern transient at $323^\circ$. Rejected from birth because $r_{\text{front}} = 77.28\text{ px} > 48.0\text{ px}$ (`r_birth_max`).

---

## 3. Pair 39 Displayed Track Audit & Discrepancy Breakdown

The following table compares what was displayed by the buggy visualization code versus the true runtime state for every track active in Pair 39:

| Track ID | Matched Cand ID | Status in Frame | Missed Count | Last Success Timestamp (UTC) | Buggy Displayed CPA | Buggy Displayed $r_{\text{front}}$ | Buggy Source Observation | True Frame CPA | True Frame $r_{\text{front}}$ | True Frame Source Observation | CPA Discrepancy | Radius Discrepancy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **40** | 2 | confirmed | 0 | 08:24:24 | $105.44^\circ$ | $95.36\text{ px}$ | Terminal state at Pair 39 (frame 38) | $105.44^\circ$ | $95.36\text{ px}$ | Current matched obs (Cand 2) | $+0.00^\circ$ | $+0.00\text{ px}$ |
| **41** | 1 | confirmed | 0 | 08:24:24 | **$53.33^\circ$** | **$122.50\text{ px}$** | Terminal state at Pair 48 (frame 47) | **$48.45^\circ$** | **$66.22\text{ px}$** | Current matched obs (Cand 1) | **$+4.88^\circ$** | **$+56.28\text{ px}$** |
| **44** | 3 | confirmed | 0 | 08:24:24 | $277.54^\circ$ | $50.14\text{ px}$ | Terminal state at Pair 40 (frame 39) | $278.92^\circ$ | $42.10\text{ px}$ | Current matched obs (Cand 3) | $-1.38^\circ$ | $+8.04\text{ px}$ |
| **45** | 4 | confirmed | 0 | 08:24:24 | **$230.11^\circ$** | **$107.43\text{ px}$** | Terminal state at Pair 51 (frame 50) | **$198.18^\circ$** | **$32.05\text{ px}$** | Current matched obs (Cand 4) | **$+31.94^\circ$** | **$+75.38\text{ px}$** |

### Numerical Audit Observations
1. **Track 40:** Coincidentally showed zero discrepancy because Pair 39 happened to be Track 40's final observation before exiting or terminating.
2. **Track 41:** Displayed at the outer boundary ($122.5\text{ px}$, Pair 48) instead of its true position at $66.22\text{ px}$, explaining why the overlay showed the CME at the outer boundary 9 frames prematurely!
3. **Track 45:** Displayed at CPA $230.1^\circ$, $r = 107.4\text{ px}$ (its Pair 51 terminal position) instead of its true position at CPA $198.2^\circ$, $r = 32.1\text{ px}$. This explains why no candidate existed at $230^\circ$ in Pair 39's angular profile: the CME structure was physically at $198^\circ$ in Pair 39 and had not yet rotated or curved to $230^\circ$!

---

## 4. Full Lifecycle Trace of Tracks 41 and 45 (Pairs 35–46)

The table below traces the exact runtime state of Track 41 (the primary 46° CME) and Track 45 (the Southwest structure) across the mandatory evaluation window:

| Pair | UTC Time | Track 41 Cand | Track 41 CPA | Track 41 $r_{\text{front}}$ | Track 41 $v_r$ | Track 41 Status | Track 45 Cand | Track 45 CPA | Track 45 $r_{\text{front}}$ | Track 45 $v_r$ | Track 45 Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **35** | 06:24 | Cand 2 | $38.94^\circ$ | $37.08\text{ px}$ | $+6.03\text{ px/hr}$ | confirmed | — | — | — | — | not born |
| **36** | 06:54 | Cand 1 | $43.39^\circ$ | $43.10\text{ px}$ | $+12.06\text{ px/hr}$ | confirmed | — | — | — | — | not born |
| **37** | 07:24 | Cand 1 | $44.85^\circ$ | $52.15\text{ px}$ | $+18.09\text{ px/hr}$ | confirmed | — | — | — | — | not born |
| **38** | 07:54 | Cand 1 | $47.14^\circ$ | $59.19\text{ px}$ | $+14.07\text{ px/hr}$ | confirmed | Cand 4 | $203.65^\circ$ | $28.03\text{ px}$ | $+0.00\text{ px/hr}$ | tentative (birth) |
| **39** | 08:24 | Cand 1 | **$48.45^\circ$** | **$66.22\text{ px}$** | $+14.07\text{ px/hr}$ | confirmed | Cand 4 | **$198.18^\circ$** | **$32.05\text{ px}$** | $+8.04\text{ px/hr}$ | confirmed (hit 2) |
| **40** | 08:54 | Cand 1 | $48.73^\circ$ | $74.26\text{ px}$ | $+16.08\text{ px/hr}$ | confirmed | Cand 3 | $208.44^\circ$ | $37.08\text{ px}$ | $+10.05\text{ px/hr}$ | confirmed |
| **41** | 09:24 | Cand 1 | $47.68^\circ$ | $84.31\text{ px}$ | $+20.10\text{ px/hr}$ | confirmed | Cand 3 | $208.19^\circ$ | $44.11\text{ px}$ | $+14.07\text{ px/hr}$ | confirmed |
| **42** | 09:54 | Cand 1 | $47.72^\circ$ | $94.36\text{ px}$ | $+20.10\text{ px/hr}$ | confirmed | Cand 3 | $208.82^\circ$ | $50.14\text{ px}$ | $+12.06\text{ px/hr}$ | confirmed |
| **43** | 10:24 | Cand 1 | $47.75^\circ$ | $100.39\text{ px}$ | $+12.06\text{ px/hr}$ | confirmed | Cand 4 | $213.62^\circ$ | $59.19\text{ px}$ | $+18.09\text{ px/hr}$ | confirmed |
| **44** | 10:54 | Cand 1 | $49.14^\circ$ | $108.43\text{ px}$ | $+16.08\text{ px/hr}$ | confirmed | Cand 4 | $209.29^\circ$ | $62.20\text{ px}$ | $+6.03\text{ px/hr}$ | confirmed |
| **45** | 11:24 | Cand 1 | $50.07^\circ$ | $113.46\text{ px}$ | $+10.05\text{ px/hr}$ | confirmed | Cand 3 | $210.08^\circ$ | $69.24\text{ px}$ | $+14.07\text{ px/hr}$ | confirmed |
| **46** | 11:54 | Cand 1 | $50.21^\circ$ | $122.50\text{ px}$ | $+18.09\text{ px/hr}$ | confirmed | Cand 4 | $219.26^\circ$ | $76.27\text{ px}$ | $+14.07\text{ px/hr}$ | confirmed |

---

## 5. Numerical Agreement Verification

We evaluated the exact numerical difference between each currently matched track's observation and its assigned candidate in Pair 39:
- **Track 40 $\leftrightarrow$ Candidate 2:**
  - Candidate: $\mathrm{CPA} = 105.442642^\circ$, $r_{\text{front}} = 95.364998\text{ px}$
  - Track Obs: $\mathrm{CPA} = 105.442642^\circ$, $r_{\text{front}} = 95.364998\text{ px}$
  - Absolute Difference: $\Delta \mathrm{CPA} = 0.000000^\circ$, $\Delta r = 0.000000\text{ px}$ (**Exact Match**)
- **Track 41 $\leftrightarrow$ Candidate 1:**
  - Candidate: $\mathrm{CPA} = 48.451557^\circ$, $r_{\text{front}} = 66.220001\text{ px}$
  - Track Obs: $\mathrm{CPA} = 48.451557^\circ$, $r_{\text{front}} = 66.220001\text{ px}$
  - Absolute Difference: $\Delta \mathrm{CPA} = 0.000000^\circ$, $\Delta r = 0.000000\text{ px}$ (**Exact Match**)
- **Track 44 $\leftrightarrow$ Candidate 3:**
  - Candidate: $\mathrm{CPA} = 278.920013^\circ$, $r_{\text{front}} = 42.099998\text{ px}$
  - Track Obs: $\mathrm{CPA} = 278.920013^\circ$, $r_{\text{front}} = 42.099998\text{ px}$
  - Absolute Difference: $\Delta \mathrm{CPA} = 0.000000^\circ$, $\Delta r = 0.000000\text{ px}$ (**Exact Match**)
- **Track 45 $\leftrightarrow$ Candidate 4:**
  - Candidate: $\mathrm{CPA} = 198.175079^\circ$, $r_{\text{front}} = 32.049999\text{ px}$
  - Track Obs: $\mathrm{CPA} = 198.175079^\circ$, $r_{\text{front}} = 32.049999\text{ px}$
  - Absolute Difference: $\Delta \mathrm{CPA} = 0.000000^\circ$, $\Delta r = 0.000000\text{ px}$ (**Exact Match**)

**Sequence-Wide Verification:** An automated assertion across all 63 observation frames confirmed that **100% of matched track observations agree with their assigned candidates to within $10^{-5}$**.

---

## 6. Visualization Bug Fix & Deliverable Regeneration

The visualization code in `experiments/ret_tracking/run_ret_v1_experiment.py` was patched with the following architectural corrections:

1. **Frame-Specific Observation Extraction:**
   Replaced all occurrences of `obs = trk.latest` with an explicit query for the observation recorded at `frame_idx`:
   ```python
   cur_obs = [o for o in trk.observations if o.frame_idx == frame_idx]
   if cur_obs:
       obs = cur_obs[0]
       # Render currently matched observation at its true historical CPA and r_front
   ```
2. **Clear Distinction of Track & Candidate States:**
   - **Currently Matched Tracks:** Plotted with a solid cyan/green ray to `obs.cpa_deg`, green triangle at `obs.r_front_px`, and label `Trk {id} (C{cid})\nCPA {cpa}° | r={r}px`.
   - **Matched Candidate Lobe Fronts:** Plotted as solid lime circular arcs across the lobe width with a green square marker at $(x_{\text{front}}, y_{\text{front}})$.
   - **Unmatched / Rejected Candidates:** Plotted as dashed crimson arcs with a crimson 'x' marker and pink annotation box indicating rejection reason (e.g., `Rej C5 (323°, r=77px)`).
   - **Temporarily Missed Tracks ($M > 0$):** Plotted with an amber dotted ray to `last_obs.cpa_deg`, hollow orange marker at the predicted radius $r_{\text{pred}} = r_{\text{prev}} + v_r \cdot \Delta t$, and label `MISS Trk {id}\nPred r={r_pred}px (M={m})`.

3. **Complete Asset Regeneration:**
   All deliverables were re-rendered using the corrected visualization engine:
   - 12 side-by-side comparison panels: `compare_pair_035.png` through `compare_pair_046.png`
   - 12 RET-V1 diagnostic frames: `ret_v1/frames/diag_pair_035.png` through `046.png`
   - 12 Baseline diagnostic frames: `baseline/frames/diag_pair_035.png` through `046.png`
   - Animated GIFs: `ret_v1_pairs35_46_comparison.gif` and `ret_v1_full_sequence.gif`
   - Diagnostic Audit CSV: `metrics/pair39_display_discrepancy_audit.csv`

---

## 7. Deliverable Links & Repository Metadata

- **Worktree:** `/home/aababc/stereo_cor2_retv`
- **Branch:** `experiment/ret-valley-symmetric`

### File Links
1. **Corrected Pair 39 Comparison Panel:**
   - [compare_pair_039.png](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/compare_pair_039.png) (Track 41 at $48^\circ, 66\text{ px}$; Track 45 at $198^\circ, 32\text{ px}$; Rej C5 at $323^\circ, 77\text{ px}$)
2. **Corrected Pair 36 Comparison Panel:**
   - [compare_pair_036.png](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/compare_pair_036.png)
3. **Audit CSV:**
   - [pair39_display_discrepancy_audit.csv](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/pair39_display_discrepancy_audit.csv)
4. **Corrected Comparison Animation:**
   - [ret_v1_pairs35_46_comparison.gif](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/ret_v1_pairs35_46_comparison.gif)
5. **Corrected Full Sequence Animation:**
   - [ret_v1_full_sequence.gif](file:///C:/Users/aabab/.gemini/antigravity/brain/643f8462-bebc-40e4-ad1a-1828c176dbad/ret_v1_full_sequence.gif)
