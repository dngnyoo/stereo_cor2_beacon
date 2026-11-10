# Stereo_COR2 — Adaptive PA-ROI Outward Motion Gate 실험 (A0 vs A2) 및 시각화 결함 교정 보고서

**실험 디렉터리**: [`runs/pilot_adaptive_outward_gate_a0_a2/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/)  
**대상 관측**: 2010-12-12 STEREO-A COR2 Beacon d7 (Pairs 30–35 핵심 분석 및 Pairs 29–39 확장 평가)  
**대조군 구성**:
- **A0**: 기존 Adaptive PA-ROI Tracking (외향 이동 제약 미적용, `outward_gate_enabled = False`)
- **A2**: 외향 이동 허용 오차 제약 적용 (`outward_gate_enabled = True`, $\delta = 4.0\text{ px}$, $R_{\mathrm{front}}(t) \ge R_{\mathrm{front}}(t-1) - 4\text{ px}$)  
**작성 일자**: 2026-11-09  

---

## 1. 개요 및 실험 목적

지난 계보 정밀 진단([`runs/pilot_adaptive_roi_lineage_diagnostic/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/))에서 발견된 두 가지 핵심 문제를 해결하고 규명하기 위해 본 통제 실험을 수행하였습니다:

1. **시각화 좌표계 결함 규명 및 교정 (Part 4)**: 이전 진단 보고서의 확대 시각화(Figure 4 및 Figure 6)에서 트랙 라벨과 실제 패치 위치가 불일치했던 원인을 프로그램 수준에서 역추적하고, 실제 패치 마스크로부터 확대 경계(Zoom bounds)를 자동 계산하여 $100\%$ 무결성이 검증된 교정 도면을 생성합니다.
2. **외향 이동 게이트(Outward Motion Gate) 통제 실험 (Part 2 & 3)**: Pair 35에서 Adaptive Track 3의 선행 전선(leading edge)이 $R=82.32\text{ px}$에서 $R=41.25\text{ px}$로 급격하게 후퇴(inward retreat, $\Delta R = -41.06\text{ px}$)했던 현상을 방지하기 위해, 단순 외향 전파 제약($R_{\mathrm{front}}(t) \ge R_{\mathrm{front}}(t-1) - 4\text{ px}$)을 도입하고 그 물리적·알고리즘적 영향을 대조군 A0와 정량 비교합니다.

> [!IMPORTANT]
> **보존 및 통제 원칙**: 기존 Baseline E1, Family Association, Recovery, Gate 규칙은 일체 변경하지 않았습니다. 외향 이동 게이트는 `TrackingConfig.outward_gate_enabled` 옵션 플래그(기본값 `False`)로 최소 침습 형태로 구현되어 A0의 기존 수치 결과가 $100\%$ 비트 단위로 재현됨을 검증하였습니다.

---

## 2. 이전 시각화 결함 원인 규명 및 교정 (Part 4)

### 2.1 결함 원인 정밀 조사 결과
이전 `run_lineage_diagnostic.py`의 `fig_04` 및 `fig_06` 렌더링 코드를 역추적한 결과, **과학적 추적 마스크 배열(`t_own_ad_all`, `lbls_e1_all`)은 완벽하게 정상이었으나, 플로팅 코드의 하드코딩된 확대 창 좌표(`zoom_x`, `zoom_y`)가 태양의 반대편(서쪽)을 지정했던 순수 플로팅 크롭 오류**였음이 밝혀졌습니다.

- **실제 패치 좌표 (동쪽 사분면, Eastern Limb)**:
  - 64×64 패치 그리드에서 동쪽 CME 본체(Track 3 / E1 T24)의 위치는 `col \in [11, 24]`, `row \in [20, 28]`였습니다.
  - 256×256 이미지 픽셀 좌표(`x = col * 4`, `y = row * 4`)로 환산하면 **$X \in [44, 100]$, $Y \in [80, 116]$**에 위치합니다.
  - 또한 근원부 앵커(Track 7 / E1 T26)는 **$X \in [72, 112]$, $Y \in [104, 132]$**에 위치합니다.
  - STEREO COR2 North-Up 좌표계에서 태양 중심은 $(128, 128)$이며, 동쪽($\mathrm{PA} \approx 90^\circ \sim 120^\circ$)은 **$X < 128$** 영역입니다.
- **기존 플로팅 코드의 오류**:
  - `fig_04`에서 하드코딩된 확대 창: `zoom_x = (115, 235)`, `zoom_y = (65, 175)`
  - `fig_06`에서 하드코딩된 확대 창: `zoom_root_x = (110, 190)`, `zoom_root_y = (90, 170)`
  - 결과적으로 $X \ge 115$인 서쪽(West) 영역만을 잘라내어 화면에 표시했기 때문에, 실제 $X \le 112$에 위치하던 **Track 3의 45개 패치 중 0개(0%), Track 7의 46개 패치 중 0개(0%)**만 화면에 포함되고 본체 전체가 화면 밖으로 잘려 나갔습니다.
  - 주석 텍스트(Annotation) 또한 $X=195, 145$라는 텅 빈 서쪽 허공에 하드코딩 배치되어 있었습니다.

### 2.2 자동 바운딩 박스 및 주석 앵커 기반 교정
새로 작성된 [`run_outward_gate_experiment.py`](file:///home/aababc/stereo_cor2/cme_pipeline/experiments/run_outward_gate_experiment.py)에서는 하드코딩 좌표를 전면 폐기하고, **대상 트랙 마스크로부터 확대 바운딩 박스와 주석 앵커를 자동 도출**하도록 수정하였습니다:
- **자동 확대 창 (Fig 4)**: `rc_comb = np.argwhere(m_t24 | m_t26)`로부터 여백 24px을 적용하여 **$X \in [20, 136]$, $Y \in [56, 156]$**로 자동 설정.
- **자동 확대 창 (Fig 6)**: Pairs 34 & 35의 Track 7 마스크 및 차폐기 경계로부터 **$X \in [38, 142]$, $Y \in [74, 162]$**로 자동 설정.
- **주석 앵커**: 마스크의 실제 질량 중심(Centroid) 좌표(`(mean(X), mean(Y))`)에 지시선을 직접 연결.

### 2.3 시각화 자동 무결성 검증 결과 ([`visualization_validation_report.csv`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/tables/visualization_validation_report.csv))

| Figure | 대상 Track | 총 패치 수 | 표시 창 내 가시 패치 수 | 가시 비율 | 주석 앵커 좌표 | 창 내부 위치 여부 | 검증 상태 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`corrected_fig_04`** | Track 3 (E1 T24) | 45 | 45 | **100.0%** | $(73.1, 93.9)$ | True | **PASS** |
| **`corrected_fig_04`** | Track 7 (E1 T26) | 46 | 46 | **100.0%** | $(90.3, 116.1)$ | True | **PASS** |
| **`corrected_fig_06`** | Track 7 (P34 & P35) | 51 (유니온) | 51 | **100.0%** | $(87.9, 118.0)$ | True | **PASS** |

- 생성된 교정 이미지:
  - [`visualizations/corrected_fig_04_t24_to_t3_absorption_zoom.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/visualizations/corrected_fig_04_t24_to_t3_absorption_zoom.png)
  - [`visualizations/corrected_fig_06_t26_to_t7_independent_birth_zoom.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/visualizations/corrected_fig_06_t26_to_t7_independent_birth_zoom.png)
  - [`visualizations/verified_fig_02_adaptive_ownership_overlay_pairs30_35.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/visualizations/verified_fig_02_adaptive_ownership_overlay_pairs30_35.png)

---

## 3. 외향 이동 게이트(Outward Motion Gate) 구현 상세 (Part 2)

### 3.1 구현 위치 및 파라미터
- **모듈**: [`cme_pipeline/stages/tracking/adaptive_roi_tracking.py`](file:///home/aababc/stereo_cor2/cme_pipeline/stages/tracking/adaptive_roi_tracking.py)
- **설정 추가** ([`cme_pipeline/core/config.py`](file:///home/aababc/stereo_cor2/cme_pipeline/core/config.py)):
  - `outward_gate_enabled: bool = False` (기본값 비활성화로 A0 완벽 재현 보장)
  - `outward_gate_tolerance_px: float = 4.0` (허용 오차 $\delta = 4.0\text{ px}$)

### 3.2 게이트 판정 및 거절 처리 메커니즘
Step D(단일 8-연결 클러스터 선택) 직후, 기존 트랙 $t$의 관측 이력(`last_obs`)이 존재하는 경우 선행 전선 반지름의 변위($\Delta R$)를 평가합니다:
$$\Delta R = R_{\mathrm{front}}(t) - R_{\mathrm{front}}(t-1)$$
$$\text{Gate Condition: } \Delta R \ge -\delta \quad (\delta = 4.0\text{ px})$$

- **게이트 통과 시 ($\Delta R \ge -4.0\text{ px}$)**:
  - 정상적으로 패치 소유권을 확정하고, `OBSERVED` 상태로 기록하며, 적응형 ROI 및 CPA를 갱신합니다.
- **게이트 위반 시 ($\Delta R < -4.0\text{ px}$)**:
  - 해당 후보 클러스터 할당을 즉시 **거절(REJECTED_OUTWARD_GATE)**합니다.
  - 패치를 임의 삭제하지 않고, 트랙 $t$는 해당 패치 소유를 포기합니다 (따라서 미할당 패치 그리드 `unclaimed_grid`로 보존됨).
  - 트랙 $t$는 기존의 결측 프레임 처리 경로(`PREDICTED_ONLY`)로 진입하여 `t.missed += 1`을 기록하고, 이전 프레임의 선행 전선 반지름($R_{\mathrm{front}}(t-1)$)과 중심각을 유지합니다.
  - 거절 이력은 [`tables/a2_outward_gate_rejections.csv`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/tables/a2_outward_gate_rejections.csv)에 전수 로깅됩니다.

---

## 4. 정량 비교 결과: A0 vs A2 (Part 3 & 6)

### 4.1 Pairs 30–35 트랙별 상태 및 수치 비교 ([`a0_vs_a2_track_comparison.csv`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/tables/a0_vs_a2_track_comparison.csv))

| Pair | Track | A0 상태 | A0 $R_{\mathrm{front}}$ | A0 패치수 | A0 $\Delta R$ | A2 상태 | A2 $R_{\mathrm{front}}$ | A2 패치수 | A2 $\Delta R$ | 비고 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **P30** | T3 | OBSERVED | 37.69 px | 36 | 0.0 px | OBSERVED | 37.69 px | 36 | 0.0 px | A0/A2 완전 일치 (동쪽 CME 탄생) |
| **P30** | T4 | OBSERVED | 39.03 px | 11 | 0.0 px | OBSERVED | 39.03 px | 11 | 0.0 px | A0/A2 완전 일치 (북동쪽 특징) |
| **P30** | T5 | OBSERVED | 22.58 px | 1 | 0.0 px | OBSERVED | 22.58 px | 1 | 0.0 px | A0/A2 완전 일치 (북쪽 극미세) |
| **P31** | T3 | OBSERVED | 48.65 px | 74 | +10.96 px | OBSERVED | 48.65 px | 74 | +10.96 px | A0/A2 완전 일치 (정상 외향 전파) |
| **P31** | T4 | OBSERVED | 36.69 px | 14 | -2.35 px | OBSERVED | 36.69 px | 14 | -2.35 px | $|\Delta R| \le 4\text{ px}$ 허용 범위 통과 |
| **P32** | T3 | OBSERVED | 59.89 px | 103 | +11.24 px | OBSERVED | 59.89 px | 103 | +11.24 px | A0/A2 완전 일치 |
| **P32** | T4 | OBSERVED | 47.87 px | 31 | +11.19 px | OBSERVED | 47.87 px | 31 | +11.19 px | A0/A2 완전 일치 |
| **P33** | T3 | OBSERVED | 71.97 px | 135 | +12.08 px | OBSERVED | 71.97 px | 135 | +12.08 px | A0/A2 완전 일치 |
| **P33** | T6 | OBSERVED | 21.95 px | 1 | 0.0 px | OBSERVED | 21.95 px | 1 | 0.0 px | A0/A2 완전 일치 (T6 정상 탄생) |
| **P34** | T3 | OBSERVED | 82.32 px | 45 | +10.35 px | OBSERVED | 82.32 px | 45 | +10.35 px | A0/A2 완전 일치 (E1 T24 흡수) |
| **P34** | T7 | OBSERVED | 57.49 px | 46 | 0.0 px | OBSERVED | 57.49 px | 46 | 0.0 px | A0/A2 완전 일치 (T7 독립 앵커 탄생) |
| **P35** | **T3** | **OBSERVED** | **41.25 px** | **7** | **-41.06 px** | **PREDICTED** | **82.32 px** | **0** | **-41.06 px** | **게이트 작동: 내부 후퇴 차단 및 전선 보존!** |
| **P35** | **T6** | OBSERVED | 46.63 px | 8 | -4.44 px | PREDICTED | 51.07 px | 0 | -4.44 px | 변위 $-4.44\text{ px} < -4.0\text{ px}$로 거절 |
| **P35** | T7 | OBSERVED | 59.89 px | 32 | +2.39 px | OBSERVED | 59.89 px | 32 | +2.39 px | A0/A2 완전 일치 (T7 정상 성장) |
| **P35** | T8 | OBSERVED | 26.02 px | 3 | 0.0 px | OBSERVED | 46.63 px | 8 | 0.0 px | T6 거절 조각이 새 트랙 T8로 탄생 |
| **P35** | T9 | ABSENT | - | 0 | - | OBSERVED | 26.02 px | 3 | 0.0 px | A0의 북쪽 T8이 ID 9로 승계됨 |

---

## 5. 5대 핵심 질문에 대한 과학적 검증 답변

### 질문 1: A2가 Pair 35에서 T3의 겉보기 내부 후퇴(Inward Retreat)를 방지하는가?
- **답변: 완벽하게 방지함.**
- A0에서는 Track 3가 Pair 34의 $R=82.32\text{ px}$에서 Pair 35의 $R=41.25\text{ px}$로 **$-41.06\text{ px}$ 급격히 후퇴**하였습니다.
- A2에서는 외향 이동 게이트가 $\Delta R = -41.06\text{ px} < -4.0\text{ px}$를 감지하여 해당 내부 후보 할당을 거절하고 `PREDICTED_ONLY`로 전환함으로써, **선행 전선 반지름 $R_{\mathrm{front}} = 82.32\text{ px}$를 물리적으로 올바르게 보존**하였습니다 ([`fig_03_eastern_cme_leading_edge_trajectory.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/visualizations/fig_03_eastern_cme_leading_edge_trajectory.png)).

---

### 질문 2: 게이트가 내부 구조를 잘못 붙이지 않고 외향 선행 CME 전선을 보존하는가?
- **답변: 완벽하게 보존함.**
- A0가 Pair 35에서 획득했던 7개 패치는 $\mathrm{PA} \approx 118^\circ$, $R \in [34.2, 41.3]\text{ px}$에 위치한 **태양 근처의 후방 잔류 코로나 코어(trailing streamer core)**였습니다.
- A2는 이 7개 패치를 Track 3의 전선에 인위적으로 들러붙지 않도록 차단하였습니다. 또한 차단된 7개 패치는 최소 반경이 $R_{\min} = 34.23\text{ px} > 30.0\text{ px}$이므로, 원치 않는 가짜 솔라 루트 앵커로 탄생하지 않고 자연스럽게 미할당으로 유지되었습니다.

---

### 질문 3: 게이트가 유효한 트랙을 조기 종료시키거나 불필요한 트랙 탄생을 유발하는가?
- **답변: 조기 종료는 없으나, 임계값 경계에서의 미세한 트랙 파편화(Fragmentation)가 발생함.**
  1. **조기 종료 여부**: T3의 경우 1개 프레임만 예측 상태(`missed = 1`)가 되었으므로 허용 결측치(`MAX_MISSED_FRAMES = 2`) 이내로서 조기 종료되지 않았습니다.
  2. **트랙 탄생 부작용 (T6 사례)**: Pair 35에서 Track 6의 후보 컴포넌트 변위는 $\Delta R = -4.44\text{ px}$였습니다. 이는 허용치($-4.0\text{ px}$)를 **단 $0.44\text{ px}$ 초과**하여 거절되었는데, 해당 후보 컴포넌트의 기저 반경이 $R_{\min} = 21.95\text{ px} \le 30.0\text{ px}$였기 때문에 Step H에서 **새로운 트랙(Track 8)으로 분리 탄생**하는 트랙 ID 증가 현상이 발생했습니다.
  3. **전일 시퀀스(Pairs 29–39) 영향**: Pairs 29–39 구간 전체에서 총 생성 트랙 수는 A0의 44개에서 A2의 65개로 증가하였습니다. 즉, 엄격한 $-4\text{ px}$ 고정 임계값은 내부 뿌리 영역($R \le 30\text{ px}$)과 맞닿은 복잡한 구조에서 경계 거절 시 신규 트랙을 파생시키는 트레이드오프가 존재합니다.

---

### 질문 4: 게이트가 Pair 30의 T4/T5 분할이나 Pair 33의 T6 탄생에 악영향을 주는가?
- **답변: 전혀 악영향 없음 (100% 동일 보존).**
  - Pair 30: T3(36p), T4(11p), T5(1p) 모두 탄생 관측이므로 게이트 판정 대상이 아니며 완벽히 동일합니다.
  - Pair 31~32: T4 및 T5의 방사상 이동량이 오차 범위($\pm 4\text{ px}$) 내이므로 전혀 거절되지 않고 동일하게 추적되었습니다.
  - Pair 33: Track 6의 최초 탄생($R_{\min} = 21.95\text{ px}$) 역시 완벽하게 동일하게 수행되었습니다.
  - Pair 34까지 A0와 A2의 모든 수치 배열과 테이블은 $100\%$ 동일합니다.

---

### 질문 5: Pair 34–35에서 외향 전선 패치는 검출(Detection) 단계에서 결손된 것인가, 추적(Tracking) 중에 유실된 것인가?
- **핵심 과학적 규명**: **외향 전선($R > 80\text{ px}$)은 추적 단계의 오류가 아니라, 검출(Detection) 단계의 원본 그리드(`grid4`)에서 신호가 사라진 것(Detection Drop-out)임.**
  - **Pair 34 검출 단계**: 동쪽 사분면($\mathrm{PA} \in [80^\circ, 150^\circ]$)에서 총 104개 패치가 활성화되었으며, $R \ge 70\text{ px}$ 패치가 21개, $R \ge 80\text{ px}$ 패치가 2개($R_{\max} = 82.32\text{ px}$) 존재하여 거대한 45패치 컴포넌트(Component 2)를 형성했습니다.
  - **Pair 35 검출 단계**: 동쪽 활성 패치 수가 104개에서 49개로 **$53\%$ 급감**하였습니다. 특히 **$R \ge 80\text{ px}$ 패치는 0개(완전 결손)**였으며, $R \ge 70\text{ px}$ 패치도 단 2개(Component 1: $R \in [73.3, 76.5]\text{ px}$, Component 2: $R \in [66.5, 69.9]\text{ px}$)의 미세 파편으로 쪼그라들었습니다.
  - **결론**: Pair 35에서 CME 전선이 급격히 희미해져(Beacon d7의 낮은 SNR 및 배경 잡음 대비 전선 분산) 검출 단계에서 외향 전선이 거의 포착되지 않았습니다.
  - **A0의 실패 메커니즘**: 전선 신호가 결손되자, Step D의 단순 시드 개수 규칙이 2개짜리 전선 파편 대신 7개짜리 내부 코어(Component 3)를 선택하여 **가짜 내부 후퇴를 발생**시켰습니다.
  - **A2의 성공 메커니즘**: 전선 신호가 결손되었음을 외향 게이트가 인식하고, 내부 코어로의 오할당을 거절하여 **트랙을 정직하게 예측(PREDICTED_ONLY) 상태로 유지**하였습니다.

---

## 6. 시각화 결과 분석 (Part 5)

### 6.1 전 시퀀스 A0 vs A2 나란히 비교 ([`fig_01_a0_vs_a2_side_by_side_pairs30_35.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/visualizations/fig_01_a0_vs_a2_side_by_side_pairs30_35.png))
- Pairs 30부터 34까지 A0(좌측)와 A2(우측)는 완벽하게 동일한 트랙 기하 및 색상을 유지합니다.
- Pair 35에서 A0는 T3가 동쪽 내부($R=41.3\text{ px}$)로 축소되어 표시되는 반면, A2는 T3를 파선 박스의 `PREDICTED_ONLY`로 유지하고 하단에 거절 로그 배너(`REJECTED: T3 disp=-41.1px`)를 표시합니다.

### 6.2 집약적 동쪽 CME 진화 비교 ([`fig_02_focused_pair33_35_eastern_cme.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/visualizations/fig_02_focused_pair33_35_eastern_cme.png))
- Pair 33 ($R=72.0\text{ px}$) $\to$ Pair 34 ($R=82.3\text{ px}$) $\to$ Pair 35에서:
  - A0는 전선이 $41.3\text{ px}$로 갑자기 차폐기 근처로 되돌아가는 비물리적 모습을 명확히 보여줍니다.
  - A2는 전선이 $82.3\text{ px}$ 위치에 안정적으로 고정되어 외향 전파 일관성을 지킵니다.

### 6.3 선행 전선 운동학 궤적 분석 ([`fig_03_eastern_cme_leading_edge_trajectory.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/visualizations/fig_03_eastern_cme_leading_edge_trajectory.png))
- 상단 패널(Leading Edge Radius): A0(빨간 실선)는 Pair 34에서 35로 갈 때 수직 낙하하는 $-41.1\text{ px}$ 계단형 불연속을 보이지만, A2(하늘색 점선)는 $R=82.3\text{ px}$ 수평 플래토를 유지합니다.
- 하단 패널(Radial Displacement per Frame): A0의 심각한 음수 변위($-41.1\text{ px}$)와 A2의 안정된 $0\text{ px}$ 예측 유지를 나란히 입증합니다.

---

## 7. 향후 개선을 위한 권장 사항 (Next Experiments)

A2 실험을 통해 외향 이동 게이트가 가짜 내부 후퇴를 방지하는 데 대단히 효과적임이 입증되었으나, 고정 허용치($-4.0\text{ px}$)의 경계값 문제(예: T6의 $-4.44\text{ px}$에 의한 파편화)가 확인되었습니다. 따라서 다음 단계의 제어 실험으로 아래 방안을 권장합니다:

1. **상대적 또는 시간 적응형 허용 오차 (Cadence/Speed-Adaptive Tolerance)**:
   - 고정 $-4\text{ px}$ 대신, 관측 시간 간격 $\Delta t$ 및 트랙의 이전 방사상 속도 $v_r$에 연동된 오차 $\delta = \max(6.0\text{ px}, 0.2 \cdot v_r \cdot \Delta t)$를 적용하여 미세한 형태 변화에 의한 우발적 거절을 완화.
2. **다중 컴포넌트 전선 선호 규칙 (Front-Prioritized Component Selection in Step D)**:
   - Step D에서 단순히 시드 개수만으로 클러스터를 고르는 대신, 후보 컴포넌트들의 $R_{\mathrm{front}}$ 중 이전 전선과 연속성을 갖는 외곽 컴포넌트에 가중치를 부여함으로써, Pair 35의 Component 1(2개 전선 파편)이 7개짜리 내부 코어에 밀려 버려지는 현상을 근본적으로 해결.

모든 정량 CSV 테이블, 과학 배열 및 검증된 시각화 도면은 [`runs/pilot_adaptive_outward_gate_a0_a2/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_outward_gate_a0_a2/)에 완전하게 아카이빙되었습니다.
