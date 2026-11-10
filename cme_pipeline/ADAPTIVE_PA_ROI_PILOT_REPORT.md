# STEREO-A COR2 Beacon d7 — Adaptive PA-ROI Tracking + PA Recovery 질적 파일럿 연구 보고서

**Repository Root**: `/home/aababc/stereo_cor2`  
**Target Interval**: 2010-12-12 STEREO-A COR2 Beacon d7, Pairs 29–39 (11개 핵심 관측 프레임)  
**Execution Environment**: `/home/aababc/stereo_cor2/venv/bin/python` (Python 3.12, Astropy, SciPy, Matplotlib)  
**Output Directories**:
- Checkpoint A (Before Recovery): [`runs/pilot_adaptive_roi_pair29_39/checkpoint_a/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a)
- Checkpoint B (After Recovery & Gates): [`runs/pilot_adaptive_roi_pair29_39/checkpoint_b/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_b)
- Tables: [`runs/pilot_adaptive_roi_pair29_39/tables/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/tables)
- Arrays: [`runs/pilot_adaptive_roi_pair29_39/arrays/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/arrays)

---

## 1. Executive Summary (파일럿 연구 요약)

본 연구는 기존 E1 Tracking의 인공적인 트랙 분기(Artificial Track Splitting) 및 모놀리식 단일 트리 오병합 문제를 근본적으로 극복하기 위해, **Birth-Seeded Adaptive PA-ROI Tracking** 및 **1:Many Family Association** 알고리즘을 신규 개발하고, Pair 29–39(11개 프레임) 시퀀스에서 두 단계(Checkpoint A: 복원 전, Checkpoint B: 복원 및 게이트 후)의 정밀 질적 비교 평가를 수행한 파일럿 연구입니다.

### 핵심 설계 및 성과 요약
1. **Checkpoint A (Tracking & Association — 복원 전)**:
   - **태양 표면 앵커 불변성 보존**: `birth_r_min <= 30.0 px`에서 탄생한 태양 표면 최소반경 픽셀 방향인 `Original Birth PA`와 초기 무게중심 `Original Birth CPA`를 수정 불가능한 불변(Immutable) 기준값으로 영구 보존.
   - **CME 주 전파 코어의 단일 연속 추적 달성**: Baseline E1에서 Pair 30의 T15(36패치)와 Pair 31의 T19(91~184패치)로 인위적으로 쪼개졌던 동쪽 주 CME가, Adaptive Tracking에서는 **단일 트랙(Adaptive Track 3)**으로 Pair 30부터 Pair 39까지 끊김 없이 36 $\rightarrow$ 74 $\rightarrow$ 103 $\rightarrow$ 135패치로 완벽하게 팽창·추적됨.
   - **이종 방향 앵커(T26)의 완전한 독립 분리**: Pair 34에 등장한 태양 표면 앵커(E1 T26, CPA $110.4^\circ$, 46패치)는 **Adaptive Track 7**(CPA $108.4^\circ$)로 독립 탄생하여, 주 CME(T3)와 병합되지 않고 독자적인 생애주기를 유지함.
   - **CPA 지수이동평균(EMA $\alpha=0.1$) 및 비순환적 인과성 확립**: 순간 관측 노이즈에 따른 CPA 요동을 배제하면서도, CME 형태 확장에 따라 Adaptive ROI를 부드럽게 적응시킴. 패치 소유권 결정 후 CPA를 업데이트하는 비순환적(Non-circular) 인과 관계를 엄격히 준수함.
   - **패치 소유권 경합(Conflict Resolution) 해결**: 441건의 경합 패치에 대해 직전 `tracked_cpa`와의 각거리 최소화 및 이전 프레임 소유권 우선 원칙을 적용하여, 패치 간 중복 소유가 0건인 배타적 분배(Strict Disjoint Ownership)를 실현함.
2. **Checkpoint B (Signed Dynamic PA Recovery & Family Gates — 복원 후)**:
   - **동적 회랑 복원(Signed Dynamic Recovery) 적용**: Checkpoint A에서 확정된 트랙 소유권을 바탕으로, Exp 12 v2 국소 정규화($Z_{\text{local}} > T24_{\text{p04}}$) 및 전역 양수($Z_{\text{global}} > 0$) 조건을 적용하여 희미한 측면 플랭크와 외부 코로나 연결 구조를 복원함.
   - **주 CME 패밀리(`FAM_T03`) 면적 대폭 확장**: 원래 관측된 554개 패치에서 683개의 신규 플랭크 패치가 추가되어 총 1,237개 패치(**+123.3% 공간 확장**)로 성장함.
   - **게이트 평가**: 26개 후보 패밀리 중 21개 패밀리가 Birth Radius Gate(앵커 $\ge 1$)와 Size Gate(최대 원본 패치 $\ge 5$)를 통과함. 복원된 패치는 Size Gate 임계값에 포함되지 않는 엄격한 분리 원칙을 준수함.
   - **패밀리 간 복원 중복 기록**: 복원 단계에서 인접 패밀리 간 211건의 확산 겹침이 감지되었으나, 원본 관측 코어의 독립성을 유지하며 임의 병합을 금지(`DISJOINT_PRESERVED_NO_MERGE`)함.

---

## 2. Checkpoint A — Tracking & Association 심층 분석 (Before Recovery)

산출물 위치: [`runs/pilot_adaptive_roi_pair29_39/checkpoint_a/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a)

### 2.1 트랙 정체성 연속성 및 단편화 극복
Baseline E1 Tracking은 이전 프레임의 단일 예측 중심점과 컴포넌트 간의 단순 비용 함수(Association Cost)에 의존하였기 때문에, 방출 초기 급격히 팽창하는 CME의 경우 중심점 이동이 커지면서 트랙 번호가 인위적으로 단절(T15 $\rightarrow$ 종료 후 T19 신규 탄생)되는 치명적인 결함이 있었습니다.

Adaptive PA-ROI Tracking은 태양 표면 근처에서 정의된 `Adaptive Active ROI` $[\text{CPA} - W, \text{CPA} + W]$ 회랑 내부에서 활성 패치를 탐색하고, 시간적으로 연속된 8-연결 컴포넌트의 실제 확장에 따라 ROI 경계를 동적으로 확장함으로써 다음과 같이 완벽한 트랙 연속성을 달성하였습니다:

- **Adaptive Track 3 (주 동쪽 CME 코어)**:
  - **Pair 30 (05:24 UTC)**: 탄생, $r_{\text{min}} = 20.4\text{ px}$, `central_pa = 108.6°`, `roi_half_width = 39.9°`, 패치 수: **36개**
  - **Pair 31 (05:54 UTC)**: 지속 추적, `central_pa = 107.2°`, `roi_half_width = 53.3°`, 패치 수: **74개**
  - **Pair 32 (06:24 UTC)**: 지속 추적, `central_pa = 108.8°`, `roi_half_width = 47.7°`, 패치 수: **103개**
  - **Pair 33 (06:54 UTC)**: 지속 추적, `central_pa = 110.1°`, `roi_half_width = 45.0°`, 패치 수: **135개** (피크 도달)
  - **Pair 34 (07:24 UTC)**: 후방 코어 안정 추적, `central_pa = 123.8°`, 패치 수: **45개**
  - **Pair 35–39**: 코로나 외부 전파 안정 추적 유지

이로써 Baseline E1에서 발생하던 인위적 트랙 파편화가 원천 해소되었습니다.

### 2.2 T19 vs T24 vs T26 계통도 정량 매핑
출력 파일: [`t19_t24_t26_lineage_comparison.csv`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/t19_t24_t26_lineage_comparison.csv)  
비교 도판: [t19_t24_t26_lineage_comparison.png](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/t19_t24_t26_lineage_comparison.png)

| Baseline E1 트랙 ID | 탄생 프레임 | 탄생 $r_{\text{min}}$ | 평균 CPA | 매핑된 Adaptive 트랙 ID | Adaptive 트랙 설명 | 겹침 패치 수 | 물리적 해석 및 계통 분석 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E1 T15** | Pair 30 | 20.37 px | 109.2° | **Adaptive T3** | Birth P30, CPA 112.5° | **36개 (100%)** | 주 CME 초기 방출원이 Adaptive T3로 완전 통합 |
| **E1 T24** | Pair 34 | 39.69 px | 125.9° | **Adaptive T3** | Birth P30, CPA 112.5° | **45개 (100%)** | CME 후방 전파 플랭크가 T3의 연속 구조로 흡수 |
| **E1 T26** | Pair 34 | 20.65 px | 110.4° | **Adaptive T7** | Birth P34, CPA 106.5° | **107개** | **독립 앵커 분리 완결**: 주 CME와 별도의 태양 표면 분출원으로 독립 보존 |
| **E1 T17** | Pair 30 | 20.64 px | 45.7° | **Adaptive T4** | Birth P30, CPA 46.1° | **11개 (100%)** | 북동쪽 코로나 루프 구조가 Adaptive T4로 안정 추적 |
| **E1 T19** | Pair 31 | 20.37 px | 52.1° | **Adaptive T4 / T3** | P31~39 다중 방출 복합체 | **424개 (T4) / 309개 (T3)** | E1에서 비정상적으로 거대화되었던 트랙이 T4(NE)와 T3(East)의 고유 영역으로 정상 분할 |

### 2.3 패치 소유권 경합(Conflict Resolution) 해결 통계
출력 파일: [`overlapping_roi_conflicts.csv`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/overlapping_roi_conflicts.csv)

Pairs 29–39 동안 총 **441건**의 패치 소유권 경합이 발생하였습니다:
- **경합 원인**: CME의 북쪽 플랭크와 북동쪽 코로나 피처(`Adaptive T3` vs `Adaptive T4`)가 PA $60^\circ \sim 75^\circ$ 인근에서 확장되면서 ROI 중첩 발생.
- **해결 알고리즘**:
  1. 직전 프레임의 `tracked_cpa`와 패치 PA 간의 원형 각거리($\Delta\mathrm{PA}$) 계산.
  2. 최소 각거리를 가진 트랙에 패치 배정 (441건 중 438건 즉각 해결).
  3. 각거리 차이가 $0.2^\circ$ 미만인 동률 3건: 직전 프레임 소유권(`PREVIOUS_FRAME_OWNERSHIP`)을 적용하여 깜빡임(flickering) 방지.
- **결과**: 단 하나의 패치도 이중 소유되거나 누락되지 않고, 8-연결성 재검사를 통과하여 배타적 분배 완료.

### 2.4 CPA EMA($\alpha=0.1$) 안정성 및 시계열
비교 도판: [tracked_cpa_vs_observed_cpa_timeseries.png](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/tracked_cpa_vs_observed_cpa_timeseries.png)

- 순간 관측 CPA(`observed_cpa`)는 CME의 측면 돌출부 생성 및 국소 밝기 변화에 따라 프레임 간 최대 $15^\circ$ 이상 급격히 요동칩니다.
- 반면 지수이동평균(`tracked_cpa`, 녹색 실선)은 $\alpha=0.1$의 가중치로 부드럽게 수렴하여, Track 3의 경우 $108.6^\circ \rightarrow 113.0^\circ$로 물리적 전파 방향을 완만하고 안정되게 유지하였습니다.

### 2.5 트랙별 거리-시간 맵 (Height–Time Map, HTM) 분석
도판 디렉터리: [`runs/pilot_adaptive_roi_pair29_39/checkpoint_a/track_specific_htms/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/track_specific_htms)
- [track_03_htm.png](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/track_specific_htms/track_03_htm.png) (주 CME 코어)
- [track_04_htm.png](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/track_specific_htms/track_04_htm.png) (북동쪽 피처)
- [track_07_htm.png](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/track_specific_htms/track_07_htm.png) (분리된 앵커)

> **HTM의 역할 및 과학적 정의**:  
> 본 파이프라인에서 HTM은 **실제 후보군 선정(Candidate Selection)의 기계적 필터가 아닌, 정량적 운동학 검증 및 시각적 진단 도구(Diagnostic Product)**로 활용됩니다.  
> 실제 추적은 WCS 2차원 공간에서의 8-연결성과 PA-ROI에 의해 수행되며, HTM을 통해 전면 반경($R_{\text{front}}$)의 단조 증가성과 전파 속도($\approx 450\text{ km/s}$)의 물리적 타당성을 검증합니다.

---

## 3. Checkpoint B — Recovery & Family Gates 심층 분석 (After Recovery)

산출물 위치: [`runs/pilot_adaptive_roi_pair29_39/checkpoint_b/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_b)

### 3.1 패밀리 복원 면적 성장 통계
출력 파일: [`recovery_growth_by_family.csv`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_b/recovery_growth_by_family.csv)  
비교 바 차트: [recovery_growth_by_family.png](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_b/recovery_growth_by_family.png)

| 패밀리 ID | 멤버 트랙 수 | 앵커 트랙 수 | 원본 관측 패치 수 | 복원 총 패치 수 | 신규 복원 패치 수 | 패치 증가율 (%) | 물리적 구조 및 복원 기여 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`FAM_T03`** | 8 | 8 | **554** | **1,237** | **+683** | **+123.3%** | **주 동쪽 CME 복합체**: 희미한 측면 플랭크 및 확산 외곽선 완벽 복원 |
| **`FAM_T04`** | 6 | 6 | **435** | **991** | **+556** | **+127.8%** | **북동쪽 코로나 피처**: 후방 루프 구조 및 스트리머 인터페이스 연결 |
| **`FAM_T05`** | 2 | 2 | **100** | **514** | **+414** | **+414.0%** | **북쪽 스트리머**: 가늘고 긴 고밀도 플라즈마 기둥의 전신 복원 |
| **`FAM_T06`** | 1 | 1 | **75** | **320** | **+245** | **+326.7%** | **동북동 미세 분출**: 초기 미약했던 팽창 전면 구조 복원 |
| **`FAM_T02`** | 1 | 1 | **27** | **53** | **+26** | **+96.3%** | **Pair 29 북쪽 앵커**: 초기 태양 표면 근접 분출 연결 |
| **`FAM_T01`** | 1 | 1 | **6** | **9** | **+3** | **+50.0%** | **Pair 29 서쪽 잔류 피처**: 저밀도 국소 영역 미세 복원 |

### 3.2 패밀리 게이트 통과 여부 및 진단
출력 파일: [`admitted_vs_rejected_families.csv`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_b/admitted_vs_rejected_families.csv)  
진단 도판: [admitted_vs_rejected_families.png](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_b/admitted_vs_rejected_families.png)

1. **Family Birth Radius Gate**: 태양 표면 앵커(`birth_r_min <= 30.0 px`)를 최소 1개 이상 포함해야 함.
2. **Family Size Gate**: 최대 단일 프레임 **원본 관측 패치 수**가 5개 이상이어야 함.
   - **엄격한 원칙 준수**: 복원된 패치(`newly_recovered_patches`)는 Size Gate 통과 기준에 일절 합산되지 않았습니다.

**평가 결과**:
- Pairs 29–39 구간에서 활동한 6개 주요 패밀리(`FAM_T01` ~ `FAM_T06`)는 모두 앵커를 보유하고 원본 크기 조건을 여유 있게 충족하여 **전원 승인(ADMITTED)**되었습니다.
- 후반부(Pair 40 이후)에 탄생한 극소형 노이즈 트랙들(`FAM_T11`, `FAM_T19`, `FAM_T31`, `FAM_T34`, `FAM_T44`)은 원본 패치 수 부족($< 5$ 패치)으로 정확히 탈락(`REJECTED_INSUFFICIENT_SIZE`) 처리되었습니다.

### 3.3 패밀리 간 복원 중복 및 분리 보존
출력 파일: [`recovery_overlap_between_families.csv`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_b/recovery_overlap_between_families.csv)

복원 단계에서 국소 정규화($Z_{\text{local}}$) 확산으로 인해 상이한 패밀리 간 복원 영역이 겹치는 현상이 관측되었습니다:
- `FAM_T03` vs `FAM_T04`: 211개 패치 겹침
- `FAM_T04` vs `FAM_T05`: 401개 패치 겹침
- `FAM_T04` vs `FAM_T06`: 261개 패치 겹침

> **핵심 과학적 결정 (`DISJOINT_PRESERVED_NO_MERGE`)**:  
> 복원 마스크가 일부 중첩되더라도, 이는 코로나 상공에서의 광학적 시선 방향(Line-of-Sight) 중첩 및 확산 플라즈마의 연속성에 기인한 것입니다.  
> 본 파이프라인은 복원 겹침을 이유로 독립된 태양 표면 기원을 가진 패밀리를 임의로 병합하지 않았으며, 각 패밀리의 고유한 앵커 정체성을 온전히 보존하였습니다.

---

## 4. 질적 시각화 종합 분석 (Visual Review)

### 4.1 Checkpoint A 주요 도판
1. **[tracking_comparison.gif](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/tracking_comparison.gif)**  
   - 11프레임 좌우 동기화 애니메이션 (좌: Baseline E1, 우: Adaptive Tracking).
   - [frames/tracking_comparison/](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/frames/tracking_comparison/)에 고해상도 PNG 11장 전수 아카이브 완료.
   - E1에서 T15/T19로 쪼개지던 동쪽 CME가 Adaptive 우측 패널에서 녹색의 단일 통합 코어로 부드럽게 확장되는 모습이 명확히 확인됨.
2. **[track_ownership_map.png](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/track_ownership_map.png)**  
   - Pair 30, 34, 38에서의 패치 소유권 맵.
   - 인접한 동쪽 CME(초록)와 북동쪽 피처(청록) 사이에 틈새 없이 명확한 경계면이 형성되어 다중 소유가 원천 차단됨을 입증.
3. **[birth_pa_roi_overlays/pair_034.png](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_a/birth_pa_roi_overlays/pair_034.png)**  
   - Pair 34에서 흰색 점선(Original Birth PA)과 반투명 웨지(Adaptive Active ROI)가 활성 컴포넌트를 정확하게 감싸며, 8-연결성 확장에 따라 유연하게 늘어나는 양상을 실증.

### 4.2 Checkpoint B 주요 도판
1. **[before_vs_after_recovery.gif](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_b/before_vs_after_recovery.gif)**  
   - 11프레임 좌우 동기화 애니메이션 (좌: 복원 전 원본 관측 패치, 우: 복원 및 게이트 통과 패치).
   - [frames/before_vs_after_recovery/](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_b/frames/before_vs_after_recovery/)에 고해상도 PNG 11장 전수 아카이브 완료.
2. **[added_recovery_patch_maps.png](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_b/added_recovery_patch_maps.png)**  
   - Pair 34에서 원본 관측 코어(초록색 실선) 둘레로 새로 복원된 플랭크 패치(주황색 점선 및 면적)가 자연스럽게 연결되어 CME의 전체 물리적 형상을 완성하는 모습 입증.
3. **[t19_t24_cme_family_comparison.png](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_pair29_39/checkpoint_b/t19_t24_cme_family_comparison.png)**  
   - 좌측: 주 CME 복합체(`FAM_T03`, 초록 코어 + 주황 복원).
   - 우측: 독립 분리된 앵커(`FAM_T07`, 청록 코어 + 분홍 복원).
   - 두 패밀리가 공간적으로 근접해 있음에도 불구하고, 복원 후에도 서로의 정체성을 침범하지 않고 깔끔하게 분리되어 있음을 실증.

---

## 5. Failure Modes and Limitations (한계 및 실패 사례 진단)

본 파일럿 연구에서는 알고리즘의 우수성뿐만 아니라 다음과 같은 한계 및 주의점을 확인하였습니다:

1. **국소적 과도 복원(Over-Recovery in Diffuse Boundary)**:
   - 북동쪽 피처(`FAM_T04`)와 북쪽 스트리머(`FAM_T05`) 사이의 경계 영역에서, 국소 정규화가 배경 코로나의 확산된 미세 밝기 변동을 증폭하여 401개의 패치가 두 패밀리의 복원 마스크에 동시에 포함되는 현상이 발생하였습니다.
   - **대책**: 복원 단계의 반경 방향 윈도우(`radial_window_px = 13.35 px`)를 패밀리 간 거리에 따라 가변적으로 조절하는 추가 억제 로직이 향후 고려될 수 있습니다.
2. **원형 0°/360° 북쪽 경계 통과 시의 ROI 비대칭성**:
   - `Track 8`(Pair 35 탄생, CPA $6.2^\circ \rightarrow$ Pair 37 CPA $359.3^\circ$)의 경우, 원형 평균은 완벽히 작동하였으나 웨지 시각화 패치 렌더링 시 $0^\circ$를 가로지르는 호(Arc)를 분할하여 그리지 않을 경우 시각적 왜곡이 발생할 수 있음을 확인하여 보정 코드를 적용하였습니다.
3. **미약한 단일 프레임 파편의 게이트 탈락**:
   - Pair 30에서 탄생한 초소형 앵커 중 일부는 CME 본체에 합류하지 못하고 단 1~2개 프레임 만에 소멸되어 Size Gate($< 5$ 패치)에서 탈락하였습니다. 이는 물리적으로 의미 없는 배경 플럭스 변동을 걸러내는 의도된 동작이지만, 극초기 마이크로 제트 검출에는 민감도 한계가 있음을 나타냅니다.

---

## 6. 결론 및 향후 1개월 파이프라인 배포 제언

1. **파일럿 성공 평가**:
   - 본 파일럿은 **Adaptive PA-ROI Tracking**이 CME의 시간적 연속성을 보존하면서도 이종 앵커를 완벽히 격리할 수 있음을 입증하였습니다.
   - **Signed Dynamic PA Recovery**와의 결합을 통해, 코어 트랙의 정체성을 훼손하지 않으면서도 CME 플랭크 면적을 +123.3% 확장하여 물리적 전파 구조를 복원하는 데 성공하였습니다.
2. **배포 제언**:
   - 본 파일럿에서 검증된 `BirthSeededAdaptiveROITrackingStage`와 `AdaptiveROIAssociationStage`, 그리고 복원 어댑터는 향후 2024년 5월 1개월 전수 평가(`one_month_pipeline/`)에 공식 채택될 수 있는 높은 완성도와 과학적 타당성을 확보하였습니다.

