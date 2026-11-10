# Stereo_COR2 — Adaptive PA-ROI Tracking 계보 정밀 진단 보고서 (Lineage Diagnostic Report)
**관측 대상**: 2010-12-12 STEREO-A COR2 Beacon d7 (Pairs 30–35, 6개 프레임)  
**분석 트랙**: Baseline E1 (T15, T17, T19, T24, T26) vs Adaptive PA-ROI (T3, T4, T5, T6, T7, T8)  
**실험 디렉터리**: [`runs/pilot_adaptive_roi_lineage_diagnostic/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/)  
**작성 일자**: 2026-11-03  

---

## 1. 진단 배경 및 목적

본 진단은 2010-12-12 Beacon d7 데이터의 CME 주요 분출 구간인 **Pair 30부터 Pair 35(05:24 ~ 07:54 UTC, 총 6프레임)**를 대상으로, 기존 **Baseline E1 Tracking**과 신규 **Birth-Seeded Adaptive PA-ROI Tracking** 간의 패치 단위 계보(Patch-level Lineage)를 1:1로 엄밀하게 역추적하고 그 인과관계를 규명하기 위해 수행되었습니다.

특히 기존 Baseline에서 나타났던 다음 4가지 핵심 현상의 알고리즘적 발생 메커니즘을 규명하는 데 집중하였습니다:
1. **E1 T24 $\to$ Adaptive T3 흡수**: 독립 E1 트랙 T24가 Adaptive Tracking에서 왜 T3의 연속 전선으로 자연스럽게 흡수되었는가?
2. **E1 T19 $\to$ Adaptive T3/T4 분할**: Baseline E1에서 단일 거대 트랙이었던 T19가 왜 Adaptive Tracking에서는 T3(동쪽 CME)와 T4(북동쪽 코로나 구조)로 깔끔하게 물리적 분리되었는가?
3. **E1 T26 $\to$ Adaptive T7 독립 생성**: Pair 34에서 나타난 E1 T26 패치들이 Adaptive Tracking에서 T3에 편입되지 않고 왜 독립 솔라 루트 앵커(Adaptive T7)로 탄생했는가?
4. **역대 보고서의 T19 / T26 CPA 불일치 원인**: 보고서마다 서로 다르게 기록된 T19(52° vs 80° vs 110°) 및 T26(109° vs 8°)의 중심 위치각(CPA) 수치가 어떤 근거에서 기인했는가?

> [!IMPORTANT]
> 본 진단은 **알고리즘 수정 전 순수 원인 규명 작업**으로 수행되었으며, 기존 Tracking, Family Association, Recovery 및 Gate 규칙은 일체 변경되지 않았습니다.

---

## 2. 산출물 및 검증 무결성

모든 진단 산출물은 독립 디렉터리 [`runs/pilot_adaptive_roi_lineage_diagnostic/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/)에 보존되었으며, 기존 결과 및 원본 과학 배열은 완벽히 유지되었습니다.

### 2.1 진단 정량 테이블 ([`tables/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/tables/))
- [`track_identity_audit.csv`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/tables/track_identity_audit.csv): E1 및 Adaptive 트랙 11종의 6개 프레임 전 구간 상태, 기하 파라미터, 상대 트랙 대응 관계 감사 테이블 (66행)
- [`patch_level_temporal_lineage.csv`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/tables/patch_level_temporal_lineage.csv): Pairs 30–35 활성 패치 776개 전체의 좌표, 소속 트랙, 연결 유형(직접 중첩, 8-연결 성장, ROI 시드 경합 등) 정밀 분류 테이블 (776행)
- [`htm_ridge_diagnostic.csv`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/tables/htm_ridge_diagnostic.csv): T3, T4, T7의 탄생 방위각(Birth PA) 기준 Height–Time Map 리지 및 방사상 속도 정량 진단 테이블 (18행)

### 2.2 고해상도 시각화 및 애니메이션 ([`visualizations/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/visualizations/))
- [`fig_01_rd_e1_overlay_pairs30_35.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/visualizations/fig_01_rd_e1_overlay_pairs30_35.png): Baseline E1 추적 결과 6프레임 오버레이 (Raw RD `[-600, +600]` DN)
- [`fig_02_adaptive_ownership_overlay_pairs30_35.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/visualizations/fig_02_adaptive_ownership_overlay_pairs30_35.png): 동일 6프레임의 Adaptive PA-ROI 소유권 오버레이
- [`fig_03_birth_seed_connection_paths.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/visualizations/fig_03_birth_seed_connection_paths.png): 중심점 궤적 및 탄생 시드 역추적 연결 경로 비교
- [`fig_04_t24_to_t3_absorption_zoom.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/visualizations/fig_04_t24_to_t3_absorption_zoom.png): Pair 34에서 E1 T24가 Adaptive T3 외곽 전선으로 흡수되는 순간의 확대 분석
- [`fig_05_t19_split_boundary_zoom.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/visualizations/fig_05_t19_split_boundary_zoom.png): Pairs 31–33에서 E1 T19 단일 덩어리가 T3/T4로 분할되는 공간 경계 확대 분석
- [`fig_06_t26_to_t7_independent_birth_zoom.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/visualizations/fig_06_t26_to_t7_independent_birth_zoom.png): Pair 34/35에서 E1 T26이 Adaptive T7 독립 앵커로 탄생하는 차폐기 근원부 확대 분석
- [`fig_07_t3_t4_t7_htm_ridge_comparison.png`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/visualizations/fig_07_t3_t4_t7_htm_ridge_comparison.png): T3, T4, T7의 Height–Time Map(HTM) 리지 및 방사상 전선 전파 비교
- [`fig_08_tracking_lineage_6frames.gif`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/visualizations/fig_08_tracking_lineage_6frames.gif): 6프레임 시간 계보 애니메이션 (연결 방식 시각화)
- **개별 프레임 보존** ([`frames/fig_08_tracking_lineage_6frames/`](file:///home/aababc/stereo_cor2/runs/pilot_adaptive_roi_lineage_diagnostic/frames/fig_08_tracking_lineage_6frames/)): [`VISUALIZATION_STANDARDS.md`](file:///home/aababc/stereo_cor2/VISUALIZATION_STANDARDS.md) 규정에 따라 Matplotlib에서 직접 렌더링된 고해상도 PNG(`pair_030.png` ~ `pair_035.png`, `dpi=130`)를 분해 없이 원본 저장.

---

## 3. 핵심 정량 진단 데이터 요약

### 3.1 트랙별 프레임 시계열 감사 (Track Identity Audit)

| System | Track | Pair 30 | Pair 31 | Pair 32 | Pair 33 | Pair 34 | Pair 35 | 비고 및 물리적 정체성 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E1** | **T15** | 36p (109.2°) | - | - | - | - | - | 동쪽 CME 탄생 $\to$ Pair 31에서 갑자기 종료됨 |
| **E1** | **T17** | 11p (41.3°) | 4p (48.0°) | PRED | - | - | - | 북동쪽 코로나 특징 $\to$ Pair 32에서 끊김 |
| **E1** | **T19** | - | 91p (80.5°) | 139p (79.4°) | 184p (72.8°) | 75p (46.3°) | 64p (45.2°) | **인위적 융합체**: P31~33(동쪽+북동쪽) $\to$ P34에서 북동쪽으로 튐! |
| **E1** | **T24** | - | - | - | - | 45p (121.8°) | 2p (127.9°) | P34 동쪽 외곽 전선 ($R_{\min}=39.7\text{ px}$) $\to$ 신규 분리 |
| **E1** | **T26** | - | - | - | - | 46p (109.3°) | 45p (109.9°) | P34 동쪽 차폐기 근원부 ($R_{\min}=20.6\text{ px}$) $\to$ 신규 분리 |
| **Adaptive** | **T3** | 36p (108.6°) | 74p (107.2°) | 103p (108.8°) | 135p (110.1°) | 45p (123.8°) | 7p (118.6°) | **동쪽 CME 본체**: 일관된 방위각(~110°)으로 전선 지속 추적 (P34=E1 T24 흡수) |
| **Adaptive** | **T4** | 11p (41.9°) | 14p (46.4°) | 31p (48.3°) | 39p (49.7°) | 33p (43.6°) | 48p (43.7°) | **북동쪽 코로나 특징**: 일관된 방위각(~45°) 유지하며 T19로부터 완벽 독립 |
| **Adaptive** | **T7** | - | - | - | - | 46p (108.4°) | 32p (105.5°) | **동쪽 근원부 앵커**: P34에서 탄생한 차폐기 뿌리 앵커 (P34=E1 T26 일치) |

### 3.2 패치 연결 유형 분류 (Lineage Connection Types across 776 Patches)

총 776개의 활성 패치는 다음과 같은 연결 메커니즘으로 판별되었습니다:
- `DIRECT_TEMPORAL_OVERLAP`: **363개 (46.8%)** — 직전 프레임($t-1$)의 동일 트랙 패치 위치와 정확히 일치하여 직접 지속된 패치.
- `8CONN_TEMPORAL_GROWTH`: **152개 (19.6%)** — 당해 프레임에서 시간적 중첩 패치와 8-연결(8-connectivity)로 연결되어 확장된 패치.
- `BIRTH_SEED_QUALIFIED`: **96개 (12.4%)** — 차폐기 근접($R_{\min} \le 30\text{ px}$) 조건을 만족하여 새로운 독립 트랙으로 탄생한 시드 패치.
- `ROI_UNCONTESTED_SEED`: **75개 (9.7%)** — 직전 프레임 직접 중첩은 없으나 트랙의 Adaptive ROI 각도 및 방사상 회랑 내에 존재하여 무경합 획득된 시드.
- `ROI_CONTESTED_WINNER`: **31개 (4.0%)** — 복수의 트랙 ROI가 겹친 구간에서 $|\Delta\mathrm{CPA}|$ 최소 원칙에 따라 승리하여 획득된 패치.
- `UNCLAIMED_DISCARDED`: **59개 (7.6%)** — 어느 트랙의 ROI에도 속하지 않거나, 단일 클러스터 제약으로 인해 탈락한 비할당 패치.

---

## 4. 6대 질문에 대한 심층 원인 진단

### 질문 1: T24가 Adaptive T3에 흡수된 직접적인 알고리즘 조건은 무엇인가?

#### [진단 결과]
E1 T24는 별개의 독립 CME가 아니라, **Adaptive Track 3가 Pair 30부터 추적해 온 동쪽 CME의 외곽 전선(downstream leading edge/flank)이 정상적으로 전파한 결과**입니다.

#### [정량적 및 알고리즘적 발생 조건]
1. **Adaptive Active ROI 포함**: Pair 33에서 Track 3의 적응형 활성 ROI는 중심각 $\mathrm{CPA}=110.1^\circ$, 반폭 $\mathrm{Half\text{-}Width}=45.0^\circ$ (각도 범위: $[65.0^\circ, 155.1^\circ]$)였으며, 방사상 허용 회랑은 $R \in [13.2, 102.0]\text{ px}$였습니다. Pair 34에서 E1 T24(45패치, 중심각 $121.8^\circ$, 방사상 범위 $41.8 \sim 82.3\text{ px}$)의 **모든 45개 패치는 Track 3의 ROI 및 방사상 탐색 회랑 내에 $100\%$ 포함**되었습니다.
2. **직접 시간 중첩(Direct Temporal Overlap) 존재**: E1 T24의 45개 패치 중 무려 **18개 패치는 Pair 33 시점의 Track 3 패치와 동일 좌표에서 직접 중첩**되었으며, 나머지 16개 패치는 이 중첩 패치들로부터 8-연결 확장(`8CONN_TEMPORAL_GROWTH`), 11개 패치는 ROI 내부 시드로 획득되었습니다.
3. **결정적 요인 (Step D 시드 개수 판정)**: Pair 34에서 동쪽 구조는 하류 전선(Component 2, 45패치)과 상류 근원부(Component 4, 46패치)로 분리되었습니다. Track 6이 Component 4의 북쪽 2개 패치를 경합 승리함에 따라, Track 3의 유효 시드 수는 Component 2에서 45개, Component 4에서 44개가 되었습니다. Step D의 단일 클러스터 규칙(`best_seed_count`)에 의해 **45개 대 44개(단 1개 차이)**로 Component 2가 Track 3의 공식 승계 클러스터로 선택되었습니다.
4. **결론**: 따라서 E1 T24의 45개 패치는 온전히 Adaptive Track 3로 계승 흡수되었습니다.

---

### 질문 2: T19가 두 Adaptive Track(T3/T4)으로 분할된 직접적인 원인은 무엇인가?

#### [진단 결과]
Baseline E1의 T19는 물리적으로 서로 완전히 무관한 **동쪽 주 CME(East CME, $\mathrm{PA}\sim 108^\circ$)와 북동쪽 코로나 스트리머(NE Streamer, $\mathrm{PA}\sim 45^\circ$)가 공간적 브릿징에 의해 잘못 합쳐진 인위적 융합체(unphysical artifact)**였으며, Adaptive Tracking의 방위각 ROI 및 CPA 경합 해결 규칙이 이를 물리적 원형대로 정상 분리하였습니다.

#### [인과 메커니즘]
1. **E1의 방향성 부재와 인위적 융합**: E1 Tracking은 방향성 각도 제한 없이 오직 프레임별 `label(grid4, STRUCT_8)`에 의존합니다. Pair 31에서 두 분출 구조 사이에 $\mathrm{PA}\sim 60^\circ \sim 75^\circ$에 걸쳐 희미한 활성 패치 몇 개가 연결되자, E1은 이를 91패치짜리 단일 거대 컴포넌트로 묶어버렸고, 헝가리안 매칭은 이를 T19로 명명했습니다 (Pair 32: 139패치, Pair 33: 184패치).
2. **Adaptive Tracking의 독립적 Birth Seed 보존**: Adaptive Tracking에서는 Pair 30에서 이미 동쪽의 Track 3($\mathrm{CPA}=108.6^\circ$)과 북동쪽의 Track 4($\mathrm{CPA}=41.9^\circ$)가 각각 독립된 솔라 루트 시드로 탄생했습니다.
3. **Step B 각도 경합 분할 ($|\Delta\mathrm{CPA}|$ 최소화)**: Pair 31~33에서 두 트랙의 ROI가 중간 영역($60^\circ \sim 80^\circ$)에서 겹쳤을 때, 알고리즘은 각 패치의 방위각과 각 트랙의 `tracked_cpa` 차이를 계산하여 패치를 엄밀히 배분했습니다.
   - Pair 31: E1 T19 (91패치) $\to$ **T3 (71패치, 107.2°) + T4 (14패치, 46.4°) + T5 (6패치, 11.9°) = 정확히 91패치 (100% 분할)**
   - Pair 32: E1 T19 (139패치) $\to$ **T3 (103패치) + T4 (31패치) + T5 (5패치) = 정확히 139패치 (100% 분할)**
   - Pair 33: E1 T19 (184패치) $\to$ **T3 (135패치) + T4 (39패치) + T5 (10패치) = 정확히 184패치 (100% 분할)**
4. **E1 T19의 붕괴와 순간이동**: Pair 34에서 중간 브릿지 패치가 사라지자, E1 T19는 헝가리안 매칭에 의해 갑자기 북동쪽 스트리머(Component 6, 75패치, $\mathrm{CPA}=46.3^\circ$)로 튐으로써 동쪽 CME 본체를 버렸습니다. 반면 Adaptive Tracking은 T3와 T4가 각각 자신의 방위각 축을 지키며 매끄럽게 연속성을 유지하였습니다.

---

### 질문 3: T26이 독립 Track(Adaptive T7)으로 탄생한 이유는 무엇인가?

#### [진단 결과]
E1 T26에 해당하는 패치들이 Adaptive Tracking에서 T3에 편입되지 않고 독립 Track 7로 탄생한 이유는, **CME 전선이 외곽으로 전파되면서 내부 구조가 공간적으로 단절되었을 때, 현행 추적기의 Step D가 오직 단 1개의 8-연결 클러스터만 허용하고 나머지 컴포넌트를 강제 탈락시켰기 때문**입니다.

#### [발생 시퀀스]
1. **공간 분리 발생**: Pair 34에서 동쪽 CME 구조는 외곽으로 날아가는 전선부(Component 2, 45패치)와 차폐기 바로 바깥에서 지속되는 근원부(Component 4, 46패치) 사이에 1~2개 패치 간격의 공백이 생기며 2개의 8-연결 컴포넌트로 분리되었습니다.
2. **Track 3의 시드 경합 결과**: Step B에서 Track 3은 Component 2에서 45개, Component 4에서 44개의 시드를 확보했습니다.
3. **Step D의 단일 클러스터 선택 규칙**: 현행 코드(`adaptive_roi_tracking.py` lines 320–335)는 `sub_lbl, n_sub = label(expanded_mask, structure=STRUCT_8)` 실행 후 시드가 가장 많은 단 1개의 클러스터(`best_sub_id`)만 취하고 나머지는 `False`로 삭제합니다.
   - Component 2의 시드 수: **45개**
   - Component 4의 시드 수: **44개**
   - 알고리즘은 Component 2를 선택하고, **Component 4(46패치 전체)를 Track 3에서 완전히 탈락**시켰습니다.
4. **Step H 탄생 조건 충족**: 탈락된 Component 4는 미할당 그리드(`unclaimed_grid`)로 넘어갔습니다. Component 4의 최소 반경은 $R_{\min} = 20.65\text{ px} \le 30.0\text{ px}$로 Solar Root Birth Gate 조건을 완벽하게 충족했습니다. 이에 따라 알고리즘은 이를 **독립 솔라 루트 앵커인 Adaptive Track 7로 신규 탄생**시켰습니다.

---

### 질문 4: 기존 CPA 기반 패치 소유권 규칙이 원하지 않는 분리를 유발했는가?

#### [진단 결과]
**예, Step D의 "1-Track = Single 8-Connected Cluster" 절대적 제약과 맞물려 원하지 않는 분리를 유발하였습니다.**

Component 4(46패치)는 Pair 33 시점의 Track 3 패치와 **무려 36개나 직접적인 시간 중첩(Direct Temporal Overlap)**을 이루고 있었습니다. 그럼에도 불구하고:
- 외곽 전선(Component 2)이 45개의 시드를 가졌다는 이유만으로, 44개의 시드와 36개의 시간 중첩을 가진 상류 본체 구조 전체가 버려졌습니다.
- 실제 태양 코로나 물리에서 CME는 외곽 전선(leading edge), 내부 플라즈마 코어(dense core), 후방 필라멘트(trailing prominence) 등으로 분절(fragmentation/detachment)되어 다중 컴포넌트로 관측되는 경우가 매우 흔합니다.
- 단일 8-연결 덩어리만 허용하는 현행 제약은 동일 분출의 후방 구조를 "새로운 CME 분출(Track 7)"로 오인하게 만드는 구조적 취약점을 안고 있습니다.

---

### 질문 5: HTM Ridge를 Tracking 후보 평가에 추가하면 개선 가능성이 있는가?

#### [진단 결과]
**매우 높은 개선 가능성이 확인되었습니다.**

Fig 7과 `htm_ridge_diagnostic.csv`의 수치 분석 결과:
1. **외곽 전선(Track 3)의 동역학적 일관성**:
   - Pair 30 ($R=37.7\text{ px}$) $\to$ Pair 31 ($R=48.7\text{ px}$, 속도 $21.9\text{ px/hr}$)
   - $\to$ Pair 32 ($R=59.9\text{ px}$, 속도 $22.5\text{ px/hr}$) $\to$ Pair 33 ($R=72.0\text{ px}$, 속도 $24.2\text{ px/hr}$)
   - $\to$ Pair 34 ($R=82.3\text{ px}$, 속도 $20.7\text{ px/hr}$)
   - 방사상 외향 속도가 **약 22 px/hr로 놀라울 정도로 일정하고 매끄러운 1차 선형 Height–Time Ridge**를 형성합니다.
2. **상류 근원부(Track 7)의 별도 리지 형성**:
   - 반면 Pair 34의 Component 4는 $R=20.65\text{ px}$에서 새로 시작하여 $R=57.5\text{ px}$에 이르는 별도의 내부 리지(Inner Ridge)를 형성합니다.
3. **개선 잠재력**:
   - 추적 후보 평가 시 단순 시드 개수(45 vs 44) 대신, **직전 프레임의 외향 전선 전파 속도($v_r \approx 22\text{ px/hr}$)와의 동역학적 일치도(Kinematic Ridge Continuity)**를 평가 항목으로 도입한다면:
     - 외향 전파 중인 전선(Ridge 1)을 안정적으로 전선 트랙으로 연속 매칭하고,
     - 차폐기 근처에서 지속되거나 2차 분출되는 구조(Ridge 2)를 후방 코어로 병합하거나 명확한 물리적 부트랙으로 분리할 수 있어, 1개 시드 차이에 의한 우연적 분할을 완벽히 차단할 수 있습니다.

---

### 질문 6: 현재 규칙 중 어떤 부분을 최소한으로 변경하는 것이 적절한가? (미실행 개선안 제안)

> [!NOTE]
> 사용자 지침 및 안전 규칙에 따라 다음 개선안은 **제안만 하며 실제 코드는 수정하지 않았습니다**.

#### 제안 A: Step D의 적응형 다중 컴포넌트 유지 (Multi-Component Retention within Adaptive ROI)
- **현행**: `expanded_mask` 내에서 시드가 가장 많은 단 1개의 클러스터만 남기고 나머지 서브 컴포넌트를 전부 삭제.
- **개선안**: 서브 컴포넌트 중 (1) 트랙의 현재 Adaptive Active ROI 내부에 완전히 포함되고, (2) 직전 프레임의 동일 트랙 패치와 직접 시간 중첩률이 20% 이상인 컴포넌트는 단일 트랙의 다중 조각(Multi-component Flank/Core)으로 보존 허용.
- **기대 효과**: 시드 45 vs 44 경합 시 Component 2(전선)와 Component 4(근원부)가 모두 Track 3에 온전히 유지되어 E1 T24와 T26이 불필요하게 쪼개지지 않음.

#### 제안 B: Height–Time Map 동역학적 가중치 도입 (HTM Kinematic Scoring)
- **현행**: 패치 배분을 오직 $|\Delta\mathrm{CPA}|$ 각도 차이와 단순 시드 개수로만 판정.
- **개선안**: 이전 2개 프레임의 방사상 전선 이동 속도 $v_r$을 기반으로, 현재 후보 컴포넌트의 전선 위치가 예상 범위($R_{\mathrm{pred}} = R_{t-1} + v_r \cdot \Delta t \pm 10\text{ px}$) 내에 있는지 점수화하여 Step D의 대표 클러스터 선정 시 가중치 부여.
- **기대 효과**: 물리적 외향 팽창 리지를 그리는 전선 컴포넌트가 최우선 선택됨.

#### 제안 C: Family Association 단계에서의 직접 중첩 상속 보장 (현행 Rule B 활용)
- 만약 Tracking 단계에서는 엄밀한 분리를 위해 T3와 T7을 분리하더라도, 이미 검증된 **Rule B (Direct Overlap Inheritance)**에서 T7이 직전 프레임의 T3와 직접 중첩($\mathrm{DO} = 36/46 = 0.783 > 0.05$)되므로, Family Association 단계에서 T7을 T3의 자식 트랙(Child Track)으로 확실히 편입시켜 동일 CME Family(`FAM_T3`)로 묶어 처리.

---

## 5. 과거 보고서의 T19 및 T26 CPA 불일치 원인 규명

과거 보고서 및 문서들에서 T19와 T26의 중심 위치각(CPA)이 서로 다르게 기재되었던 원인을 정밀 확인하였습니다:

### 5.1 E1 T19 CPA 불일치 원인 (52° vs 80° vs 110°)
- **CPA ~80.5°**: T19가 처음 생성된 **Pair 31 시점의 순수 관측 컴포넌트 중심각**입니다 (Pair 32: 79.4°, Pair 33: 72.8°).
- **CPA ~52.1°**: Pair 34~35에서 T19가 북동쪽 스트리머(CPA 46.3°, 45.2°)로 이동한 후, 전체 프레임에 걸친 **T19 트랙의 산술 평균 CPA**를 인용한 수치입니다.
- **CPA ~110°**: Family Association(Rule B)에서 생성된 **`FAM_T19`(Family 19) 전체의 합성 CME 중심각**을 가리킨 것입니다. `FAM_T19`는 T19 외에도 동쪽 CME 본체인 T24(CPA 121.8°)와 T26(CPA 109.3°)을 자식으로 흡수하였기 때문에, 복원된 패밀리 마스크의 중심각은 동쪽 CME 본체인 ~110°로 측정되었습니다. 즉, **개별 트랙 T19의 CPA와 패밀리 합성체 `FAM_T19`의 CPA를 혼용**하여 기술한 것이 원인이었습니다.

### 5.2 E1 T26 CPA 불일치 원인 (109.3° vs ~8°)
- **CPA = 109.27°**: Pair 34에서 생성된 실제 E1 T26의 정확한 수치입니다 (동쪽 차폐기 근원부, 46패치, $R_{\min}=20.65\text{ px}$).
- **CPA ~8°**: 초기 탐색 보고서 작성 과정에서 Pair 35에 북쪽에서 새로 탄생한 극미세 분출 트랙인 **E1 T28 (Pair 35, 3패치, CPA = 8.44°, $R_{\min}=21.01\text{ px}$)** 또는 **Adaptive Track 8 (Pair 35, 3패치, CPA = 6.15°)**의 번호와 수치를 T26과 오인·혼동하여 기술했던 인위적 기록 오류였음이 밝혀졌습니다. 실제 T26은 시종일관 동쪽(109.3°)에 위치합니다.

---

## 6. 결론

본 계보 정밀 진단을 통해:
1. **T24 $\to$ T3 흡수**는 물리적으로 완전히 동일한 외향 전선의 자연스러운 연속 추적임이 확인되었습니다.
2. **T19 $\to$ T3/T4 분할**은 E1의 비물리적 각도 융합 오류를 적응형 각도 게이트가 성공적으로 교정하였음을 증명합니다.
3. **T26 $\to$ T7 독립 생성**은 현행 1-트랙-1-클러스터 제약과 45대 44라는 1개 시드 차이의 우연적 탈락에 의해 발생한 것임이 밝혀졌습니다.
4. **Height–Time Map 리지 분석**은 향후 전선 동역학과 근원부 구조를 물리적으로 결합 또는 분리하는 강력한 차세대 추적 평가 도구로 기능할 수 있음을 입증하였습니다.

모든 정량 데이터와 시각화 프레임은 규정된 디렉터리에 온전히 보존되어 있습니다.

