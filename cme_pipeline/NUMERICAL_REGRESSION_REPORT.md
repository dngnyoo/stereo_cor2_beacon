# CME Pipeline — Phase 3 Numerical Regression & Integration Validation Report

**검증 대상 데이터셋**: `2010-12-12 STEREO-A COR2 Beacon d7` (Pairs 29–60, 32개 프레임 전수)  
**검증 스위트**: [`tests/test_tracking_integration.py`](file:///home/aababc/stereo_cor2/tests/test_tracking_integration.py) + 기존 45개 단위 테스트  
**실행 상태**: **전수 통과 (11/11 Integration Checks OK + 45/45 Unit Tests OK)**

---

## 1. 핵심 수치 회귀 검증 결과표 (Reference vs Integrated Pipeline)

| 실험 조건 (YAML Configuration) | 지표 | 기존 참조값 (Reference) | 통합 파이프라인 측정값 | 비트 단위 일치 여부 (`np.array_equal`) |
| :--- | :--- | :---: | :---: | :---: |
| **E1 + Rule B** (`configs/tracking_e1_rule_b.yaml`) | Total Tracks | **219** | **219** | PASS |
| | Candidate / Admitted Families | **185 / 15** | **185 / 15** (37 tracks) | PASS |
| | Recovered Patches | **5,174** | **5,174** | **`True` (0 diff)** |
| | Final Segmented Pixels | **50,665** | **50,665** | **`True` (0 diff)** |
| | Catalog Angular IoU (SEEDS / CACTus CME24 / MVCC / Göttingen) | 0.5242 / 0.7414 / 0.4032 / 0.5547 | **0.5242 / 0.7414 / 0.4032 / 0.5547** | EXACT MATCH |
| **A3 Fixed + Identity** (`configs/tracking_a3_fixed_identity.yaml`, `outward_gate=True, tol=6.0`) | Total Tracks | **50** (`a3_fixed`) | **50** | PASS |
| | Tracking Ownership Array vs Exp 04 | `a3_fixed_ownership_masks.npy` | 일치 | **`True` (0 diff)** |
| | Admitted Families / Recovered Patches / Final Pixels | *(신규 엔드투엔드 계측)* | **43 / 5,600 / 52,578** | PASS |
| | Catalog Angular IoU (SEEDS / CACTus CME24 / MVCC / Göttingen) | - | 0.5451 / 0.7450 / 0.4606 / 0.5120 | PASS |
| **A4 + Identity** (`configs/tracking_a4_identity.yaml`, `outward_gate=True, tol=6.0`) | Total Tracks | **18** (`Exp 04 A4`) | **18** | PASS |
| | Tracking Ownership Array vs Exp 04 | `a4_ownership_masks.npy` | 일치 | **`True` (0 diff)** |
| | T3 Multi-Component Retention (Pair 34) | `num_components >= 2` | `num_components >= 2` | PASS |
| | Admitted Families / Recovered Patches / Final Pixels | *(신규 엔드투엔드 계측)* | **16 / 6,662 / 60,567** | PASS |
| | Catalog Angular IoU (SEEDS / CACTus CME24 / MVCC / Göttingen) | - | 0.5320 / 0.8194 / 0.4783 / 0.5181 | PASS |
| **A5b + Identity** (`configs/tracking_a5b_identity.yaml`, `outward_gate=True, tol=6.0, min_pa_width=8.0, ema_alpha=0.1`) | Total Tracks | **20** (`Exp 07`) | **20** | PASS |
| | Admitted Families | **12** | **12** | PASS |
| | Recovered Patches | **6,549** | **6,549** | **`True` (0 diff)** |
| | Final Union Pixels | **59,899** | **59,899** | **`True` (0 diff)** |
| | Individual Track Pixel Sum / Shared Overlap Pixels | **67,184 / 7,285** | **67,184 / 7,285** | **`True` (`union(tracks) == union`)** |
| | Catalog Angular IoU (SEEDS / CACTus CME24 / MVCC / Göttingen) | 0.5501 / 0.8105 / 0.4826 / 0.5181 | **0.5501 / 0.8105 / 0.4826 / 0.5181** | EXACT MATCH |
| **A5b + Rule B vs A5b + Identity** | Recovered Patches & Final Union Pixels | 비트 동일 | **XOR 차이 = 0** | **`True` (100% Bitwise Equal)** |

---

## 2. 아키텍처 불변량(Invariants) 및 호환성 검증 결과

1. **Detection 입력 동등성 (`test_04`)**:
   - E1, A3, A4, A5b 모두 `load_frozen_detection_inputs()`로부터 비트 단위로 동일한 `patch_activation_grid4` 및 `raw_component_labels`를 수신함을 검증 (`np.array_equal == True`).
2. **Track ID vs Component ID 비혼동 검증 (`test_05`)**:
   - 모든 방법에서 `patch_component_labels`(원본 8-연결 컴포넌트 ID)와 `track_ownership_masks`(배타적 Track ID)가 서로 명확히 구별됨을 확인 (`np.array_equal(own, comp) == False`).
   - 활성 격자 외(`~patch_activation_grid4`) 소유권 0건 및 1패치 다중 소유권 0건 검증.
3. **Predicted-Only 합성 픽셀 차단 (`test_05`)**:
   - 모든 방법의 `PREDICTED_ONLY` 상태 프레임에서 `track_ownership_masks == tid`가 정확히 **0 패치**임을 전수 확인.
4. **엄격한 파라미터 검증 및 실제 반영 (`test_02`, `test_03`)**:
   - 지원하지 않는 키(예: `a5b`에 `nonexistent_param`, `a4`에 `min_pa_width_deg`) 전달 시 조용히 무시하지 않고 즉시 `ValueError` 발생 검증.
   - `min_pa_width_deg`를 `8.0` $\rightarrow$ `40.0`으로 변경 실행 시 추적 결과(`20` $\rightarrow$ `23` 트랙) 및 분할 픽셀(`59,899` $\rightarrow$ `59,695` px)이 즉시 반영되고 `manifest.json` provenance에 정확히 기록됨.
5. **기존 테스트 스위트 역방향 호환성**:
   - 기존 단위 테스트 45건(`test_a4_tracking`, `test_a5_tracking`, `test_a5b_tracking`, `test_a5b_end_to_end`, `test_association_and_gates`, `test_segmentation_reproducibility`, `test_cache`, `test_circular_pa`, `test_direction`, `test_run_isolation`) 전체 통과 확인.
6. **Recovery 중첩 보존 검증 (`test_09`)**:
   - `build_track_level_masks`를 통해 A5b + Identity의 개별 트랙 합계(`67,184 px`), 공유 중첩 화소(`7,285 px`), Union 합계(`59,899 px`)를 임의 재할당 없이 보존 및 비트 일치(`np.array_equal == True`) 검증 완료.

---

## 3. 확인된 설계 주의사항 및 잔존 한계

1. **기본값(`outward_gate_enabled=False`)과 실험 검증값(`True, tol=6.0`) 구분**:
   - 기존 Exp 06 및 `test_a4_tracking` / `test_a5b_tracking`과의 하위 호환을 위해 `TrackingConfig` 클래스 자체의 기본값은 `outward_gate_enabled=False, tolerance=4.0`으로 유지하였습니다.
   - Exp 04(A3 50트랙, A4 18트랙) 및 Exp 07(A5b 20트랙)의 표준 동작을 원할 경우 공식 설정 파일(`configs/tracking_a5b_identity.yaml` 등)처럼 `parameters` 내에 명시적으로 `outward_gate_enabled: true`, `outward_gate_tolerance_px: 6.0`, `ema_alpha: 0.1`을 지정해야 합니다.
2. **Rule B 기반 다중 멤버 패밀리에서의 Per-Track 마스크 분해 한계**:
   - `build_track_level_masks`는 A5b/A4 + Identity 및 A5b + Rule B처럼 모든 승인 패밀리가 `member_count == 1`인 경우에는 완벽히 1:1 트랙 마스크를 분리합니다.
   - 다만 E1 + Rule B처럼 서로 다른 여러 Track이 하나의 병합된 Family로 묶이는 경우(`unsplit_multi_member_families`), 하위 분할(Island-First Segmentation)이 패밀리 결합 마스크 단위로 연산되므로 개별 트랙별 픽셀을 임의로 분리하지 않고 해당 다중 멤버 패밀리 목록을 별도 메타데이터(`unsplit_multi_member_families`)로 보고하도록 설계하였습니다.
3. **E1 Baseline의 `alpha` 파라미터 성격**:
   - E1은 사전 산출된 Exp 11 동결 표(`03_patch_e1_track_states.csv`)를 재사용하므로 런타임 중 CPA EMA 갱신을 재연산하지 않습니다. A3/A4/A5b에서는 `ema_alpha` 파라미터가 런타임에 직접 적용됩니다.

