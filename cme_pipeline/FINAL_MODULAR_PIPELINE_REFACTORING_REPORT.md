# Stereo_COR2 — Modular Pipeline Refactoring 최종 종합 보고서

**프로젝트 식별자**: `cme_pipeline`  
**대상 데이터**: 2010-12-12 STEREO-A COR2 Beacon d7 (Pairs 29–60, 32개 전체 관측 페어)  
**기준 파이프라인**: Baseline Rule B (E1 Tracking + Signed Dynamic PA Recovery + Stage 32B-2 Rule B Direct Overlap + Family Birth Radius Gate + Size Gate + Island-First Segmentation + External Validation)  
**실행 인터프리터**: `/home/aababc/stereo_cor2/venv/bin/python`

---

## 1. 개요 및 달성 목표 (Executive Summary)

Stereo_COR2 프로젝트의 7대 핵심 과학 처리 단계(검출/추적, 미세 복원, 패밀리 구성, 발생 기원 게이트, 크기 게이트, 서브패치 화소 세그멘테이션, 외부 공인 카탈로그 재검증)를 기존 모놀리식 스크립트에서 **독립 실행 가능한 클래스 기반 모듈형 패키지(`cme_pipeline/`)**로 완전히 리팩터링하였습니다.

### 핵심 달성 성과
1. **기존 과학적 수치 및 마스크 100% 완전 재현 (Bit-Exact Reproducibility)**:
   - Cumulative Recovered Patches: **5,174개** (기존 결과와 비트 단위 일치 `np.array_equal == True`, diff count = 0)
   - Cumulative Segmented Pixels: **50,665픽셀** (기존 결과와 비트 단위 일치 `np.array_equal == True`, diff count = 0)
   - Admitted Families (15개) 및 Admitted Tracks (37개): 기존 승인 목록과 100% 완전 일치
   - External Validation 일치도: SEEDS(0.5242, 1.7413 $R_\odot$), CACTus All(0.2715), CACTus CME 0024(0.7414), MVCC(0.4032, 1.1843 $R_\odot$), Göttingen(0.5547) 전 지표 4자리 소수점 완전 일치
2. **독립 Stage 인터페이스 및 명시적 Typed Data Contract**:
   - `PipelineStage` 추상 클래스 기반 7대 스테이지 구현.
   - dataclass 기반 입출력 데이터 모델(`TrackingOutput`, `RecoveryOutput`, `AssociationOutput`, `BirthGateOutput`, `SizeGateOutput`, `SegmentationOutput`, `ValidationOutput`)로 전역 상태 변이 제거.
3. **Stage-Level 로컬 파일 캐싱 및 증분 실행 (`--from-stage`)**:
   - 알고리즘 버전, 설정 파라미터, 입력 데이터 해시 기반 결정론적 SHA-256 핑거프린트 캐시.
   - 캐시 히트 시 전체 파이프라인 0.28초 내 복원.
   - `--from-stage` 옵션으로 특정 단계 이후만 선별 재실행 가능.
4. **실험 격리 및 단위 테스트 완비**:
   - 각 실험 런은 `runs/<run_name>/` 디렉토리에 완전 격리되어 기존 산출물 보존.
   - 14개 자동화 단위 테스트 통과 (`tests/` discover: 14 tests, 0 failures, 0 errors).

---

## 2. 모듈 아키텍처 및 디렉토리 구조 (Module Architecture)

```text
cme_pipeline/
├── __init__.py               # 패키지 초기화 및 버전
├── __main__.py               # CLI 엔트리포인트 (run, status)
├── README.md                 # 개발자 가이드 및 아키텍처 설명서
├── core/
│   ├── stage.py              # PipelineStage 추상 클래스 및 7대 Typed Dataclass
│   ├── pipeline.py           # 순차 파이프라인 오케스트레이터 및 보고서 생성기
│   ├── config.py             # PipelineConfig 및 세부 스테이지 설정 모델
│   ├── context.py            # PipelineContext (런타임 경로, 디렉토리 격리, 로거)
│   └── cache.py              # StageCache (SHA-256 핑거프린트 유효성 및 무효화)
├── stages/
│   ├── tracking/
│   │   └── e1_tracking.py    # E1TrackingStage (전처리 및 Exp 11 Gate OFF 트래킹)
│   ├── recovery/
│   │   └── signed_pa_recovery.py # SignedPARecoveryStage (Exp 12 v2 Signed Dynamic PA 복원)
│   ├── association/
│   │   └── pure_do_association.py # PureDOFamilyAssociationStage (Stage 32B-2 Rule B DO)
│   ├── gates/
│   │   ├── birth_radius_gate.py   # FamilyBirthRadiusGateStage (r_min <= 30.0 px)
│   │   └── size_gate.py           # FamilySizeGateStage (max_det >= 5 & 복원 마스크 생성)
│   ├── segmentation/
│   │   └── island_first_segmentation.py # IslandFirstSegmentationStage (Stage 26 서브패치 세그멘테이션)
│   └── validation/
│       └── external_validation.py       # ExternalValidationStage (5대 카탈로그 일치도 재계산)
├── experiments/
│   ├── runner.py             # 파이썬 스크립트용 실행 인터페이스
│   └── comparison.py         # 런 간 수치 배열 및 지표 회귀 비교 도구
└── utils/
    ├── geometry.py           # WCS 래스터, S^1 순환 각도 차이, 최소 포괄 호 계산
    └── io.py                 # SHA-256 해시, JSON/YAML 직렬화 헬퍼

configs/
└── baseline_rule_b.yaml      # 표준 기준 설정 파일

runs/
└── baseline_rule_b/          # 실행 산출물 완전 격리 디렉토리
    ├── manifest.json         # 실행 시점, 런타임, 캐시 메타데이터
    ├── pipeline.log          # 전체 실행 로그
    ├── report.md             # 마크다운 자동 종합 요약표
    ├── .cache/               # 스테이지별 pkl 캐시 및 cache_manifest.json
    ├── stages/               # 스테이지별 상세 CSV/NPY
    ├── tables/               # 표준 번호 체계 CSV 테이블 (01~09)
    └── arrays/               # 복원 및 최종 세그멘테이션 과학 배열 (.npy, .npz)
```

---

## 3. 각 Stage의 입력과 출력 데이터 규격 (Data Contracts)

엄격한 실행 순서:
$$\text{Tracking} \rightarrow \text{Recovery} \rightarrow \text{Association} \rightarrow \text{Birth Gate} \rightarrow \text{Size Gate} \rightarrow \text{Segmentation} \rightarrow \text{Validation}$$

| 순서 | Stage 클래스 | 입력 (Input Contract) | 출력 (Output Contract) | 생성 테이블 및 과학 배열 |
|:---:|:---|:---|:---|:---|
| **1** | `E1TrackingStage` | `None` (YAML 설정 및 FITS/NPY 원본) | `TrackingOutput` | `01_full_day_frame_manifest.csv`<br>`patch_component_labels.npy`<br>`patch_activation_grid4.npy` |
| **2** | `SignedPARecoveryStage` | `TrackingOutput` | `RecoveryOutput` | `DynamicPARecoveryRunner` (동적 PA 복원 연산자) |
| **3** | `PureDOFamilyAssociationStage` | `TrackingOutput` & `RecoveryOutput` | `AssociationOutput` | `02_full_day_e1_track_states.csv`<br>`03_full_day_parent_child_edges.csv` |
| **4** | `FamilyBirthRadiusGateStage` | `AssociationOutput` | `BirthGateOutput` | `birth_gated_family_ids` (23개 통과) |
| **5** | `FamilySizeGateStage` | `BirthGateOutput` | `SizeGateOutput` | `04_full_day_family_summary.csv`<br>`05_full_day_size_gate_decisions.csv`<br>`recovered_patch_masks_pipeline_b_full_day.npy`<br>`family_recovered_patch_masks.npz` |
| **6** | `IslandFirstSegmentationStage`| `SizeGateOutput` | `SegmentationOutput` | `06_full_day_framewise_pixel_summary.csv`<br>`07_full_day_family_pixel_statistics.csv`<br>`final_pixel_segmentation_union.npy`<br>`family_pixel_masks.npz` |
| **7** | `ExternalValidationStage` | `SegmentationOutput` | `ValidationOutput` | `08_comparison_external_validation_metrics.csv`<br>`09_frame_level_external_validation.csv` |

> [!NOTE]
> **Birth Gate 순서 무결성**: 트랙 단계에서 사전에 트랙을 제거하지 않고, 219개 전체 트랙으로 Family Association을 완결한 뒤 Family 단위의 앵커 존재 여부(`anchor_count > 0`)로 게이트를 적용하여 트랙 간 연관 정보 유실을 원천 방지하였습니다.

---

## 4. 실행 명령어 및 CLI 사용법 (CLI Execution)

```bash
# 1. 표준 베이스라인 전체 실행
python -m cme_pipeline run --config configs/baseline_rule_b.yaml

# 2. 특정 Stage부터 선별 재실행 (이전 단계는 캐시 재사용)
python -m cme_pipeline run --config configs/baseline_rule_b.yaml --from-stage segmentation

# 3. 캐시 상태 및 핑거프린트 조회
python -m cme_pipeline status --config configs/baseline_rule_b.yaml

# 4. 전체 캐시 무시 강제 재계산
python -m cme_pipeline run --config configs/baseline_rule_b.yaml --force
```

---

## 5. 캐시 동작 검증 결과 (Cache Verification)

1. **최초 실행 (Cold Run)**:
   - 7개 Stage 전체 순차 연산 완료: 총 43.8초 소요.
   - 각 Stage별 SHA-256 핑거프린트 및 직렬화 파일이 `runs/baseline_rule_b/.cache/`에 저장됨.
2. **동일 설정 재실행 (Warm Run)**:
   - 7개 Stage 전체 `CACHE HIT` 판정.
   - 총 소요 시간: **0.28초** (150배 이상 가속).
3. **`--from-stage segmentation` 증분 재실행**:
   - `tracking`, `recovery`, `association`, `birth_gate`, `size_gate`: `CACHE HIT` (0.09초).
   - `segmentation`, `validation`: 캐시 무효화 후 재계산 (1.06초).
   - 총 1.19초 만에 다운스트림 단계만 완벽하게 갱신됨을 확인.

---

## 6. Baseline 정량 수치 및 마스크 일치 검증 (Phase 4 Reproduction)

기존 `validation/frozen/20101212/full_day_pipeline_validation/01_full_day_pipeline/` 산출물과 신규 모듈 산출물을 비트 단위 및 정량 지표로 대조한 결과입니다.

### [표 1] 핵심 거버넌스 및 과학적 마스크 일치 비교표
| 검증 항목 | 역사적 기준값 (Historical Baseline) | 신규 모듈값 (Modular Output) | 오차 (Diff) | 판정 |
|:---|:---:|:---:|:---:|:---:|
| 유효 관측 페어 수 (Pairs 29–60) | 32 | 32 | 0 | **일치** |
| Gate OFF E1 트랙 총수 | 219 | 219 | 0 | **일치** |
| 생성된 후보 패밀리 수 | 185 | 185 | 0 | **일치** |
| Birth Gate 통과 패밀리 수 | 23 | 23 | 0 | **일치** |
| Size Gate 통과 최종 승인 패밀리 수 | **15** | **15** | 0 | **일치** |
| 최종 승인 트랙 수 | **37** | **37** | 0 | **일치** |
| 누적 회복 패치 수 (Recovered Patches) | **5,174** | **5,174** | **0** | **Bit-exact (True)** |
| 누적 세그멘테이션 화소 수 (Segmented Pixels) | **50,665** | **50,665** | **0** | **Bit-exact (True)** |

### [표 2] 외부 공인 카탈로그 재검증 지표 비교표
| 외부 레퍼런스 카탈로그 | 평가 프레임 수 | Baseline Angular IoU | Modular Angular IoU | Baseline Radial Error | Modular Radial Error | 일치 판정 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SEEDS** | 24 | **0.5242** | **0.5242** | **1.7413 $R_\odot$** | **1.7413 $R_\odot$** | **100% 일치** |
| **CACTus (All CMEs)** | 8 (43) | **0.2715** | **0.2715** | N/A | N/A | **100% 일치** |
| **CACTus (CME 0024)** | 11 | **0.7414** | **0.7414** | N/A | N/A | **100% 일치** |
| **MVCC / CORSET** | 22 | **0.4032** | **0.4032** | **1.1843 $R_\odot$** | **1.1843 $R_\odot$** | **100% 일치** |
| **Göttingen BestOf** | 2 | **0.5547** | **0.5547** | N/A | N/A | **100% 일치** |

---

## 7. 향후 새로운 알고리즘 추가 가이드 (Extensibility Guide)

새로운 실험을 수행할 때 기존 검증 베이스라인 코드를 수정하지 않고 독립적으로 확장할 수 있습니다.

### 7.1 새 트래킹 알고리즘 추가 (예: E3 다대다 트래킹)
1. 파일 생성: `cme_pipeline/stages/tracking/e3_tracking.py`
   - `PipelineStage`를 상속받고 `name = "tracking"` 정의.
   - `run(inputs, config, context) -> TrackingOutput` 구현.
2. 설정 모델 확장: `cme_pipeline/core/config.py`의 `TrackingConfig`에 필요한 파라미터 추가.
3. YAML 설정 작성: `configs/experiment_e3_tracking.yaml`

### 7.2 새 패밀리 연관 알고리즘 추가 (예: Hybrid DO + Anchor PA Separation)
1. 파일 생성: `cme_pipeline/stages/association/hybrid_do_pa_association.py`
   - `PipelineStage`를 상속받고 `name = "association"` 정의.
   - `run(inputs, config, context) -> AssociationOutput` 내에 $\Delta\text{CPA} > 45^\circ$ 분리 조건 구현.
2. YAML 설정 작성:
   ```yaml
   run_name: experiment_hybrid_pa
   association:
     method: hybrid_do
     do_threshold: 0.05
     pa_separation_threshold_deg: 45.0
   ```
3. 실행 및 비교:
   ```bash
   python -m cme_pipeline run --config configs/experiment_hybrid_pa.yaml
   ```
   이후 `cme_pipeline.experiments.comparison.compare_runs("runs/baseline_rule_b", "runs/experiment_hybrid_pa")`로 차이점 자동 분석.

---

## 8. 구현 제약 및 참고 사항 (Limitations)

1. **로컬 파일 시스템 캐시 한정**: 현재 캐시는 로컬 파일(`runs/<run_name>/.cache/`) 기반이므로 분산 클라우드 환경에서는 공유 볼륨이 필요합니다.
2. **트래킹 전처리 범위**: 본 리팩터링에서는 2010-12-12 관측일에 대해 기검증된 Exp 11 트래킹 배열을 완벽히 재현하도록 구성되었습니다. 다른 날짜의 새로운 관측 시퀀스를 처리할 때는 원본 FITS 수집 및 전처리 파이프라인(`01_d7_baseline_segmentation`)이 선행 실행되어야 합니다.

