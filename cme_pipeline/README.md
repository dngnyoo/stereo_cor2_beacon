# CME Pipeline: Modular Architecture & Developer Guide

`cme_pipeline`은 STEREO-A COR2 코로나 질량 방출(CME)의 검출, 추적, 미세 구조 복원, 패밀리 연관(Family Association), 세그멘테이션 및 외부 공인 카탈로그 검증을 클래스 기반 모듈형 구조로 실행하는 과학 파이프라인 패키지입니다.

기존 각 실험(Experiments 11–18)의 과학적 알고리즘과 수치 결과를 100% 비트 단위로 보존하면서, 향후 각 Stage의 알고리즘 변형을 독립적으로 교체하고 비교할 수 있도록 설계되었습니다.

---

## 1. Pipeline Architecture (파이프라인 아키텍처)

파이프라인은 현재 **8개 Stage**로 구성되며, 엄격한 실행 순서를 준수합니다.

```text
[1. E1TrackingStage]
       │  (Detection & 1:1 Tracking)
       ▼
[2. SignedPARecoveryStage]
       │  (Exp 12 v2 Signed Dynamic PA Patch Recovery)
       ▼
[3. PureDOFamilyAssociationStage]
       │  (Stage 32B-2 Rule B Direct Overlap Association)
       ▼
[4. FamilyBirthRadiusGateStage]
       │  (Family-level Solar Root Anchor Filter, r_min <= 30 px)
       ▼
[5. FamilySizeGateStage]
       │  (Family-level 5-Patch Size Filter & Recovery Union)
       ▼
[6. IslandFirstSegmentationStage]
       │  (Stage 26 Sub-Patch Pixel Segmentation, Z > 0, Area >= 16 px)
       ▼
[7. CMEProductsStage]
       │  (Independent CME Catalog + PNG/GIF Diagnostic Visualization)
       ▼
[8. ExternalValidationStage]
          (SEEDS, CACTus, MVCC/CORSET, Göttingen Recalculation)
```

> [!IMPORTANT]
> **실행 순서 원칙**: Track 단위의 Birth Radius 필터링으로 Track을 사전에 제거하지 않습니다. 모든 219개 Gate OFF 트랙을 대상으로 Family Association을 먼저 수행한 후, Family 단위로 Birth Gate와 Size Gate를 순차 평가합니다.

### 패키지 디렉토리 구조
```text
cme_pipeline/
├── __init__.py
├── __main__.py               # CLI 엔트리포인트 (run, status)
├── core/
│   ├── config.py             # YAML 설정 모델 및 데이터클래스
│   ├── context.py            # 런타임 환경, 디렉토리 격리, 로거
│   ├── cache.py              # 로컬 파일 기반 StageCache 매니저
│   ├── stage.py              # PipelineStage 추상 클래스 및 입출력 데이터 모델
│   └── pipeline.py           # 순차 파이프라인 오케스트레이터
├── stages/
│   ├── tracking/             # E1TrackingStage
│   ├── recovery/             # SignedPARecoveryStage
│   ├── association/          # PureDOFamilyAssociationStage
│   ├── gates/                # FamilyBirthRadiusGateStage, FamilySizeGateStage
│   ├── segmentation/         # IslandFirstSegmentationStage, CMEProductsStage
│   └── validation/           # ExternalValidationStage
├── experiments/
│   ├── runner.py             # 파이썬 코드용 실험 실행기
│   └── comparison.py         # 런 간 수치 및 마스크 회귀 비교 유틸리티
└── utils/
    ├── geometry.py           # S^1 순환 방위각 차이, 호 최소 포괄, WCS 래스터
    └── io.py                 # 해시 핑거프린트, 입출력 헬퍼
```

---

## 2. Stage Data Contracts (데이터 규격)

모든 Stage는 `PipelineStage` 추상 클래스를 상속하며 명확한 Typed Dataclass를 입력받아 명시적인 Dataclass를 반환합니다.

| Stage Name | Input Contract | Output Contract | 핵심 산출물 |
|:---|:---|:---|:---|
| `tracking` | `None` (Config 기반 로드) | `TrackingOutput` | `manifest_df`, `patch_component_labels` (32, 64, 64), `track_states_df` (219 트랙), `geometry` |
| `recovery` | `TrackingOutput` | `RecoveryOutput` | `recovery_func` (Dynamic PA 복원 연산자) |
| `association` | `TrackingOutput` / `RecoveryOutput` | `AssociationOutput` | `track_to_family`, `family_members`, `edges_df` (222개 후보 에지), `track_membership_df` |
| `birth_gate` | `AssociationOutput` | `BirthGateOutput` | `birth_gated_family_ids` (23개 통과 패밀리), `rejected_family_reasons` |
| `size_gate` | `BirthGateOutput` | `SizeGateOutput` | `admitted_family_ids` (15개 패밀리), `admitted_track_ids` (37개 트랙), `recovered_patch_masks` (**5,174개 패치**) |
| `segmentation`| `SizeGateOutput` | `SegmentationOutput` | `final_pixel_segmentation_union` (**50,665픽셀**), `framewise_pixel_summary_df` |
| `products` | `SegmentationOutput` | `CMEProductsOutput` | independent catalog CSV, diagnostic PNG/GIF |
| `validation` | `CMEProductsOutput` | `ValidationOutput` | `comparison_metrics_df` (5대 카탈로그 일치도 지표) |

---

## 3. How to Run the Baseline (실행 방법)

### 3.1 단일 명령 실행
```bash
python -m cme_pipeline run --config configs/baseline_rule_b.yaml
```

### 3.2 특정 Stage부터 재실행 (`--from-stage`)
이전 Stage의 캐시는 재사용하고, 지정한 Stage 및 그 이후의 모든 Downstream Stage만 무조건 재실행합니다:
```bash
python -m cme_pipeline run --config configs/baseline_rule_b.yaml --from-stage association
```

### 3.3 캐시 상태 확인 (`status`)
각 Stage의 캐시 유효성, 핑거프린트 해시 및 캐시 생성 시점을 조회합니다:
```bash
python -m cme_pipeline status --config configs/baseline_rule_b.yaml
```

### 3.4 전체 강제 재실행 (`--force`)
```bash
python -m cme_pipeline run --config configs/baseline_rule_b.yaml --force
```

---

## 4. Cache Invalidation Mechanics (캐시 동작 원리)

- **Fingerprint 기반 유효성 검증**:
  각 Stage는 실행 전 자신의 구현 버전(`version`), 알고리즘 설정(`config`), 그리고 Upstream 입력 데이터의 요약 해시를 결합하여 결정론적 SHA-256 핑거프린트를 생성합니다.
  - 핑거프린트와 캐시 파일이 정확히 일치하면 `CACHE HIT`으로 디스크에서 즉시 로드됩니다 (수십 밀리초 소요).
  - 설정 변경, 코드 버전 변경, 입력 데이터 변경 시 자동으로 `CACHE MISS`가 발생하여 재계산됩니다.
- **캐시 격리**:
  각 실험 런은 `runs/<run_name>/` 디렉토리에 완전히 격리되어 저장되므로, 서로 다른 설정의 실험 결과가 캐시 키나 결과 파일을 덮어쓰거나 오염시키지 않습니다.

---

## 5. How to Add a New Tracking Variant (새 트랙커 추가 가이드)

1. `cme_pipeline/stages/tracking/` 하위에 새 모듈 생성 (예: `e3_tracking.py`):
   ```python
   from cme_pipeline.core.stage import PipelineStage, TrackingOutput

   class E3TrackingStage(PipelineStage):
       name = "tracking"
       version = "1.0.0"

       def run(self, inputs, config, context) -> TrackingOutput:
           # 새 E3 다대다 트래킹 로직 구현 또는 기존 함수 호출
           ...
           return TrackingOutput(...)
   ```
2. `cme_pipeline/core/config.py`의 `TrackingConfig`에 필요한 옵션 추가 (예: `search_radius_px`).
3. 파이프라인 실행 시 커스텀 스테이지 리스트를 전달하거나 설정 파일에서 `tracking.method: e3`로 선택.

---

## 6. How to Add a New Family Association Variant (새 패밀리 연관 추가 가이드)

1. `cme_pipeline/stages/association/` 하위에 새 모듈 생성 (예: `hybrid_do_pa_association.py`):
   ```python
   from cme_pipeline.core.stage import PipelineStage, AssociationOutput

   class HybridDOAssociationStage(PipelineStage):
       name = "association"
       version = "1.0.0"

       def run(self, inputs, config, context) -> AssociationOutput:
           # Hybrid Direct Overlap + Anchor PA Separation 로직 구현
           ...
           return AssociationOutput(...)
   ```
2. `cme_pipeline/core/config.py`의 `AssociationConfig`에 `method: hybrid_do` 및 `pa_separation_threshold_deg: 45.0` 옵션 추가.
3. YAML 설정 파일 생성 (`configs/hybrid_variant.yaml`) 후 실행:
   ```bash
   python -m cme_pipeline run --config configs/hybrid_variant.yaml
   ```

---

## 7. How to Compare Multiple Runs (실험 결과 비교)

두 실험 런의 결과 배열 및 지표를 정량 비교하려면 `cme_pipeline.experiments.comparison` 모듈을 사용합니다:

```python
from cme_pipeline.experiments.comparison import compare_runs

diff = compare_runs("runs/baseline_rule_b", "runs/experimental_run")
print("Recovery masks identical?:", diff["array_comparisons"]["recovered_patch_masks"]["is_equal"])
print("Segmentation pixels identical?:", diff["array_comparisons"]["final_pixel_segmentation_union"]["is_equal"])
```



---


---

## 8. CME Products Stage: Independent Catalog & Visualization

`products` is a separate **Stage 7 of 8**, positioned between Segmentation and External Validation. It receives `SegmentationOutput`, produces `CMEProductsOutput`, and preserves the original `SegmentationOutput` for unchanged external validation. It does **not** use external catalog values to construct event candidates.

The `feature/independent-cme-catalog` branch enables the following by default:

```yaml
products:
  catalog: true
  visualization: true
  animation: true
```

Generated files in `runs/<run_name>/`:

| File | Content |
|:---|:---|
| `tables/cme_event_catalog.csv` | Candidate Family event metrics; exploratory speed quality flags |
| `tables/cme_frame_measurements.csv` | Per-family per-frame radius, CPA, angular width, segmented area |
| `visualizations/segmentation_overlays/frame_*.png` | RD image with union segmentation overlay for every frame |
| `visualizations/family_tracking/*.png` | Family peak-area frame snapshot |
| `visualizations/height_time/*.png` | Maximum mask-radius time series and exploratory linear fit |
| `visualizations/cme_animation.gif` | Time-ordered animation of union segmentation overlays |

To generate or regenerate these outputs without recomputing Segmentation:

```bash
python -m cme_pipeline run --config configs/baseline_rule_b.yaml --from-stage products
```

When product files are missing, the Products stage detects that condition and regenerates them even if its cached object exists. Set `products.visualization: false` if only the CSV catalog is needed. The `animation` option controls GIF creation and only applies when visualization is enabled.

Speed is an exploratory line-fit using the outermost segmented pixel over time. It should not be interpreted as a verified CME leading-front speed. The exported families are **candidates**, not independently confirmed CMEs. The reference benchmark still uses precomputed Detection inputs, not raw-FITS-to-catalog processing.

Detailed clone-to-results instructions: [repository README](../README.md).
