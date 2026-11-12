"""Pipeline Orchestrator for CME Pipeline."""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import PipelineStage
from cme_pipeline.stages.association.v03_adaptive_roi_association import AdaptiveROIAssociationStage
from cme_pipeline.stages.association.v04_birth_anchored_association import BirthAnchoredAssociationStage
from cme_pipeline.stages.association.v01_identity_association import IdentityAssociationStage
from cme_pipeline.stages.association.v02_pure_do_rule_b import PureDOFamilyAssociationStage
from cme_pipeline.stages.gates.v01_birth_radius_gate import FamilyBirthRadiusGateStage
from cme_pipeline.stages.gates.v01_family_size_gate import FamilySizeGateStage
from cme_pipeline.stages.recovery.v01_signed_pa_recovery import SignedPARecoveryStage
from cme_pipeline.stages.segmentation.v01_island_first_segmentation import IslandFirstSegmentationStage
from cme_pipeline.stages.tracking.registry import (
    get_tracking_stage,
    resolve_and_validate_tracking_parameters,
    resolve_tracking_method_name,
)
from cme_pipeline.stages.validation.v01_external_catalog_validation import ExternalValidationStage
from cme_pipeline.stages.registry import resolve_stage
from cme_pipeline.utils.io import save_json


def _get_code_revision(repo_root: Path) -> Dict[str, Any]:
    """Safely queries current git commit hash and clean status for provenance."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"git_commit": sha, "is_dirty": bool(status)}
    except Exception:
        return {"git_commit": "UNKNOWN", "is_dirty": None}


def resolve_association_stage(method: str) -> PipelineStage:
    """Resolve association through the unified registry."""
    from cme_pipeline.stages.association.registry import get_association_stage
    return get_association_stage(method)


def resolve_stages(config: PipelineConfig) -> List[PipelineStage]:
    """Resolves and instantiates the canonical 8 stages via algorithm registries."""
    # Validate parameters immediately on resolution so unsupported keys fail fast
    resolve_and_validate_tracking_parameters(config.tracking.method, config.tracking)

    track_stage = get_tracking_stage(config.tracking.method)
    rec_stage = resolve_stage("recovery", config.recovery.method)
    assoc_stage = resolve_association_stage(config.association.method)
    birth_stage = resolve_stage("birth_gate", "v01_birth_radius_gate")
    size_stage = resolve_stage("size_gate", "v01_family_size_gate")
    seg_stage = resolve_stage("segmentation", config.segmentation.method)
    products_stage = resolve_stage("products", "default")
    val_stage = resolve_stage("validation", "v01_external_catalog_validation")

    return [
        track_stage,
        rec_stage,
        assoc_stage,
        birth_stage,
        size_stage,
        seg_stage,
        products_stage,
        val_stage,
    ]


class Pipeline:
    """Sequential pipeline runner with stage-level caching and run isolation."""

    def __init__(self, config: PipelineConfig, stages: Optional[List[Any]] = None):
        self.config = config
        self.context = PipelineContext(config)
        if stages is not None:
            self.stages: List[PipelineStage] = [cls() if isinstance(cls, type) else cls for cls in stages]
        else:
            self.stages: List[PipelineStage] = resolve_stages(config)
        self.stage_names: List[str] = [s.name for s in self.stages]

    def run(self, from_stage: Optional[str] = None, *,\n            initial_input: Any = None) -> Dict[str, Any]:
        """Executes the full pipeline or resumes from a specific stage."""
        t_start = time.perf_counter()
        effective_from_stage = from_stage or self.config.from_stage

        self.context.info("=" * 80)
        self.context.info(f"Starting CME Pipeline Run: {self.config.run_name}")
        self.context.info(f"Run Directory: {self.context.run_dir}")
        self.context.info(f"Stage Order: {' -> '.join(self.stage_names)}")
        if effective_from_stage:
            self.context.info(f"Execution Mode: Re-running from stage '{effective_from_stage}'")
        self.context.info("=" * 80)

        # Handle --from-stage cache invalidation
        if effective_from_stage:
            if effective_from_stage not in self.stage_names:
                raise ValueError(
                    f"Unknown --from-stage '{effective_from_stage}'. Available stages: {self.stage_names}"
                )
            self.context.cache.invalidate_from(effective_from_stage, self.stage_names)

        # Handle global force rerun
        if self.config.force_rerun:
            self.context.info("Force rerun requested: invalidating all cached stages.")
            for s_name in self.stage_names:
                self.context.cache.invalidate(s_name)

        current_output: Any = initial_input
        executed_stages: List[str] = []
        cached_stages: List[str] = []
        stage_outputs: Dict[str, Any] = {}

        for idx, stage in enumerate(self.stages):
            s_name = stage.name
            stage_start_t = time.perf_counter()

            # Compute fingerprint
            fp = stage.compute_fingerprint(current_output, self.config, self.context)
            can_use_cache = self.config.cache_enabled and self.context.cache.is_valid(s_name, fp)

            # A cached product object is insufficient if its PNG/CSV/GIF
            # side-effect files have since been removed.
            if can_use_cache and callable(getattr(stage, "outputs_exist", None)):
                cached_output = self.context.cache.load_output(s_name)
                if not stage.outputs_exist(cached_output):
                    self.context.info(f"Stage {s_name}: missing output artifacts, regenerating.")
                    self.context.cache.invalidate_from(s_name, self.stage_names)
                    can_use_cache = False

            if can_use_cache:
                self.context.info(f"[Stage {idx+1}/{len(self.stages)}] {s_name}: CACHE HIT (loading cached output)")
                current_output = self.context.cache.load_output(s_name)
                cached_stages.append(s_name)
            else:
                self.context.info(f"[Stage {idx+1}/{len(self.stages)}] {s_name}: EXECUTING...")
                current_output = stage.run(current_output, self.config, self.context)
                if self.config.cache_enabled:
                    self.context.cache.save_output(s_name, fp, current_output, version=stage.version)
                    if idx + 1 < len(self.stage_names):
                        self.context.cache.invalidate_from(self.stage_names[idx + 1], self.stage_names)
                executed_stages.append(s_name)

            stage_outputs[s_name] = current_output
            elapsed = time.perf_counter() - stage_start_t
            self.context.info(f"[Stage {idx+1}/{len(self.stages)}] {s_name} completed in {elapsed:.2f} s")

        total_elapsed = time.perf_counter() - t_start
        self.context.info("=" * 80)
        self.context.info(f"Pipeline Run Completed Successfully in {total_elapsed:.2f} s.")
        self.context.info(f"Executed stages: {executed_stages} | Cached stages: {cached_stages}")
        self.context.info("=" * 80)

        # Save run manifest with complete reproducibility & provenance metadata
        canonical_track_method = resolve_tracking_method_name(self.config.tracking.method)
        effective_track_params = resolve_and_validate_tracking_parameters(canonical_track_method, self.config.tracking)
        track_stage_obj = self.stages[0]
        assoc_stage_obj = self.stages[2] if len(self.stages) > 2 else None
        track_out_obj = stage_outputs.get("tracking")

        manifest_data = {
            "run_name": self.config.run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_runtime_sec": round(total_elapsed, 2),
            "executed_stages": executed_stages,
            "cached_stages": cached_stages,
            "provenance": {
                "tracking_algorithm": canonical_track_method,
                "tracking_stage_class": f"{track_stage_obj.__class__.__name__} v{track_stage_obj.version}",
                "effective_tracking_parameters": effective_track_params,
                "association_algorithm": str(self.config.association.method).lower(),
                "association_stage_class": (
                    f"{assoc_stage_obj.__class__.__name__} v{assoc_stage_obj.version}"
                    if assoc_stage_obj is not None
                    else None
                ),
                "code_revision": _get_code_revision(self.context.repo_root),
                "input_dataset": getattr(track_out_obj, "dataset_metadata", None) if track_out_obj is not None else None,
            },
            "config": self.config.to_dict(),
        }
        save_json(manifest_data, self.context.run_dir / "manifest.json")

        # Generate run summary report
        self._generate_run_report(total_elapsed, executed_stages, cached_stages, current_output)

        return {
            "status": "SUCCESS",
            "total_elapsed_sec": total_elapsed,
            "executed_stages": executed_stages,
            "cached_stages": cached_stages,
            "stage_outputs": stage_outputs,
            "final_output": current_output,
        }

    def status(self) -> Dict[str, Any]:
        """Returns the cache and execution status of the pipeline stages."""
        cache_stat = self.context.cache.get_status(self.stage_names)
        self.context.info(f"Pipeline Status for '{self.config.run_name}':")
        self.context.info("-" * 60)
        self.context.info(f"{'Stage Name':<20} {'Status':<15} {'Fingerprint':<12} {'Cached At'}")
        self.context.info("-" * 60)
        for s_name in self.stage_names:
            info = cache_stat[s_name]
            self.context.info(
                f"{s_name:<20} {info['status']:<15} {info['fingerprint']:<12} {info['timestamp']}"
            )
        self.context.info("-" * 60)
        return cache_stat

    def _generate_run_report(
        self,
        runtime_sec: float,
        executed: List[str],
        cached: List[str],
        final_output: Any,
    ) -> None:
        """Generates report.md in run directory."""
        report_file = self.context.run_dir / "report.md"
        content = [
            f"# Pipeline Run Report: `{self.config.run_name}`\n",
            f"- **Execution Time (UTC)**: {datetime.now(timezone.utc).isoformat()}",
            f"- **Total Runtime**: {runtime_sec:.2f} s",
            f"- **Executed Stages**: {', '.join(executed) if executed else 'None (All Cached)'}",
            f"- **Cached Stages**: {', '.join(cached) if cached else 'None'}\n",
            "## Stage Summary\n",
        ]

        if hasattr(final_output, "comparison_metrics_df"):
            val_out = final_output
            seg_out = val_out.segmentation_output
            size_out = seg_out.size_gate_output
            track_out = size_out.birth_gate_output.association_output.tracking_output

            # Format dataframe as markdown table without requiring tabulate
            df_comp = val_out.comparison_metrics_df
            headers = list(df_comp.columns)
            header_line = "| " + " | ".join(headers) + " |"
            sep_line = "| " + " | ".join(["---" for _ in headers]) + " |"
            data_lines = [
                "| " + " | ".join(str(val) for val in row) + " |"
                for row in df_comp.itertuples(index=False)
            ]
            table_md = "\n".join([header_line, sep_line] + data_lines)

            content.extend([
                "### Key Quantitative Metrics\n",
                f"- **Observation Pairs**: {len(track_out.manifest_df)}",
                f"- **Total Gate OFF Tracks**: {track_out.total_tracks}",
                f"- **Candidate Families**: {size_out.birth_gate_output.association_output.candidate_family_count}",
                f"- **Admitted Families**: {len(size_out.admitted_family_ids)}",
                f"- **Admitted Member Tracks**: {len(size_out.admitted_track_ids)}",
                f"- **Total Recovered Patches**: {size_out.total_recovered_patches:,}",
                f"- **Total Segmented Pixels**: {seg_out.total_segmented_pixels:,}\n",
                "### External Catalogue Validation\n",
                table_md,
                "\n",
            ])

        report_file.write_text("\n".join(content), encoding="utf-8")
