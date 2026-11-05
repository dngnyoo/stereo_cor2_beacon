"""Pipeline Orchestrator for CME Pipeline."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import PipelineStage
from cme_pipeline.stages.association.birth_anchored_association import BirthAnchoredAssociationStage
from cme_pipeline.stages.association.pure_do_association import PureDOFamilyAssociationStage
from cme_pipeline.stages.gates.birth_radius_gate import FamilyBirthRadiusGateStage
from cme_pipeline.stages.gates.size_gate import FamilySizeGateStage
from cme_pipeline.stages.recovery.signed_pa_recovery import SignedPARecoveryStage
from cme_pipeline.stages.segmentation.island_first_segmentation import IslandFirstSegmentationStage
from cme_pipeline.stages.tracking.birth_anchored_tracking import BirthAnchoredTrackingStage
from cme_pipeline.stages.tracking.e1_tracking import E1TrackingStage
from cme_pipeline.stages.validation.external_validation import ExternalValidationStage
from cme_pipeline.utils.io import save_json


def resolve_stages(config: PipelineConfig) -> List[PipelineStage]:
    """Resolves and instantiates the canonical 7 stages based on configuration."""
    # 1. Tracking
    if config.tracking.method in ("birth_anchored", "directional"):
        track_stage = BirthAnchoredTrackingStage()
    else:
        track_stage = E1TrackingStage()

    # 2. Recovery
    rec_stage = SignedPARecoveryStage()

    # 3. Association
    if config.association.method in ("birth_anchored", "directional"):
        assoc_stage = BirthAnchoredAssociationStage()
    else:
        assoc_stage = PureDOFamilyAssociationStage()

    # 4. Gates
    birth_stage = FamilyBirthRadiusGateStage()
    size_stage = FamilySizeGateStage()

    # 5. Segmentation
    seg_stage = IslandFirstSegmentationStage()

    # 6. Validation
    val_stage = ExternalValidationStage()

    return [
        track_stage,
        rec_stage,
        assoc_stage,
        birth_stage,
        size_stage,
        seg_stage,
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

    def run(self, from_stage: Optional[str] = None) -> Dict[str, Any]:
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

        current_output: Any = None
        executed_stages: List[str] = []
        cached_stages: List[str] = []

        for idx, stage in enumerate(self.stages):
            s_name = stage.name
            stage_start_t = time.perf_counter()

            # Compute fingerprint
            fp = stage.compute_fingerprint(current_output, self.config, self.context)
            can_use_cache = self.config.cache_enabled and self.context.cache.is_valid(s_name, fp)

            if can_use_cache:
                self.context.info(f"[Stage {idx+1}/{len(self.stages)}] {s_name}: CACHE HIT (loading cached output)")
                current_output = self.context.cache.load_output(s_name)
                cached_stages.append(s_name)
            else:
                self.context.info(f"[Stage {idx+1}/{len(self.stages)}] {s_name}: EXECUTING...")
                current_output = stage.run(current_output, self.config, self.context)
                if self.config.cache_enabled:
                    self.context.cache.save_output(s_name, fp, current_output, version=stage.version)
                executed_stages.append(s_name)

            elapsed = time.perf_counter() - stage_start_t
            self.context.info(f"[Stage {idx+1}/{len(self.stages)}] {s_name} completed in {elapsed:.2f} s")

        total_elapsed = time.perf_counter() - t_start
        self.context.info("=" * 80)
        self.context.info(f"Pipeline Run Completed Successfully in {total_elapsed:.2f} s.")
        self.context.info(f"Executed stages: {executed_stages} | Cached stages: {cached_stages}")
        self.context.info("=" * 80)

        # Save run manifest
        manifest_data = {
            "run_name": self.config.run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_runtime_sec": round(total_elapsed, 2),
            "executed_stages": executed_stages,
            "cached_stages": cached_stages,
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
