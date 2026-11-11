"""Family-level Birth Radius Gate Stage (max_birth_radius_px <= 30.0)."""

from __future__ import annotations

from typing import Any, Dict, List

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.context import PipelineContext
from cme_pipeline.core.stage import AssociationOutput, BirthGateOutput, PipelineStage
from cme_pipeline.utils.io import compute_dict_hash


class FamilyBirthRadiusGateStage(PipelineStage):
    """Evaluates Family-level Solar Root Birth Radius Gate."""
    name: str = "birth_gate"
    version: str = "1.0.0"

    def compute_fingerprint(
        self, inputs: AssociationOutput, config: PipelineConfig, context: PipelineContext
    ) -> str:
        desc = {
            "stage": self.name,
            "version": self.version,
            "max_birth_radius_px": config.birth_gate.max_birth_radius_px,
            "upstream_candidate_families": inputs.candidate_family_count,
        }
        return compute_dict_hash(desc)

    def run(
        self, inputs: AssociationOutput, config: PipelineConfig, context: PipelineContext
    ) -> BirthGateOutput:
        context.info("Running FamilyBirthRadiusGateStage...")

        r_max_gate = config.birth_gate.max_birth_radius_px
        birth_gated_fams: List[str] = []
        rejected_reasons: Dict[str, str] = {}

        for fid, members in inputs.family_members.items():
            if fid.startswith("UNRESOLVED"):
                rejected_reasons[fid] = "UNRESOLVED_COMPETING"
                continue

            anchors = inputs.family_anchors.get(fid, [])
            if len(anchors) > 0:
                birth_gated_fams.append(fid)
            else:
                rejected_reasons[fid] = "NO_SOLAR_ROOT_ANCHOR"

        birth_gated_fams = sorted(birth_gated_fams)
        context.info(
            f"Birth Radius Gate ({r_max_gate:.1f} px): {len(birth_gated_fams)} passed, "
            f"{len(rejected_reasons)} rejected."
        )

        return BirthGateOutput(
            association_output=inputs,
            birth_gated_family_ids=birth_gated_fams,
            rejected_family_reasons=rejected_reasons,
            max_birth_radius_px=r_max_gate,
        )

