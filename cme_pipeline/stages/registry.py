"""Registry for pipeline stages with a single current implementation."""

from cme_pipeline.stages.recovery.v01_signed_pa_recovery import SignedPARecoveryStage
from cme_pipeline.stages.gates.v01_birth_radius_gate import FamilyBirthRadiusGateStage
from cme_pipeline.stages.gates.v01_family_size_gate import FamilySizeGateStage
from cme_pipeline.stages.segmentation.v01_island_first_segmentation import IslandFirstSegmentationStage
from cme_pipeline.stages.validation.v01_external_catalog_validation import ExternalValidationStage


STAGE_REGISTRY = {
    "recovery": {
        "v01_signed_pa_recovery": SignedPARecoveryStage,
    },
    "birth_gate": {
        "v01_birth_radius_gate": FamilyBirthRadiusGateStage,
    },
    "size_gate": {
        "v01_family_size_gate": FamilySizeGateStage,
    },
    "segmentation": {
        "v01_island_first_segmentation": IslandFirstSegmentationStage,
    },
    "validation": {
        "v01_external_catalog_validation": ExternalValidationStage,
    },
}

ALIASES = {
    "recovery": {
        "signed_dynamic_pa": "v01_signed_pa_recovery",
    },
    "birth_gate": {},
    "size_gate": {},
    "segmentation": {
        "island_first": "v01_island_first_segmentation",
    },
    "validation": {},
}


def resolve_stage(stage_name: str, algorithm: str):
    """Instantiate a registered algorithm, accepting legacy aliases."""
    methods = STAGE_REGISTRY[stage_name]
    canonical = ALIASES[stage_name].get(algorithm, algorithm)

    if canonical not in methods:
        raise ValueError(
            f"Unknown {stage_name} algorithm '{algorithm}'. "
            f"Available: {sorted(methods)}"
        )

    return methods[canonical]()
