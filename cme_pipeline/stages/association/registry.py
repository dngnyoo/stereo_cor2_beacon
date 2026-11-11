"""Association algorithm registry with legacy name compatibility."""

from dataclasses import dataclass
from typing import Type

from cme_pipeline.core.stage import PipelineStage
from cme_pipeline.stages.association.v01_identity_association import IdentityAssociationStage
from cme_pipeline.stages.association.v02_pure_do_rule_b import PureDOFamilyAssociationStage
from cme_pipeline.stages.association.v03_adaptive_roi_association import AdaptiveROIAssociationStage
from cme_pipeline.stages.association.v04_birth_anchored_association import BirthAnchoredAssociationStage


@dataclass(frozen=True)
class AssociationMethodSpec:
    canonical_name: str
    stage_cls: Type[PipelineStage]
    aliases: tuple[str, ...] = ()
    description: str = ""


_METHODS = (
    AssociationMethodSpec(
        "identity",
        IdentityAssociationStage,
        ("1to1", "v01_identity_association"),
        "No family merging; preserves individual tracks.",
    ),
    AssociationMethodSpec(
        "pure_do",
        PureDOFamilyAssociationStage,
        ("rule_b", "v02_pure_do_rule_b"),
        "PureDO family association, Rule B.",
    ),
    AssociationMethodSpec(
        "adaptive_roi",
        AdaptiveROIAssociationStage,
        ("adaptive_pa_roi", "v03_adaptive_roi_association"),
        "Adaptive ROI association.",
    ),
    AssociationMethodSpec(
        "birth_anchored",
        BirthAnchoredAssociationStage,
        ("directional", "v04_birth_anchored_association"),
        "Birth-anchored directional association.",
    ),
)

_REGISTRY = {spec.canonical_name: spec for spec in _METHODS}
_ALIASES = {}

for spec in _METHODS:
    for name in (spec.canonical_name, *spec.aliases):
        key = name.lower()
        if key in _ALIASES:
            raise ValueError(f"Duplicate association name: {key}")
        _ALIASES[key] = spec.canonical_name


def resolve_association_method_name(method: str) -> str:
    key = str(method).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown association method '{method}'. "
            f"Registered methods: {sorted(_ALIASES)}"
        )
    return _ALIASES[key]


def get_association_spec(method: str) -> AssociationMethodSpec:
    return _REGISTRY[resolve_association_method_name(method)]


def get_association_stage(method: str) -> PipelineStage:
    return get_association_spec(method).stage_cls()


def list_association_methods() -> list[str]:
    return sorted(_REGISTRY)
