"""Validated configuration reader for the standard nine-stage pipeline.

This schema is intentionally distinct from legacy PipelineConfig. No implicit
cross-run inputs or unsupported scientific methods may be inferred.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cme_pipeline.core.standard_stages import STAGE_BY_NAME, resolve_visualization_mode, normalize_visualization_mode
from cme_pipeline.utils.io import load_yaml

KNOWN_TOP = frozenset({"experiment", "input", "visualization", *STAGE_BY_NAME})
DEFAULT_METHODS = {
    "preprocessing": "dual_archive_rd",
    "detection": "d4",
    "tracking": "a5b",
    "association": "identity",
    "gates": "consecutive_size_qualified",
    "recovery": "signed_dynamic_pa",
    "segmentation": "island_first",
    "catalog": "event_kinematics",
    "validation": "external_catalogs",
}


@dataclass(frozen=True)
class StandardConfig:
    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "StandardConfig":
        src = Path(path).resolve()
        data = load_yaml(src)
        if not isinstance(data, dict):
            raise ValueError("Expected YAML mapping at document root")
        unknown = set(data) - KNOWN_TOP
        if unknown:
            raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
        exp = data.get("experiment", {})
        if not isinstance(exp, dict):
            raise ValueError("experiment must be a mapping")
        date = str(exp.get("date", ""))
        if len(date) != 8 or not date.isdigit():
            raise ValueError("experiment.date must be YYYYMMDD (quote it in YAML)")
        for name in STAGE_BY_NAME:
            if name in data and not isinstance(data[name], dict):
                raise ValueError(f"{name} must be a mapping")
            if (data.get(name) or {}).get("enabled") is False:
                raise ValueError(f"Cannot disable mandatory stage {name}; use --until instead")
            resolve_visualization_mode(name, config=data)
        viz = data.get("visualization") or {}
        if not isinstance(viz, dict):
            raise ValueError("visualization must be a mapping")
        if set(viz) - {"default"}:
            raise ValueError("visualization only accepts the default key; stage overrides go under that stage")
        normalize_visualization_mode(viz.get("default", "off"))
        return cls(path=src, data=data)

    def method(self, stage: str) -> str:
        if stage not in STAGE_BY_NAME:
            raise ValueError(f"Unknown stage: {stage}")
        section = self.data.get(stage) or {}
        return str(section.get("method", DEFAULT_METHODS[stage]))

    def vis_mode(self, stage: str, cli_override: str | None = None) -> str:
        return resolve_visualization_mode(stage, config=self.data, cli_override=cli_override)
