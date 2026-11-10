"""Programmatic experiment runner for CME Pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from cme_pipeline.core.config import PipelineConfig
from cme_pipeline.core.pipeline import Pipeline


def run_experiment(
    config: Union[str, Path, PipelineConfig],
    from_stage: Optional[str] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Runs a CME pipeline experiment programmatically.
    
    Args:
        config: Path to YAML config or PipelineConfig instance.
        from_stage: Optional stage to resume from.
        force_rerun: Whether to invalidate cache and re-run all stages.
        
    Returns:
        Dictionary containing pipeline execution results.
    """
    if isinstance(config, (str, Path)):
        cfg = PipelineConfig.from_yaml(config)
    else:
        cfg = config

    if force_rerun:
        cfg.force_rerun = True

    pipeline = Pipeline(cfg)
    return pipeline.run(from_stage=from_stage)

