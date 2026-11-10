"""Execution context for CME Pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

from cme_pipeline.core.config import PipelineConfig


class PipelineContext:
    """Manages filesystem layout, caching, logging, and shared execution state."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.repo_root = Path(config.repo_root).resolve()
        
        # Setup run directories
        self.run_dir = (self.repo_root / config.output_base_dir / config.run_name).resolve()
        self.stages_dir = self.run_dir / "stages"
        self.tables_dir = self.run_dir / "tables"
        self.arrays_dir = self.run_dir / "arrays"
        self.visualizations_dir = self.run_dir / "visualizations"
        self.cache_dir = self.run_dir / ".cache"

        for d in [self.stages_dir, self.tables_dir, self.arrays_dir, self.visualizations_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = self._setup_logger()

        # Cache manager initialized lazily/externally
        self._cache = None

    @property
    def cache(self) -> Any:
        if self._cache is None:
            from cme_pipeline.core.cache import StageCache
            self._cache = StageCache(self.cache_dir, self.logger)
        return self._cache

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"cme_pipeline.{self.config.run_name}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            fmt = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

            # Console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
            logger.addHandler(ch)

            # File handler
            log_file = self.run_dir / "pipeline.log"
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        return logger

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

