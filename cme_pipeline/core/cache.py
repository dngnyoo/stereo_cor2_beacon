"""Stage-level local file cache for CME Pipeline."""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cme_pipeline.utils.io import load_json, save_json


class StageCache:
    """Manages stage-level caching with fingerprint-based validity and invalidation."""

    def __init__(self, cache_dir: Path, logger: Optional[logging.Logger] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.cache_dir / "cache_manifest.json"
        self.logger = logger or logging.getLogger("cme_pipeline.cache")
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_file.is_file():
            try:
                return load_json(self.manifest_file)
            except Exception as e:
                self.logger.warning(f"Could not load cache manifest ({e}); resetting.")
        return {}

    def _save_manifest(self) -> None:
        save_json(self._manifest, self.manifest_file)

    def is_valid(self, stage_name: str, expected_fingerprint: str) -> bool:
        """Checks if a valid cached output exists matching the expected fingerprint."""
        if stage_name not in self._manifest:
            return False
        meta = self._manifest[stage_name]
        if meta.get("status") != "VALID":
            return False
        if meta.get("fingerprint") != expected_fingerprint:
            return False
        data_file = self.cache_dir / meta.get("data_file", "")
        return data_file.is_file()

    def load_output(self, stage_name: str) -> Any:
        """Loads cached stage output object."""
        if stage_name not in self._manifest:
            raise KeyError(f"No cache entry for stage '{stage_name}'")
        meta = self._manifest[stage_name]
        data_file = self.cache_dir / meta["data_file"]
        with open(data_file, "rb") as f:
            return pickle.load(f)

    def save_output(self, stage_name: str, fingerprint: str, output: Any, version: str = "1.0.0") -> None:
        """Saves stage output and updates manifest."""
        data_filename = f"{stage_name}.pkl"
        data_path = self.cache_dir / data_filename
        with open(data_path, "wb") as f:
            pickle.dump(output, f, protocol=pickle.HIGHELOG_PROTOCOL if hasattr(pickle, "HIGHELOG_PROTOCOL") else pickle.HIGHEST_PROTOCOL)

        self._manifest[stage_name] = {
            "version": version,
            "fingerprint": fingerprint,
            "status": "VALID",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_file": data_filename,
        }
        self._save_manifest()
        self.logger.info(f"Cached stage '{stage_name}' (fingerprint: {fingerprint[:8]}...)")

    def invalidate(self, stage_name: str) -> None:
        """Invalidates cache for a specific stage."""
        if stage_name in self._manifest:
            self._manifest[stage_name]["status"] = "INVALIDATED"
            data_file = self.cache_dir / self._manifest[stage_name].get("data_file", "")
            if data_file.is_file():
                try:
                    data_file.unlink()
                except Exception:
                    pass
            self._save_manifest()
            self.logger.info(f"Invalidated cache for stage '{stage_name}'")

    def invalidate_from(self, stage_name: str, stage_order: List[str]) -> None:
        """Invalidates all stages from stage_name onwards in stage_order."""
        if stage_name not in stage_order:
            raise ValueError(f"Stage '{stage_name}' not found in pipeline stage order: {stage_order}")
        start_idx = stage_order.index(stage_name)
        for name in stage_order[start_idx:]:
            self.invalidate(name)

    def get_status(self, stage_order: List[str]) -> Dict[str, Any]:
        """Returns cache status summary for all stages in order."""
        summary = {}
        for name in stage_order:
            if name in self._manifest:
                meta = self._manifest[name]
                summary[name] = {
                    "status": meta.get("status", "UNKNOWN"),
                    "fingerprint": meta.get("fingerprint", "")[:8],
                    "timestamp": meta.get("timestamp", ""),
                }
            else:
                summary[name] = {"status": "NOT_CACHED", "fingerprint": "-", "timestamp": "-"}
        return summary
