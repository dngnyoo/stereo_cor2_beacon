"""I/O and fingerprinting utilities for CME Pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml


def compute_file_hash(path: Union[str, Path], block_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def compute_dict_hash(d: Dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 hash of a JSON-serializable dictionary."""
    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.ndarray, list)):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    encoded = json.dumps(d, sort_keys=True, default=_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_array_fingerprint(arr: np.ndarray) -> str:
    """Computes a lightweight deterministic fingerprint for a numpy array."""
    hasher = hashlib.sha256()
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    # Hash a subsample or full buffer if reasonably sized
    if arr.nbytes <= 10_000_000:
        hasher.update(arr.tobytes())
    else:
        # Sample slices from start, middle, and end
        flat = arr.ravel()
        sample = np.concatenate([flat[:1000], flat[len(flat)//2:len(flat)//2 + 1000], flat[-1000:]])
        hasher.update(sample.tobytes())
        hasher.update(str(float(np.nansum(flat))).encode("utf-8"))
    return hasher.hexdigest()


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Saves a dictionary to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data: Dict[str, Any], path: Union[str, Path], indent: int = 2) -> None:
    """Saves a dictionary to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a JSON file."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

