"""Non-destructive run directory metadata for the nine-stage architecture."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from cme_pipeline.core.standard_stages import STAGE_BY_NAME


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_run(root: str | Path, config: Mapping[str, Any], revision: str, provenance: Mapping[str, Any]) -> Path:
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(f"Run already exists: {root}")
    root.mkdir(parents=True)
    _write_json(root / "config_snapshot.json", dict(config))
    _write_json(root / "run_manifest.json", {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision, "input_provenance": dict(provenance),
        "stages": {name: "NOT_RUN" for name in STAGE_BY_NAME},
    })
    return root


def complete_stage(root: str | Path, stage: str, inputs: Mapping[str, str],
                   outputs: Mapping[str, str], implementation: str, parameters: Mapping[str, Any]) -> None:
    root = Path(root).resolve()
    spec = STAGE_BY_NAME[stage]
    missing = set(spec.produces) - set(outputs)
    if missing:
        raise ValueError(f"Missing {stage} outputs: {sorted(missing)}")
    output_info = {}
    for name, relative_path in outputs.items():
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Output must be a file inside run: {path}")
        output_info[name] = {"path": str(path.relative_to(root)), "sha256": _hash_file(path)}
    _write_json(root / spec.folder / "metadata.json", {
        "stage": stage, "implementation": implementation, "parameters": dict(parameters),
        "inputs": dict(inputs), "outputs": output_info, "status": "COMPLETED",
    })
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["stages"][stage] = "COMPLETED"
    _write_json(path, manifest)


def verify_stage(root: str | Path, stage: str) -> dict[str, Any]:
    root = Path(root).resolve()
    spec = STAGE_BY_NAME[stage]
    metadata = json.loads((root / spec.folder / "metadata.json").read_text())
    for name, item in metadata["outputs"].items():
        file = (root / item["path"]).resolve()
        if not file.is_relative_to(root) or not file.is_file() or _hash_file(file) != item["sha256"]:
            raise ValueError(f"Corrupt or missing artifact: {stage}/{name}")
    return metadata
