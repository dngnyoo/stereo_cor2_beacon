"""Read-only inventory of existing pipeline Python files and imports.

python -m cme_pipeline.script_inventory --output docs/SCRIPT_INVENTORY.json
Does not move/delete files. Relative references are reported for migration review.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def build_inventory(root: Path) -> list[dict]:
    package = root / "cme_pipeline"
    records = []
    for path in sorted(package.rglob("*.py")):
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    prefix = "." * node.level + (node.module or "")
                    imported.append(prefix)
            error = None
        except SyntaxError as exc:
            imported, error = [], str(exc)
        location = path.relative_to(package)
        if location.parts[0] == "stages":
            suggested = "existing_stage"
        elif location.parts[0] in ("core", "utils"):
            suggested = "shared_infrastructure"
        elif "visual" in path.stem or "diagnostic" in path.stem:
            suggested = "diagnostics"
        elif "experiment" in path.stem or "comparison" in path.stem:
            suggested = "experiments_or_runner"
        else:
            suggested = "manual_classification_required"
        records.append({
            "path": str(path.relative_to(root)),
            "module": module, "lines": source.count("\n") + 1,
            "imports": sorted(set(imported)),
            "suggested_group": suggested, "parse_error": error,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    data = build_inventory(root)
    payload = {
        "root": str(root), "total_scripts": len(data),
        "note": "Suggestions are preliminary; review import dependencies before moving.",
        "files": data,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"Inventory: {len(data)} scripts -> {target}")
    else:
        print(text)


if __name__ == "__main__":
    main()
