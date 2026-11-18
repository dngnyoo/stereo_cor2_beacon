"""Inspect standardized stage contracts and current registered tracking variants.

Safe read-only command: python -m cme_pipeline.stage_info [--stage tracking] [--methods]
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from cme_pipeline.core.standard_stages import STAGES, STAGE_BY_NAME


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=tuple(STAGE_BY_NAME))
    p.add_argument("--methods", action="store_true")
    args = p.parse_args()
    stages = [STAGE_BY_NAME[args.stage]] if args.stage else STAGES
    for s in stages:
        print(json.dumps({
            "stage": s.name.value, "order": s.ordinal, "folder": s.folder,
            "inputs": s.requires, "outputs": s.produces, "substeps": s.substeps,
        }, indent=2))
    if args.methods:
        if args.stage and args.stage != "tracking":
            print("Method introspection is currently implemented only for tracking.")
            return
        from cme_pipeline.stages.tracking.registry import _ensure_defaults_registered, _REGISTRY
        _ensure_defaults_registered()
        print("Registered tracking methods:")
        for name, spec in sorted(_REGISTRY.items()):
            print(json.dumps({"method": name, "description": spec.description,
                              "parameters": sorted(spec.supported_parameters),
                              "defaults": spec.default_parameters}, default=str))


if __name__ == "__main__":
    main()
