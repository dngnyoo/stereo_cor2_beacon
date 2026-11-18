"""Read-only preflight for the new nine-stage architecture.

Use as: python -m cme_pipeline.standard_plan --config FILE --until gates
The execution adapter is not yet wired; this command never creates a run.
"""
from __future__ import annotations

import argparse
import json
from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.core.standard_stages import VIS_MODES, stages_until


def build_plan(config: StandardConfig, until: str, vis: str | None = None) -> dict:
    return {
        "config": str(config.path),
        "date": str(config.data["experiment"]["date"]),
        "until": until,
        "creates_run": False,
        "automatic_cross_run_reuse": False,
        "stages": [
            {
                "name": s.name.value,
                "folder": s.folder,
                "method": config.method(s.name.value),
                "visualization": config.vis_mode(s.name.value, vis),
                "required_inputs": list(s.requires),
                "required_outputs": list(s.produces),
            } for s in stages_until(until)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--visualize", choices=sorted(VIS_MODES), default=None)
    args = parser.parse_args()
    config = StandardConfig.load(args.config)
    print(json.dumps(build_plan(config, args.until, args.visualize), indent=2))


if __name__ == "__main__":
    main()
