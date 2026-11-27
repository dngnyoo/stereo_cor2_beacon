"""Repeat verified Stage 09 evaluation with per-date pinned references."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from cme_pipeline.core.run_layout import verify_stage, _hash_file
from cme_pipeline.core.standard_config import StandardConfig
from cme_pipeline.external_reference_acquisition import prepare
from cme_pipeline.standard_external_validation_parent import (
    NAMES, KEYS, parameters, references, execute_external, MATCH_COLUMNS,
)
from cme_pipeline.standard_seeds_validation_parent import resolve_catalog_parent

DEFAULTS = dict(max_time_delta_sec=900, min_mean_iou=0.1,
                min_common_frames=1, cactus_min_iou=0,
                mvcc_max_time_delta_sec=60, mvcc_min_mean_iou=0.1)
SUMMARY_COLUMNS = ["algorithm", "date", "catalog", "status", "matched_events",
                   "mean_matched_angular_iou", "metric", "reason", "input_run",
                   "validation_run", "observation_sha256"]
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, columns, rows):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_plan(path):
    path = Path(path).resolve()
    plan = yaml.safe_load(path.read_text())
    if not isinstance(plan, dict) or set(plan) - {
        "catalogs", "reference_cache", "thresholds", "runs"
    }:
        raise ValueError("Invalid batch plan")
    catalogs = plan.get("catalogs", list(NAMES))
    if (not isinstance(catalogs, list) or not catalogs
            or any(x not in NAMES for x in catalogs)
            or len(set(catalogs)) != len(catalogs)):
        raise ValueError("Specify unique supported catalogs")
    thresholds = plan.get("thresholds", {})
    if not isinstance(thresholds, dict) or set(thresholds) - set(DEFAULTS):
        raise ValueError("Unknown batch threshold")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           for v in thresholds.values()):
        raise ValueError("Thresholds must be numeric")
    settings = dict(DEFAULTS, **thresholds)
    settings.update(method="external_catalogs", catalogs=catalogs,
                    visualization={"mode": "off"})
    cache = plan.get("reference_cache", "data/catalogs")
    if not isinstance(cache, str) or not cache.strip():
        raise ValueError("Invalid reference_cache")
    cache = str(Path(cache).expanduser().resolve())
    rows = plan.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Specify at least one input Run")
    jobs, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"algorithm", "date", "input_run"}:
            raise ValueError("Each Run needs algorithm, date, input_run")
        algorithm, date, parent = row["algorithm"], str(row["date"]), row["input_run"]
        if not isinstance(algorithm, str) or not SAFE.fullmatch(algorithm):
            raise ValueError("Unsafe algorithm label")
        if not re.fullmatch(r"\d{8}", date):
            raise ValueError("Expected YYYYMMDD")
        datetime.strptime(date, "%Y%m%d")
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError("Invalid input_run")
        key = (algorithm, date)
        if key in seen:
            raise ValueError("Duplicate algorithm/date")
        seen.add(key)
        parent = Path(parent).expanduser().resolve()
        # Parent snapshot is the authority for all upstream settings.
        snapshot = parent / "config_snapshot.json"
        cfg = json.loads(snapshot.read_text())
        if str(cfg["experiment"]["date"]) != date:
            raise ValueError(f"Input date mismatch: {parent}")
        cfg["validation"] = dict(settings)
        config = StandardConfig(path=snapshot, data=cfg)
        parameters(config)
        jobs.append((algorithm, date, parent, config))
    return settings, cache, jobs


def pin_references(date, settings, cache, output):
    """Acquire once per date/catalog and copy immutable bytes into batch."""
    frozen = {}
    auto = dict(settings, acquisition={"mode": "auto", "reuse_existing": True,
                                      "cache_root": cache})
    for name in settings["catalogs"]:
        try:
            effective, acquisition = prepare(name, date, auto)
            paths, reason = references(effective, name, date)
            state = acquisition["status"]
            if state in ("SOURCE_UNAVAILABLE", "ACQUISITION_FAILED"):
                paths = None
                reason = acquisition["reason"]
            if paths is None:
                frozen[name] = dict(
                    status="ACQUISITION_FAILED" if state == "ACQUISITION_FAILED"
                    else "REFERENCE_UNAVAILABLE", reason=reason, acquisition=acquisition)
                continue
            folder = output / "references" / date / name.lower()
            folder.mkdir(parents=True, exist_ok=False)
            pinned = {}
            for key, source in paths.items():
                before = _hash_file(source)
                target = folder / (key + ".csv")
                shutil.copyfile(source, target)
                if _hash_file(target) != before or _hash_file(source) != before:
                    raise ValueError("Reference changed during snapshot")
                pinned[key] = {"path": str(target), "sha256": before,
                               "source_path": str(source)}
            frozen[name] = dict(status="PINNED", files=pinned, acquisition=acquisition)
        except Exception as exc:
            # Preserve the failure; never retry differently for another algorithm.
            frozen[name] = dict(status="REFERENCE_ERROR",
                                reason=f"{type(exc).__name__}: {exc}")
    return frozen


def verify_pins(frozen):
    for entry in frozen.values():
        for item in entry.get("files", {}).values():
            if _hash_file(Path(item["path"])) != item["sha256"]:
                raise ValueError("Pinned reference checksum mismatch")


def plot_results(output, rows, catalogs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(dict.fromkeys((r["date"], r["algorithm"]) for r in rows))
    fig, axes = plt.subplots(len(catalogs), 1, squeeze=False,
                             figsize=(max(7, min(20, len(labels)*1.2)), 3*len(catalogs)))
    for ax, name in zip(axes[:, 0], catalogs):
        lookup = {(r["date"], r["algorithm"]): r for r in rows if r["catalog"] == name}
        for i, label in enumerate(labels):
            row = lookup[label]
            value = row["mean_matched_angular_iou"]
            if value is not None and row["status"] == "COMPLETED":
                ax.bar(i, value)
            else:
                ax.text(i, .03, "N/A", ha="center", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(name)
        ax.set_ylabel("Matched angular IoU")
        ax.set_xticks(range(len(labels)), [f"{d}\n{a}" for d, a in labels],
                      rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=140)
    plt.close(fig)


def run_batch(plan, output):
    settings, cache, jobs = load_plan(plan)
    output = Path(output).expanduser().resolve()
    if any(output.is_relative_to(parent) for _, _, parent, _ in jobs):
        raise ValueError("Batch output cannot be inside an input Run")
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    # Check actual parent provenance before fetching references.
    valid, failures, observations = {}, {}, {}
    for algorithm, date, parent, config in jobs:
        key = algorithm, date
        try:
            *_, obs = resolve_catalog_parent(config, parent, include_observations=True)
            # Full observation manifest fingerprint; equal hashes are not proof
            # of identical image arrays or identical scientific parameters.
            observations[key] = hashlib.sha256(
                obs.to_csv(index=False).encode("utf-8")).hexdigest()
            valid[key] = True
        except Exception as exc:
            failures[key] = f"{type(exc).__name__}: {exc}"
    refs = {date: pin_references(date, settings, cache, output)
            for date in sorted({d for a, d in valid})}
    write_json(output / "reference_manifest.json", refs)
    rows, matches, run_records = [], [], []
    for algorithm, date, parent, config in jobs:
        key = algorithm, date
        child = output / "runs" / f"{algorithm}_{date}"
        frozen = refs.get(date, {})
        reports = {}
        reason = failures.get(key)
        run_status = "INPUT_FAILED" if reason else "NOT_RUN"
        if not reason:
            try:
                verify_pins(frozen)
                chosen = [n for n in settings["catalogs"] if frozen[n]["status"] == "PINNED"]
                if chosen:
                    explicit = dict(settings, catalogs=chosen)
                    for name in chosen:
                        explicit.update({k: v["path"] for k, v in frozen[name]["files"].items()})
                    cfg = dict(config.data, validation=explicit)
                    execute_external(StandardConfig(config.path, cfg), input_run=parent,
                                     until="validation", run_dir=child, visualize="off")
                    meta = verify_stage(child, "validation")
                    summary = json.loads((child / meta["outputs"]["evaluation_metrics"]["path"]).read_text())
                    if (summary["upstream_recomputed"] is not False
                            or Path(summary["input_catalog_run"]).resolve() != parent
                            or summary["date"] != date or set(summary["catalogs"]) != set(chosen)):
                        raise ValueError("Invalid Stage 09 provenance")
                    verify_pins(frozen)
                    reports = summary["catalogs"]
                    for name in chosen:
                        expected = {k: v["sha256"] for k, v in frozen[name]["files"].items()}
                        actual = {k: v["sha256"] for k, v in reports[name].get("references", {}).items()}
                        if actual != expected:
                            raise ValueError("Stage 09 reference checksum mismatch")
                    with (child / meta["outputs"]["catalog_matches"]["path"]).open() as f:
                        local_matches = [{"algorithm": algorithm, "date": date, **r}
                                         for r in csv.DictReader(f)]
                    matches.extend(local_matches)
                    run_status = "COMPLETED"
                else:
                    run_status = "NO_PINNED_REFERENCE"
            except Exception as exc:
                reports = {}
                run_status = "EVALUATION_FAILED"
                reason = f"{type(exc).__name__}: {exc}"
        for name in settings["catalogs"]:
            ref = frozen.get(name, {})
            report = reports.get(name)
            if run_status == "INPUT_FAILED":
                status, why = run_status, reason
            elif ref.get("status") != "PINNED":
                status, why = ref.get("status", "REFERENCE_ERROR"), ref.get("reason")
            elif report is None:
                status, why = run_status, reason
            else:
                status, why = report["status"], report.get("reason")
            rows.append(dict(algorithm=algorithm, date=date, catalog=name, status=status,
                             matched_events=report.get("matched_events") if report else None,
                             mean_matched_angular_iou=report.get("mean_matched_angular_iou") if report else None,
                             metric=report.get("metric") if report else None,
                             reason=why, input_run=str(parent),
                             validation_run=str(child) if child.exists() else None,
                             observation_sha256=observations.get(key)))
        run_records.append(dict(algorithm=algorithm, date=date, input_run=str(parent),
                                status=run_status, reason=reason))
    write_csv(output / "comparison_summary.csv", SUMMARY_COLUMNS, rows)
    write_csv(output / "matched_events.csv", ["algorithm", "date", *MATCH_COLUMNS], matches)
    plot_results(output, rows, settings["catalogs"])
    errors = {"INPUT_FAILED", "EVALUATION_FAILED", "REFERENCE_ERROR", "ACQUISITION_FAILED"}
    failed = any(r["status"] in errors for r in rows)
    by_date = {date: sorted({v for (a, d), v in observations.items() if d == date})
               for date in sorted({d for a, d in observations})}
    write_json(output / "batch_manifest.json", {
        "status": "COMPLETED_WITH_ERRORS" if failed else "COMPLETED",
        "plan": str(Path(plan).resolve()), "plan_sha256": _hash_file(Path(plan)),
        "settings": settings, "runs": run_records, "observation_hashes_by_date": by_date,
        "note": "Different observation hashes flag nonidentical manifests; matching hashes do not prove identical inputs. No pooled metric.",
        "outputs": {p.name: _hash_file(p) for p in
                    (output / "reference_manifest.json", output / "comparison_summary.csv",
                     output / "matched_events.csv", output / "comparison.png")},
    })
    return not failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    return 0 if run_batch(args.plan, args.output_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
