"""Acquire both COR2 archives and resolve same-name d7 files independently of date.

Raw FITS provenance is recorded before de-duplication. The ordinary selection
policy prefers Science L0 d7 when present, otherwise the dedicated Beacon d7.
Frozen reference checks are downstream validation, not source-selection rules.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from cme_pipeline.data_preparation import download_cor2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_dual_archive(days: list[str], root: Path) -> tuple[list[Path], Path, Path]:
    """Download complete FITS listings from both archives, select d7 and audit.

    Source directories are separate. Only byte-identical same-name files are
    removed (the Beacon copy); conflicting versions are retained on disk.
    """
    root = Path(root)
    raw = root / "raw"
    records = []
    chosen = {}
    for day in days:
        downloaded = {}
        for source, product in (("science", "science_raw"), ("beacon", "beacon_raw")):
            directory = raw / source / day
            downloaded[source] = download_cor2(day, product, directory)
        by_source = {
            source: {p.name: p for p in paths}
            for source, paths in downloaded.items()
        }
        names = sorted(set(by_source["science"]) | set(by_source["beacon"]))
        for name in names:
            science = by_source["science"].get(name)
            beacon = by_source["beacon"].get(name)
            d7 = "_d7c2" in name.lower()
            science_hash = sha256_file(science) if science else None
            beacon_hash = sha256_file(beacon) if beacon else None
            equal = science is not None and beacon is not None and science_hash == beacon_hash
            conflict = science is not None and beacon is not None and not equal
            # Consistent policy for any date: Science L0 preferred where present.
            selected = science if science is not None else beacon
            selected_source = "science" if science is not None else "beacon"
            if equal:
                # Already verified the copies byte-for-byte via SHA-256.
                beacon.unlink()
            if d7:
                if name in chosen:
                    raise ValueError(f"Repeated d7 filename across days: {name}")
                chosen[name] = selected
            records.append({
                "date": day, "filename": name,
                "product": "d7" if d7 else ("d4" if "_d4c2" in name.lower() else "other"),
                "science_path": str(science) if science else "",
                "beacon_path": str(beacon) if beacon and not equal else "",
                "science_sha256": science_hash or "",
                "beacon_sha256": beacon_hash or "",
                "duplicate_status": ("identical_removed" if equal else
                                     "different_versions_retained" if conflict else
                                     "single_source"),
                "selected_source": selected_source if d7 else "",
                "selected_path": str(selected) if d7 else "",
            })
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    inventory = manifests / "raw_inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    selected_manifest = manifests / "selected_d7_sources.csv"
    d7_records = [r for r in records if r["product"] == "d7"]
    with selected_manifest.open("w", newline="", encoding="utf-8") as stream:
        if d7_records:
            writer = csv.DictWriter(stream, fieldnames=list(d7_records[0]))
            writer.writeheader()
            writer.writerows(d7_records)
    if not chosen:
        raise ValueError("Neither COR2 archive provided d7 FITS")
    return sorted(chosen.values(), key=lambda p: p.name), inventory, selected_manifest
