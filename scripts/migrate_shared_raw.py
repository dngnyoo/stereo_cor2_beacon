"""Safely migrate legacy per-run COR2 raw FITS into the shared data tree.

Default: read-only dry run. --apply creates verified copies (or hard links
with --hardlink) and does not delete or rewrite old data/manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def migrate(source: Path, destination: Path, *, apply: bool = False,
            hardlink: bool = False) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if source.name != "raw":
        raise ValueError("Source must be a legacy acquisition/raw directory")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination == source or source in destination.parents:
        raise ValueError("Destination must not be nested within source")
    planned = []
    for archive in ("science", "beacon"):
        archive_root = source / archive
        if not archive_root.exists():
            continue
        for day_dir in sorted(archive_root.iterdir()):
            if not day_dir.is_dir() or len(day_dir.name) != 8 or not day_dir.name.isdigit():
                continue
            for raw in sorted(day_dir.iterdir()):
                if raw.is_file() and raw.suffix.lower() in {".fts", ".fits", ".fit"}:
                    planned.append((raw, destination / archive / day_dir.name / raw.name))
    if not planned:
        raise ValueError("No source FITS found; nothing to migrate")
    created = unchanged = 0
    for src, dst in planned:
        if dst.is_file():
            if src.stat().st_size != dst.stat().st_size or digest(src) != digest(dst):
                raise FileExistsError(f"Existing destination differs: {dst}")
            unchanged += 1
            continue
        if dst.exists():
            raise FileExistsError(f"Destination not a regular file: {dst}")
        if not apply:
            print(f"WOULD ADD {src} -> {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        temp = dst.with_name(dst.name + ".migrating")
        if temp.exists():
            raise FileExistsError(f"Partial prior migration file: {temp}")
        try:
            if hardlink:
                os.link(src, temp)
            else:
                shutil.copy2(src, temp)
            if src.stat().st_size != temp.stat().st_size or digest(src) != digest(temp):
                raise IOError(f"Migration checksum mismatch: {src}")
            temp.rename(dst)
            created += 1
        finally:
            temp.unlink(missing_ok=True)
    return {
        "source": str(source),
        "destination": str(destination),
        "discovered": len(planned),
        "created": created,
        "already_identical": unchanged,
        "dry_run": not apply,
        "hardlink": hardlink,
        "legacy_paths_preserved": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-raw", type=Path, required=True)
    parser.add_argument("--to-raw", type=Path, default=Path("data/raw/cor2"))
    parser.add_argument("--apply", action="store_true", help="Perform migration")
    parser.add_argument("--hardlink", action="store_true",
                        help="Share disk blocks with old files; same filesystem only")
    args = parser.parse_args()
    if args.hardlink and not args.apply:
        parser.error("--hardlink requires --apply")
    print(migrate(args.from_raw, args.to_raw, apply=args.apply,
                  hardlink=args.hardlink))


if __name__ == "__main__":
    main()
