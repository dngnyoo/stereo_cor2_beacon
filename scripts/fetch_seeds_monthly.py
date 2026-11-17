"""Download official GMU SEEDS STEREO-A COR2 monthly TXT assets.

Example:
  python -m scripts.fetch_seeds_monthly --date 20101212

This intentionally preserves the ORIGINAL text without guessing column order.
The SEEDS text format must be inspected before translating into the separate
frame-level validation schema. Daily files are statistical/context data only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FILES = ("monthly.txt", "daily.txt", "daily_images.txt")
GMU_BASE = "http://spaceweather.gmu.edu/seeds/secchi/detection_cor2/a"
MAX_BYTES = 25 * 1024 * 1024


def month_url(date: str, filename: str, base: str = GMU_BASE) -> str:
    parsed = datetime.strptime(date, "%Y%m%d")
    if parsed.strftime("%Y%m%d") != date:
        raise ValueError("Date must be YYYYMMDD")
    if filename not in FILES:
        raise ValueError(f"Unsupported SEEDS asset {filename}")
    return f"{base.rstrip('/')}/{parsed:%Y/%m}/{filename}"


def fetch_asset(url: str, *, timeout: float = 30) -> tuple[bytes, str]:
    request = Request(url, headers={
        "User-Agent": "StereoCOR2-SEEDS-research/0.1",
        "Accept": "text/plain,*/*",
    })
    with urlopen(request, timeout=timeout) as res:
        resolved = res.geturl()
        data = res.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError(f"SEEDS download exceeds 25 MiB: {url}")
        if not data.strip():
            raise ValueError(f"Empty SEEDS download: {url}")
        preview = data[:4096].lstrip().lower()
        if (preview.startswith(b"<!doctype html") or
                preview.startswith(b"<html") or
                preview.startswith(b"<?xml")):
            raise ValueError(f"Server returned markup instead of SEEDS text: {url}")
        return data, resolved


def acquire(date: str, *, output_root: Path, base: str = GMU_BASE,
            timeout: float = 30, force: bool = False,
            fetch=fetch_asset) -> dict:
    dt = datetime.strptime(date, "%Y%m%d")
    dest = Path(output_root) / f"{dt:%Y}" / f"{dt:%m}"
    metadata_file = dest / "source_provenance.json"
    if not force and (metadata_file.exists() or any((dest / file).exists() for file in FILES)):
        raise FileExistsError(f"SEEDS raw monthly assets already exist: {dest}; use --force")

    # Fetch and validate all requested files BEFORE changing the destination.
    data = {}
    resolved = {}
    for filename in FILES:
        url = month_url(date, filename, base)
        blob, final_url = fetch(url, timeout=timeout)
        data[filename] = blob
        resolved[filename] = final_url
    metadata = {
        "source": "GMU SEEDS STEREO-A COR2",
        "url_pattern": f"{base.rstrip('/')}/YYYY/MM/",
        "requested_date": date,
        "archive_month": f"{dt:%Y-%m}",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            filename: {
                "requested_url": month_url(date, filename, base),
                "resolved_url": resolved[filename],
                "bytes": len(data[filename]),
                "sha256": hashlib.sha256(data[filename]).hexdigest(),
            }
            for filename in FILES
        },
        "parser_status": "RAW_DOWNLOADED_SCHEMA_NOT_YET_VERIFIED",
        "note": (
            "monthly.txt is the SEEDS event list; daily.txt and "
            "daily_images.txt are daily statistics. Raw download is not "
            "an evaluable frame-level SEEDS reference. No interpolation performed."
        ),
    }
    dest.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        (dest / filename).write_bytes(data[filename])
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"directory": str(dest), "provenance": str(metadata_file)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="UTC date YYYYMMDD")
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/catalogs/seeds/raw"))
    parser.add_argument("--base-url", default=GMU_BASE)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    result = acquire(args.date, output_root=args.output_root,
                     base=args.base_url, timeout=args.timeout,
                     force=args.force)
    from scripts.parse_seeds_monthly import parse_monthly
    raw_path = Path(result["directory"]) / "monthly.txt"
    events = parse_monthly(raw_path.read_text(encoding="utf-8-sig"), date=args.date)
    event_path = Path(result["directory"]) / f"seeds_events_{args.date}.csv"
    events.to_csv(event_path, index=False)
    print(json.dumps(result, indent=2))
    print(f"Parsed {len(events)} SEEDS events: {event_path}")
    print("Event summaries only; no invented SEEDS frame-level measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
