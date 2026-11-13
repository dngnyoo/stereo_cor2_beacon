"""Download original GMU SEEDS COR2-A event detail TXT files for a UTC date.

Source filenames are derived from parsed GMU monthly.txt event rows, never
guessed from frame rows. This stage only archives source bytes. The header
and per-frame measurement parser is intentionally separate so unexpected
upstream schema cannot silently generate inaccurate scientific measurements.

Usage:
  python -m scripts.fetch_seeds_details --date 20101212
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from scripts.fetch_seeds_monthly import GMU_BASE, fetch_asset, month_url
from scripts.parse_seeds_monthly import parse_monthly


def fetch_details(
    date: str,
    *,
    raw_root: Path = Path("data/catalogs/seeds/raw"),
    output_root: Path = Path("data/catalogs/seeds/details"),
    base: str = GMU_BASE,
    timeout: float = 30.,
    force: bool = False,
    fetch=fetch_asset,
) -> dict:
    dt = datetime.strptime(date, "%Y%m%d")
    month_dir = Path(raw_root) / f"{dt:%Y}" / f"{dt:%m}"
    monthly = month_dir / "monthly.txt"
    if not monthly.is_file():
        raise FileNotFoundError(
            f"Monthly list missing: {monthly}. First run "
            f"'python -m scripts.fetch_seeds_monthly --date {date}'."
        )
    events = parse_monthly(monthly.read_text(encoding="utf-8-sig"), date=date)
    dest = Path(output_root) / date
    # Pull exact event identifiers already available from monthly records.
    rows = []
    blobs = {}
    for row in events.itertuples(index=False):
        filename = (
            f"{dt:%Y%m%d}."
            f"{row.timestamp_utc[11:13]}{row.timestamp_utc[14:16]}{row.timestamp_utc[17:19]}."
            f"w{int(row.angular_width_deg):03d}."
            f"v{int(round(row.speed_km_s)):04d}."
            f"p{int(row.pa_deg):03d}.txt"
        )
        path = dest / filename
        if not force and path.exists():
            body = path.read_bytes()
            url = month_url(date, "monthly.txt", base).rsplit("/", 1)[0] + "/" + quote(filename)
            resolved = "local-cached"
        else:
            url = month_url(date, "monthly.txt", base).rsplit("/", 1)[0] + "/" + quote(filename)
            body, resolved = fetch(url, timeout=timeout)
        blobs[filename] = body
        rows.append({
            "event_id": row.event_id,
            "source_filename": filename,
            "requested_url": url,
            "resolved_url": resolved,
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        })
    # Only create new files after every request succeeds.
    dest.mkdir(parents=True, exist_ok=True)
    for name, body in blobs.items():
        target = dest / name
        if force or not target.exists():
            target.write_bytes(body)
    metadata = {
        "date": date,
        "instrument": "STEREO-A COR2",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "monthly_source": str(monthly.resolve()),
        "monthly_sha256": hashlib.sha256(monthly.read_bytes()).hexdigest(),
        "detail_count": len(rows),
        "details": rows,
        "parser_status": "DOWNLOADED_RAW_DETAILS_NOT_YET_PARSED",
    }
    (dest / "details_provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {"directory": str(dest), "detail_count": len(rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--raw-root", type=Path, default=Path("data/catalogs/seeds/raw"))
    ap.add_argument("--output-root", type=Path, default=Path("data/catalogs/seeds/details"))
    ap.add_argument("--base-url", default=GMU_BASE)
    ap.add_argument("--timeout", type=float, default=30.)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.timeout <= 0:
        ap.error("--timeout must be positive")
    result = fetch_details(
        args.date, raw_root=args.raw_root, output_root=args.output_root,
        base=args.base_url, timeout=args.timeout, force=args.force,
    )
    from scripts.parse_seeds_details import parse_directory
    events, frames = parse_directory(Path(result["directory"]), args.date)
    output = Path("data/catalogs/seeds") / args.date
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "seeds_events.csv", index=False)
    frames.to_csv(output / "seeds_frames.csv", index=False)
    print(json.dumps(result, indent=2))
    print(f"Parsed {len(events)} SEEDS events and {len(frames)} frames -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
