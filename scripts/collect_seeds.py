"""Fetch and normalize a SEEDS COR2-A CSV reference with provenance.

The official GMU SEEDS download endpoint is not verified at this time.
Supply an HTTPS source URL or --url-template explicitly. This utility fails
closed on unexpected HTML, missing columns and other instruments (LASCO/C2).
It does not interpolate trajectories or create fictitious SEEDS frame records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pandas as pd

FIELDS = ["parent_event_id", "seeds_timestamp", "is_evaluable",
          "seeds_bpa", "seeds_width", "seeds_height", "seeds_lead"]
REQUIRED = set(FIELDS[:-1])
MAX_BYTES = 20 * 1024 * 1024


def download(url: str, *, timeout: int = 30) -> tuple[bytes, str]:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Only HTTPS SEEDS source URLs are accepted")
    request = Request(url, headers={
        "User-Agent": "StereoCOR2-SEEDS-validator/0.1 (academic research)",
        "Accept": "text/csv,text/plain,application/octet-stream",
    })
    with urlopen(request, timeout=timeout) as response:
        data = response.read(MAX_BYTES + 1)
        actual_url = response.geturl()
        if urlsplit(actual_url).scheme != "https":
            raise ValueError("Source redirected to a non-HTTPS URL")
        if len(data) > MAX_BYTES:
            raise ValueError("SEEDS source exceeds 20 MiB limit")
        if not data:
            raise ValueError("Empty SEEDS source")
        return data, actual_url


def normalize(raw: bytes, date: str) -> pd.DataFrame:
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError("Date must be YYYYMMDD")
    datetime.strptime(date, "%Y%m%d")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError("SEEDS source is not a parseable CSV") from exc
    if not REQUIRED.issubset(df.columns):
        raise ValueError(
            "Unrecognized SEEDS CSV schema (expected source frame columns; "
            f"missing {sorted(REQUIRED - set(df.columns))}). "
            "HTML pages and event-only catalogs must not be treated as "
            "frame measurements."
        )
    # This curated frame schema may also contain old candidate scores:
    # never preserve those fields in the new independent reference.
    if "seeds_lead" not in df:
        df["seeds_lead"] = float("nan")
    result = df[FIELDS].copy()
    result["seeds_timestamp"] = pd.to_datetime(
        result["seeds_timestamp"], utc=True, errors="raise"
    )
    result = result[result["seeds_timestamp"].dt.strftime("%Y%m%d").eq(date)]
    result["is_evaluable"] = result["is_evaluable"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if result["is_evaluable"].isna().any():
        raise ValueError("Unrecognized is_evaluable values")
    for col in ("seeds_bpa", "seeds_width", "seeds_height", "seeds_lead"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    if result.empty:
        raise ValueError(f"No SEEDS observations for {date}")
    if (result["seeds_bpa"].isna().any()
            or result["seeds_width"].isna().any()
            or (~result["seeds_bpa"].between(0, 360)).any()
            or (~result["seeds_width"].between(0.01, 360)).any()):
        raise ValueError("Invalid SEEDS position angle/width")
    if result.loc[result["is_evaluable"], "seeds_height"].isna().any():
        raise ValueError("Evaluable frames require a SEEDS height for visualization")
    result["seeds_timestamp"] = result["seeds_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    result = result.sort_values(
        ["seeds_timestamp", "parent_event_id"], kind="stable"
    ).reset_index(drop=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="UTC date YYYYMMDD")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-url", help="Verified HTTPS CSV source")
    source.add_argument(
        "--url-template",
        help="HTTPS CSV URL template with {date}, {year}, {month}, {day}",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/catalogs/seeds"))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    datetime.strptime(args.date, "%Y%m%d")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    dt = datetime.strptime(args.date, "%Y%m%d")
    url = args.source_url or args.url_template.format(
        date=args.date, year=dt.strftime("%Y"),
        month=dt.strftime("%m"), day=dt.strftime("%d"),
    )
    root = args.output_root.expanduser().resolve() / args.date
    raw_path = root / "original_source.csv"
    normalized_path = root / "seeds_frame_reference.csv"
    metadata_path = root / "source_provenance.json"
    if not args.force and any(
        p.exists() for p in (raw_path, normalized_path, metadata_path)
    ):
        raise FileExistsError("SEEDS artifacts already exist; use --force to replace")
    data, final_url = download(url, timeout=args.timeout)
    normalized = normalize(data, args.date)
    # Validate before touching output files.
    root.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(data)
    normalized.to_csv(normalized_path, index=False)
    meta = {
        "date": args.date,
        "instrument": "STEREO-A/COR2 (source must be independently verified)",
        "source_url": url,
        "resolved_url": final_url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": hashlib.sha256(data).hexdigest(),
        "raw_bytes": len(data),
        "normalized_rows": len(normalized),
        "schema": "SEEDS curated frame measurements (not generic event catalog)",
        "normalization": "column selection, UTC date filter and schema validation; no interpolation",
    }
    metadata_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"SEEDS reference: {normalized_path}")
    print(f"Provenance: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
