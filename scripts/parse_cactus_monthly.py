"""Parse NRL CACTus COR2-A monthly event summaries without interpolation."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

COLUMNS = [
    "catalog_event_id", "event_type", "original_event_id", "catalog_month",
    "onset_utc", "liftoff_duration_hours", "cpa_deg", "angular_width_deg",
    "median_speed_km_s", "speed_std_km_s", "min_speed_km_s",
    "max_speed_km_s", "halo_flag",
]
HEADER = re.compile(r"^#\s*(CME|Flow)\s*\|\s*t0\s*\|\s*dt0\s*\|\s*pa\s*\|\s*da\s*\|\s*v\s*\|\s*dv\s*\|\s*minv\s*\|\s*maxv\s*\|\s*halo\?\s*$")


def parse_date(date: str) -> datetime:
    dt = datetime.strptime(date, "%Y%m%d")
    if dt.strftime("%Y%m%d") != date:
        raise ValueError("Expected YYYYMMDD")
    return dt


def parse_monthly(raw: str, catalog_month: str) -> pd.DataFrame:
    """Preserve all rows, including adjacent-month onsets and suspicious flows."""
    parse_date(catalog_month + "01")
    if not re.search(r"(?m)^# Instrument: SECCHI\s*\|\s*Detector: cor2\s*$", raw):
        raise ValueError("Missing CACTus SECCHI COR2 instrument header")
    for line in raw.splitlines():
        if re.match(r"^\s*(first|last)\s+cor2:", line):
            if not re.search(r"_04c2A\.fts\s*$", line):
                raise ValueError("Reference is not verified COR2-A")
    if not re.search(r"(?m)^\s*first\s+cor2:.*_04c2A\.fts\s*$", raw):
        raise ValueError("Missing COR2-A observation header")
    rows, seen, sections = [], set(), set()
    section = None
    for number, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        header = HEADER.fullmatch(line)
        if header:
            section = header.group(1)
            if section in sections:
                raise ValueError(f"Repeated CACTus section at line {number}")
            sections.add(section)
            continue
        if not line or line.startswith(("#", ":")):
            continue
        if section is None and re.match(r"^(first|last)\s+cor2:", line):
            continue
        fields = [x.strip() for x in line.split("|")]
        if section is None or len(fields) != 10:
            raise ValueError(f"Malformed CACTus row {number}: {line!r}")
        ident, onset, duration, pa, width, speed, dv, lo, hi, halo = fields
        if not re.fullmatch(r"\d{4}", ident) or not all(
            re.fullmatch(r"\d+", x) for x in (duration, pa, width, speed, dv, lo, hi)
        ):
            raise ValueError(f"Malformed CACTus numeric fields at line {number}")
        stamp = datetime.strptime(onset, "%Y/%m/%d %H:%M")
        duration, pa, width, speed, dv, lo, hi = map(
            int, (duration, pa, width, speed, dv, lo, hi)
        )
        if not (0 <= pa <= 360 and 0 < width <= 360 and lo <= speed <= hi):
            raise ValueError(f"Invalid CACTus geometry or speed at line {number}")
        if halo not in ("", "II", "III", "IV"):
            raise ValueError(f"Invalid CACTus halo flag at line {number}")
        key = (section, ident)
        if key in seen:
            raise ValueError(f"Duplicate CACTus event at line {number}")
        seen.add(key)
        rows.append(dict(zip(COLUMNS, (
            f"CACTUS_COR2A_{catalog_month}_{section.upper()}_{ident}",
            section, ident, catalog_month, stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            duration, pa, width, speed, dv, lo, hi, halo or None,
        ))))
    if "CME" not in sections:
        raise ValueError("Missing CACTus CME section")
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="Archive month selected by YYYYMMDD")
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output-root", type=Path, default=Path("data/catalogs/cactus"))
    ap.add_argument("--day-only", action="store_true", help="Select onset date only")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    dt = parse_date(args.date)
    month = dt.strftime("%Y%m")
    raw = args.input or args.output_root / "raw" / dt.strftime("%Y/%m") / "cmecat.txt"
    contents = raw.read_bytes()
    table = parse_monthly(contents.decode("utf-8-sig"), month)
    if args.day_only:
        table = table.loc[table.onset_utc.str.startswith(dt.strftime("%Y-%m-%d"))]
    dest = args.output_root / "normalized" / (args.date if args.day_only else month)
    paths = [dest / n for n in ("cactus_events.csv", "cactus_flows.csv", "source_provenance.json")]
    if not args.force and any(p.exists() for p in paths):
        raise FileExistsError(f"Normalized output already exists: {dest}; use --force")
    dest.mkdir(parents=True, exist_ok=True)
    counts = {}
    for kind, path in zip(("CME", "Flow"), paths):
        selected = table.loc[table.event_type.eq(kind)]
        selected.to_csv(path, index=False)
        counts[kind] = len(selected)
    paths[2].write_text(json.dumps({
        "catalog_month": month, "selected_onset_date": args.date if args.day_only else None,
        "input": str(raw.resolve()), "input_sha256": hashlib.sha256(contents).hexdigest(),
        "counts": counts, "interpretation": "Event summaries only; no frame measurements inferred.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Parsed {counts} into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
