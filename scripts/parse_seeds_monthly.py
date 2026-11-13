"""Parse GMU SEEDS COR2-A monthly.txt event rows (no frame interpolation).

Source row:
YYYY/MM/DD HH:MM:SS N PA WIDTH SPEED AUX

The last field is preserved as an uninterpreted auxiliary value until the
official parameter description can be checked. '-------' is missing.
The parser uses the known event-level fields only. It does not invent
framewise SEEDS PA/height tracks.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROW = re.compile(
    r"^(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?|-+)$"
)
COLUMNS = [
    "event_id", "timestamp_utc", "observation_count",
    "pa_deg", "angular_width_deg", "speed_km_s",
    "auxiliary_raw", "auxiliary_numeric",
]


def parse_monthly(raw: str, date: str | None = None) -> pd.DataFrame:
    if date is not None:
        parsed = datetime.strptime(date, "%Y%m%d")
        if parsed.strftime("%Y%m%d") != date:
            raise ValueError("Expected YYYYMMDD")
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = ROW.fullmatch(line)
        if m is None:
            raise ValueError(f"Malformed SEEDS monthly.txt row {number}: {line!r}")
        dat, clock, n, pa, width, speed, extra = m.groups()
        stamp = datetime.strptime(f"{dat} {clock}", "%Y/%m/%d %H:%M:%S")
        if date is not None and stamp.strftime("%Y%m%d") != date:
            continue
        n, pa, width, speed = int(n), int(pa), int(width), float(speed)
        if n < 1 or not 0 <= pa <= 360 or not 0 < width <= 360:
            raise ValueError(f"Invalid SEEDS event geometry in row {number}")
        if not np.isfinite(speed):
            raise ValueError(f"Invalid SEEDS speed in row {number}")
        rows.append({
            "event_id": f"SEEDS_COR2A_{stamp:%Y%m%dT%H%M%S}_W{width:03d}_V{int(round(speed)):04d}_P{pa:03d}",
            "timestamp_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observation_count": n,
            "pa_deg": pa,
            "angular_width_deg": width,
            "speed_km_s": speed,
            "auxiliary_raw": extra,
            "auxiliary_numeric": float(extra) if not extra.startswith("--") else np.nan,
        })
    if not rows:
        raise ValueError("No SEEDS event records matched the requested date")
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    dt = datetime.strptime(args.date, "%Y%m%d")
    raw = args.input or (
        Path("data/catalogs/seeds/raw")
        / f"{dt:%Y}" / f"{dt:%m}" / "monthly.txt"
    )
    output = args.output or (
        Path("data/catalogs/seeds") / args.date / "seeds_events.csv"
    )
    contents = raw.read_bytes()
    data = parse_monthly(contents.decode("utf-8-sig"), date=args.date)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    provenance = {
        "date": args.date,
        "input": str(raw.resolve()),
        "input_sha256": hashlib.sha256(contents).hexdigest(),
        "event_count": len(data),
        "interpretation": "monthly event summaries; NOT framewise SEEDS measurements",
        "auxiliary_column": "unverified; preserved raw and numeric, not used for evaluation",
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Parsed {len(data)} SEEDS events: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
