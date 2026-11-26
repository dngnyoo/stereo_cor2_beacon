"""Normalize COR2-A MVCC monthly JSON and CORSET source measurements."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
import pandas as pd

FIELDS = ("database_id spacecraft event_id date time event_type sepAngle CPA width Vrad "
          "alpha_rad Vexp alpha_exp notes cross_event_id eventPrefix dataPath Mproj "
          "Mdeproj latitude longitude txtFile pngFile").split()
EVENT_COLS = FIELDS + ["catalog_event_id", "measurement_status"]
FRAME_COLS = ["catalog_event_id", "timestamp_utc", "cpa_deg", "angular_width_deg",
              "front_rsun", "source_filename", "source_line"]
EDGE_COLS = ["catalog_event_id", "timestamp_utc", "leading_edge_height",
             "position_angle", "height_unit", "angle_unit", "TEL", "FC",
             "COL", "ROW", "source_filename", "source_line"]
HT_ROW = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.?\s+([+-]?\d+(?:\.\d+)?)\s+(\S+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$")


def parse_date(value):
    day = datetime.strptime(value, "%Y%m%d")
    if day.strftime("%Y%m%d") != value:
        raise ValueError("Expected YYYYMMDD")
    if not "200703" <= value[:6] <= "201409":
        raise ValueError("REFERENCE_UNAVAILABLE: outside MVCC catalog coverage")
    return day.strftime("%Y-%m-%d")


def safe_name(value):
    if not value or value in ("None", "NULL", ".", "..") or PurePosixPath(value).name != value or "\\" in value:
        raise ValueError(f"Invalid source filename: {value}")
    return value


def daily_events(payload, date):
    day = parse_date(date)
    rows = payload["data"]
    if not isinstance(rows, list) or int(payload["recordsFiltered"]) != len(rows):
        raise ValueError("Incomplete monthly JSON")
    result = []
    for values in rows:
        if not isinstance(values, list) or len(values) != len(FIELDS):
            raise ValueError("Unexpected MVCC monthly row schema")
        row = dict(zip(FIELDS, values))
        if row["spacecraft"] != "A" or row["date"] != day:
            continue
        ident = safe_name(row["event_id"])
        row["catalog_event_id"] = ident
        row["measurement_status"] = ("CORSET" if ident.endswith("c") else
                                     "MANUAL" if ident.endswith("m") else "NOT_EVALUABLE")
        result.append(row)
    if len({r["catalog_event_id"] for r in result}) != len(result):
        raise ValueError("Duplicate event IDs")
    return result


def detail_id(row):
    return safe_name(row["event_id"][:-1])


def parse_morphology(path, row):
    lines = path.read_text().splitlines()
    ident = detail_id(row)
    if not any(re.fullmatch(r"\s*A id:\s*" + re.escape(ident) + r"\s*", line) for line in lines):
        raise ValueError(f"STA event ID mismatch: {path}")
    starts = [i for i, line in enumerate(lines) if line.strip() == "Center of Mass Time Series in STA"]
    if len(starts) != 1:
        raise ValueError(f"Missing or ambiguous STA morphology section: {path}")
    start = starts[0]
    out = []
    for i in range(start+1, len(lines)):
        line = lines[i].strip()
        if line.startswith("Center of Mass Time Series in STB"):
            break
        if not line:
            if out:
                break
            continue
        if not re.match(r"^\d{8}_", line):
            if out:
                raise ValueError(f"Unexpected morphology row: {path}:{i+1}")
            continue
        fields = line.split()
        if len(fields) != 10:
            raise ValueError(f"Unexpected morphology row: {path}:{i+1}")
        stamp = datetime.strptime(fields[0], "%Y%m%d_%H%M%S").isoformat() + "Z"
        cpa, height, width = map(float, fields[-3:])
        if not (0 <= cpa <= 360 and 0 < width <= 360 and height > 0):
            raise ValueError(f"Invalid morphology geometry: {path}:{i+1}")
        out.append(dict(catalog_event_id=row["event_id"], timestamp_utc=stamp,
                        cpa_deg=cpa, angular_width_deg=width, front_rsun=height,
                        source_filename=path.name, source_line=i+1))
    if not out or len({r["timestamp_utc"] for r in out}) != len(out):
        raise ValueError(f"Empty or duplicate STA morphology: {path}")
    expected = [int(m.group(1)) for line in lines
                if (m := re.fullmatch(r"\s*Times for STA:\s*(\d+)\s*", line))]
    if expected and expected != [len(out)]:
        raise ValueError(f"STA frame count mismatch: {path}")
    return out


def parse_edges(path, row):
    lines = path.read_text().splitlines()
    headers = {}
    out = []
    for i, line in enumerate(lines, 1):
        if line.startswith("#"):
            key, sep, val = line[1:].partition(":")
            if sep:
                headers[key.strip()] = val.strip()
            continue
        if not line.strip():
            continue
        m = HT_ROW.fullmatch(line)
        if not m:
            raise ValueError(f"Unexpected CORSET HT row: {path}:{i}")
        h, stamp, pa, tel, fc, col, pixel_row = m.groups()
        if tel != "COR2" or not (float(h)>0 and 0<=float(pa)<=360):
            raise ValueError(f"Invalid CORSET HT row: {path}:{i}")
        datetime.fromisoformat(stamp)
        out.append(dict(catalog_event_id=row["event_id"], timestamp_utc=stamp+"Z",
                        leading_edge_height=float(h), position_angle=float(pa),
                        height_unit="not_explicit_in_ht_file", angle_unit="not_explicit_in_ht_file",
                        TEL=tel, FC=int(fc), COL=int(col), ROW=int(pixel_row),
                        source_filename=path.name, source_line=i))
    if headers.get("DETECTOR") != "STEREO_A-COR2" or headers.get("EVENT_ID") != detail_id(row):
        raise ValueError(f"CORSET HT header mismatch: {path}")
    if not out:
        raise ValueError(f"No CORSET HT rows: {path}")
    return out


def normalize(raw, date):
    rows = daily_events(json.loads((raw/"monthly.json").read_text()), date)
    frames, edges = [], []
    for row in rows:
        if row["measurement_status"] != "CORSET":
            continue
        frames.extend(parse_morphology(raw/safe_name(row["txtFile"]), row))
        edges.extend(parse_edges(raw/(detail_id(row)+".corset.ht"), row))
    return (pd.DataFrame(rows, columns=EVENT_COLS),
            pd.DataFrame(frames, columns=FRAME_COLS),
            pd.DataFrame(edges, columns=EDGE_COLS))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args(argv)
    parse_date(args.date)
    raw = args.raw_dir or Path("data/catalogs/mvcc/raw")/args.date[:6]
    out = args.output_dir or Path("data/catalogs/mvcc/normalized")/args.date
    events, frames, edges = normalize(raw, args.date)
    sources = {"monthly.json"}
    for row in events.to_dict("records"):
        if row["measurement_status"] == "CORSET":
            sources.update([safe_name(row["txtFile"]), detail_id(row)+".corset.ht"])
    summary = dict(date=args.date, catalog_events=len(events),
                   evaluable_events=int(frames.catalog_event_id.nunique()),
                   morphology_frames=len(frames), leading_edge_points=len(edges),
                   inputs={name: hashlib.sha256((raw/name).read_bytes()).hexdigest() for name in sorted(sources)},
                   note="Manual and unmeasured events preserved but not evaluated. HT units not explicit.")
    out.mkdir(parents=True, exist_ok=False)
    events.to_csv(out/"mvcc_events.csv", index=False)
    frames.to_csv(out/"mvcc_frames.csv", index=False)
    edges.to_csv(out/"mvcc_leading_edges.csv", index=False)
    (out/"provenance.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
