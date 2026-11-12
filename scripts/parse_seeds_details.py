"""Parse official GMU SEEDS COR2-A event detail TXT into event/frame CSVs.

Does not infer missing values, timestamps or pixel masks. Source metadata
is preserved, including original filename and SHA-256 provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import re

import pandas as pd


EVENT_KEYS = (
    "DATE_CREATED", "DATE_OBS", "TIME_OBS", "DETECTOR", "OBSERVER",
    "IMAGE_TYPE", "NUM_FRAMES", "ONSET1", "ONSET2", "ONSET2_RSUN",
    "CEN_PA", "WIDTH", "SPEED", "ACCEL", "REMARK",
)
FRAME_KEYS = ("DATE", "TIME", "HMLEAD", "HMHEIGHT", "HMFOLLOW",
              "TEL", "PASTART", "PAEND", "BPA")
FILENAME = re.compile(
    r"^(\d{8})\.(\d{6})\.w(\d{3})\.v(\d{4})\.p(\d{3})\.txt$"
)
MISSING = {"", "-------", "------", "-----", "----", "---", "--", "-"}


def parse_detail(contents: bytes, filename: str) -> tuple[dict, list[dict]]:
    match = FILENAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unrecognized SEEDS event filename: {filename}")
    headers: dict[str, str] = {}
    frames: list[dict] = []
    table_started = False
    for line_no, raw_line in enumerate(contents.decode("utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if body.upper().startswith("DATE ") and not ":" in body:
                fields = body.split()
                if tuple(fields) != FRAME_KEYS:
                    raise ValueError(f"Unexpected frame columns at line {line_no}: {fields}")
                table_started = True
            elif ":" in body:
                if table_started:
                    raise ValueError("Header found after frame measurements")
                key, value = body.split(":", 1)
                key = key.strip().upper()
                if key not in EVENT_KEYS:
                    raise ValueError(f"Unexpected SEEDS header {key}")
                if key in headers:
                    raise ValueError(f"Duplicate SEEDS header {key}")
                headers[key] = value.strip()
            else:
                raise ValueError(f"Unexpected comment at line {line_no}: {line}")
            continue
        if not table_started:
            raise ValueError(f"Data before #DATE frame header at line {line_no}")
        tokens = line.split()
        if len(tokens) != len(FRAME_KEYS):
            raise ValueError(f"Frame line {line_no} expected 9 fields, found {len(tokens)}")
        row = dict(zip(FRAME_KEYS, tokens))
        for field in ("HMLEAD", "HMHEIGHT", "HMFOLLOW", "PASTART", "PAEND", "BPA"):
            row[field] = float(row[field])
        datetime.strptime(row["DATE"] + " " + row["TIME"], "%Y/%m/%d %H:%M:%S")
        frames.append(row)
    missing = set(EVENT_KEYS) - set(headers)
    if missing:
        raise ValueError(f"Missing detail headers: {sorted(missing)}")
    if not table_started:
        raise ValueError("Missing per-frame #DATE table header")
    if headers["DETECTOR"].lower() != "cor2a":
        raise ValueError("Expected STEREO-A COR2 detail")
    date, time, width, speed, pa = match.groups()
    obs = datetime.strptime(headers["DATE_OBS"] + " " + headers["TIME_OBS"],
                            "%Y/%m/%d %H:%M:%S")
    if (obs.strftime("%Y%m%d") != date or obs.strftime("%H%M%S") != time
        or int(headers["WIDTH"]) != int(width)
        or int(headers["SPEED"]) != int(speed)
        or int(headers["CEN_PA"]) != int(pa)):
        raise ValueError(f"SEEDS filename and event header disagree: {filename}")
    if int(headers["NUM_FRAMES"]) != len(frames):
        raise ValueError(f"NUM_FRAMES mismatch: {filename}")
    if any(row["TEL"].lower() != "cor2a" for row in frames):
        raise ValueError("Unexpected per-frame detector")
    event_id = (
        f"SEEDS_COR2A_{date}T{time}_W{width}_V{speed}_P{pa}"
    )
    def nullify(value: str):
        return None if value in MISSING else value
    event = {"event_id": event_id}
    event.update({key: nullify(headers[key]) for key in EVENT_KEYS})
    for key in ("NUM_FRAMES", "CEN_PA", "WIDTH", "SPEED"):
        event[key] = int(event[key])
    for key in ("ONSET2_RSUN", "ACCEL"):
        event[key] = float(event[key]) if event[key] is not None else None
    event["source_filename"] = filename
    event["provenance"] = (
        f"gmu_seeds_cor2a_event_detail;sha256={hashlib.sha256(contents).hexdigest().upper()}"
    )
    for row in frames:
        row["parent_event_id"] = event_id
        row["source_filename"] = filename
    return event, frames


def parse_directory(source_dir: Path, date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    datetime.strptime(date, "%Y%m%d")
    sources = sorted(Path(source_dir).glob(f"{date}.*.txt"))
    if not sources:
        raise FileNotFoundError(f"No SEEDS event detail TXT files in {source_dir}")
    event_rows = []
    frame_rows = []
    for path in sources:
        event, frames = parse_detail(path.read_bytes(), path.name)
        event_rows.append(event)
        frame_rows.extend(frames)
    events = pd.DataFrame(
        event_rows, columns=["event_id", *EVENT_KEYS, "source_filename", "provenance"]
    )
    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate SEEDS event IDs")
    frames = pd.DataFrame(
        frame_rows, columns=[*FRAME_KEYS, "parent_event_id", "source_filename"]
    )
    return events, frames


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="UTC date YYYYMMDD")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    source = args.input_dir or Path("data/catalogs/seeds/details") / args.date
    output = args.output_dir or Path("data/catalogs/seeds") / args.date
    events, frames = parse_directory(source, args.date)
    output.mkdir(parents=True, exist_ok=True)
    events.to_csv(output / "seeds_events.csv", index=False)
    frames.to_csv(output / "seeds_frames.csv", index=False)
    print(f"SEEDS COR2-A: {len(events)} events, {len(frames)} frames -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
