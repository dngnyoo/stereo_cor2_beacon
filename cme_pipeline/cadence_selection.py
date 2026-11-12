"""Select STEREO-A COR2 d7 observations on a strict 30-minute cadence.

Adapted from segmentation_pipeline/01_select_d7_30min.py in stereo_cor2_dev.
DATE-OBS is authoritative; filename timestamp is independently validated.
The May 2024 phase (23m54s) is an explicit historical default, not universal.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

CADENCE_SEC = 1800.0
CADENCE_TOL_SEC = 2.0
DEFAULT_PHASE_SEC = 23 * 60 + 54
PHASE_TOL_SEC = 2.0
FILENAME_TIME_RE = re.compile(r"(\d{8})_(\d{6})")


def parse_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if timestamp.tzinfo is not None:
        timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    return timestamp


def phase_distance(timestamp: datetime, phase_sec: float) -> float:
    elapsed = (
        timestamp.hour * 3600 + timestamp.minute * 60 +
        timestamp.second + timestamp.microsecond / 1e6
    )
    difference = abs(elapsed % CADENCE_SEC - phase_sec % CADENCE_SEC)
    return min(difference, CADENCE_SEC - difference)


def is_d7(filename: str) -> bool:
    return "_d7c2" in Path(filename).name.lower()


def select_d7_pairs(
    records: list[dict], *,
    phase_sec: float = DEFAULT_PHASE_SEC,
    phase_tolerance_sec: float = PHASE_TOL_SEC,
    cadence_tolerance_sec: float = CADENCE_TOL_SEC,
    filename_tolerance_sec: float = 1.0,
) -> dict:
    """Return selected observations, strict consecutive pairs and explicit gaps.

    Each input record must provide 'file' and 'date_obs'. Rejects duplicate
    observation timestamps, non-d7 images, and filename/header time conflicts.
    A gap is never bridged to create a spurious running-difference pair.
    """
    if not 0 <= phase_sec < CADENCE_SEC:
        raise ValueError("phase_sec must be between 0 and 1800 (exclusive)")
    if min(phase_tolerance_sec, cadence_tolerance_sec, filename_tolerance_sec) < 0:
        raise ValueError("tolerances must be nonnegative")
    frames = []
    for record in records:
        name = Path(record["file"]).name
        if not is_d7(name):
            raise ValueError(f"Non-d7 FITS in d7 selection: {name}")
        observed = parse_timestamp(record["date_obs"])
        match = FILENAME_TIME_RE.search(name)
        if match is None:
            raise ValueError(f"Missing filename timestamp: {name}")
        file_time = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
        if abs((file_time - observed).total_seconds()) > filename_tolerance_sec:
            raise ValueError(f"FITS filename/DATE-OBS mismatch: {name}")
        frames.append({"file": record["file"], "date_obs": record["date_obs"],
                       "timestamp": observed})
    frames.sort(key=lambda frame: (frame["timestamp"], frame["file"]))
    if any(cur["timestamp"] <= prev["timestamp"] for prev, cur in zip(frames, frames[1:])):
        raise ValueError("Non-unique or non-increasing DATE-OBS timestamps")
    selected = [f for f in frames if phase_distance(f["timestamp"], phase_sec) <= phase_tolerance_sec]
    if not selected:
        raise ValueError(f"No d7 images match the requested 30-minute phase ({phase_sec}s)")
    pairs, gaps = [], []
    for previous, current in zip(selected, selected[1:]):
        dt = (current["timestamp"] - previous["timestamp"]).total_seconds()
        payload = {
            "start_file": previous["file"], "end_file": current["file"],
            "start_time": previous["timestamp"].isoformat(),
            "end_time": current["timestamp"].isoformat(),
            "delta_t_sec": dt, "delta_t_min": dt / 60,
        }
        if abs(dt - CADENCE_SEC) <= cadence_tolerance_sec:
            pairs.append({"pair_index": len(pairs) + 1, **payload})
        elif dt > CADENCE_SEC + cadence_tolerance_sec:
            expected_steps = round(dt / CADENCE_SEC)
            if expected_steps < 2 or abs(dt - expected_steps * CADENCE_SEC) > 2 * phase_tolerance_sec + 1e-8:
                raise ValueError(f"Gap inconsistent with selected phase: {payload}")
            gaps.append({**payload, "estimated_missing_slots": expected_steps - 1})
        else:
            raise ValueError(f"Interval is not a valid pair or gap: {payload}")
    return {
        "phase_sec": phase_sec,
        "cadence_sec": CADENCE_SEC,
        "selected_frames": [{"file": f["file"], "date_obs": f["date_obs"]}
                            for f in selected],
        "pairs": pairs, "gaps": gaps,
        "n_input": len(frames), "n_selected": len(selected),
        "n_pairs": len(pairs), "n_gaps": len(gaps),
    }
