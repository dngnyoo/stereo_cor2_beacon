"""Causal 30-minute phase inference must not assume the May 2024 clock phase."""
from datetime import datetime, timedelta

import pytest

from cme_pipeline.cadence_selection import (
    DEFAULT_PHASE_SEC, infer_d7_phase, select_d7_pairs,
)


def _record(t):
    return {
        "file": f"/tmp/{t:%Y%m%d_%H%M%S}_d7c2A.fts",
        "date_obs": t.isoformat(),
    }


def test_infer_2010_clock_phase_from_strict_30_minute_chain():
    # The user's downloaded 2010-12-12 23:54:24 filename is NOT phase 1434.
    origin = datetime(2010, 12, 11, 0, 24, 24)
    frames = [_record(origin + timedelta(minutes=30 * i)) for i in range(96)]
    assert infer_d7_phase(frames) == 1464.0
    pairs = select_d7_pairs(frames, phase_sec=infer_d7_phase(frames))
    assert pairs["n_pairs"] == 95
    with pytest.raises(ValueError, match="No d7 images"):
        select_d7_pairs(frames, phase_sec=DEFAULT_PHASE_SEC)


def test_phase_inference_keeps_gaps_not_bridged():
    origin = datetime(2010, 12, 12, 0, 24, 24)
    offsets = (0, 30, 60, 120, 150)
    records = [_record(origin + timedelta(minutes=m)) for m in offsets]
    result = select_d7_pairs(records, phase_sec=infer_d7_phase(records))
    assert result["n_pairs"] == 3
    assert result["n_gaps"] == 1


def test_phase_inference_errors_without_any_30_minute_pair():
    origin = datetime(2010, 12, 12, 0, 24, 24)
    with pytest.raises(ValueError, match="No strict 30-minute"):
        infer_d7_phase([_record(origin), _record(origin + timedelta(minutes=11))])
