from unittest.mock import patch
from urllib.error import HTTPError
import pytest
from cme_pipeline.external_reference_acquisition import prepare, acquisition_settings


def settings(tmp_path):
    return {"acquisition":{"mode":"auto","cache_root":str(tmp_path)}}


def test_mvcc_outside_coverage(tmp_path):
    with patch("scripts.fetch_mvcc.main") as fetch:
        _,r=prepare("MVCC","20240503",settings(tmp_path))
        assert r["status"]=="OUTSIDE_COVERAGE"
        fetch.assert_not_called()


def test_cactus_normalized_reuse(tmp_path):
    p=tmp_path/"cactus/normalized/20240503/cactus_events.csv"
    p.parent.mkdir(parents=True)
    p.write_text("catalog_event_id\n")
    with patch("scripts.fetch_cactus_monthly.acquire") as fetch:
        effective,r=prepare("CACTus","20240503",settings(tmp_path))
        assert effective["cactus_events_csv"]==str(p)
        assert r["status"]=="REUSED_NORMALIZED"
        fetch.assert_not_called()


def test_cactus_auto_fetch_and_parse(tmp_path):
    p=tmp_path/"cactus/normalized/20240503/cactus_events.csv"
    def parse(args):
        p.parent.mkdir(parents=True)
        p.write_text("catalog_event_id\n")
    with patch("scripts.fetch_cactus_monthly.acquire") as fetch, patch("scripts.parse_cactus_monthly.main",side_effect=parse):
        _,r=prepare("CACTus","20240503",settings(tmp_path))
        assert r["status"]=="PREPARED"
        assert fetch.call_args.kwargs["reuse_existing"] is False


def test_http_failure_distinguished(tmp_path):
    with patch("scripts.fetch_cactus_monthly.acquire",side_effect=HTTPError("url",503,"failure",None,None)):
        _,r=prepare("CACTus","20240503",settings(tmp_path))
        assert r["status"]=="ACQUISITION_FAILED"


def test_explicit_path_not_replaced(tmp_path):
    s=settings(tmp_path)
    s["cactus_events_csv"]="missing.csv"
    with patch("scripts.fetch_cactus_monthly.acquire") as fetch:
        effective,r=prepare("CACTus","20240503",s)
        assert effective["cactus_events_csv"]=="missing.csv"
        assert r["status"]=="EXPLICIT_PATHS"
        fetch.assert_not_called()


def test_no_overwrite(tmp_path):
    p=tmp_path/"mvcc/normalized/20101212/mvcc_events.csv"
    p.parent.mkdir(parents=True)
    p.write_text("original")
    with pytest.raises(ValueError,match="Incomplete"):
        prepare("MVCC","20101212",settings(tmp_path))
    assert p.read_text()=="original"
