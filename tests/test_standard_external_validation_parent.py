from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import pandas as pd
import pytest
from cme_pipeline.standard_external_validation_parent import parameters, references, execute_external
from cme_pipeline.core.run_layout import verify_stage


def config(date="20240503", catalogs=None):
    return SimpleNamespace(data={"experiment":{"date":date},"validation":{
        "method":"external_catalogs","catalogs":catalogs or ["SEEDS","CACTus","MVCC"]}},
        vis_mode=lambda *args:"off")


def test_coverage_and_missing():
    cfg=config()
    settings,names,date=parameters(cfg)
    assert references(settings,"MVCC",date)[0] is None
    assert references(settings,"CACTus",date)[0] is None


def test_duplicate_catalogs_rejected():
    with pytest.raises(ValueError):
        parameters(config(catalogs=["SEEDS","SEEDS"]))


def test_unavailable_is_not_zero(tmp_path):
    cfg=config()
    parent=tmp_path/"parent"
    parent.mkdir()
    frames=parent/"frames.csv"
    frames.write_text("event_id,timestamp\n")
    meta={"outputs":{"height_time":{"path":"frames.csv"}}}
    with patch("cme_pipeline.standard_external_validation_parent.resolve_catalog_parent",
               return_value=(parent,meta,parent,frames,pd.DataFrame(),pd.DataFrame(),pd.DataFrame())):
        out=execute_external(cfg,input_run=parent,until="validation",run_dir=tmp_path/"out")
    result=verify_stage(out,"validation")
    summary=json.loads((out/result["outputs"]["evaluation_metrics"]["path"]).read_text())
    assert all(r["status"]=="REFERENCE_UNAVAILABLE" and r["mean_matched_angular_iou"] is None
               for r in summary["catalogs"].values())
    assert "mean_matched_angular_iou" not in summary
