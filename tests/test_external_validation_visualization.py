from pathlib import Path
import pandas as pd
from cme_pipeline.external_validation_visualization import choose,xy,draw_sector,gif,render


def test_selection():
    assert choose(range(15),"all")==list(range(15))
    picked=choose(range(15),"diagnostic")
    assert len(picked)==6 and picked[0]==0 and picked[-1]==14


def test_pa_convention():
    import numpy as np
    assert np.allclose(xy((128,128),90,20),(108,128))
    assert np.allclose(xy((128,128),0,20),(128,148))


def test_unavailable_summary(tmp_path):
    outputs=render(None,None,None,{},{"MVCC":{"status":"REFERENCE_UNAVAILABLE"}},{},None,tmp_path/"viz","all")
    assert outputs["validation_summary_png"].is_file()
    assert outputs["visualization_manifest"].is_file()
    assert not (tmp_path/"viz/mvcc").exists()
