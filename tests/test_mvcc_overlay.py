from pathlib import Path
import pandas as pd
from PIL import Image
from scripts.visualize_mvcc_overlay import edge_segments, overlay_frame


def test_edge_segments_wrap_and_gap():
    edges=pd.DataFrame({"position_angle":[359,0,1,5],"leading_edge_height":[4,5,6,7]})
    assert edge_segments(edges,0)==[[(-1,4),(0,5),(1,6)],[(5,7)]]


def test_overlay_preserves_source_and_appends_footer(tmp_path):
    source=tmp_path/"pair_033_20101212T052424Z_families.png"
    Image.new("RGB",(864,840),"black").save(source)
    original=source.read_bytes()
    stamp=pd.Timestamp("2010-12-12T05:24:00Z")
    active=pd.DataFrame([dict(catalog_event_id="R",timestamp_utc=stamp,pair_index=33,
                             cpa_deg=90,angular_width_deg=30,front_rsun=5,delta_sec=24)])
    edges=pd.DataFrame([dict(catalog_event_id="R",timestamp_utc=stamp,
                             position_angle=90,leading_edge_height=5)])
    det=pd.DataFrame(columns=["event_id","pair_index","cpa_deg","angular_width_deg"])
    scores=pd.DataFrame([dict(catalog_event_id="R",event_id="D",pair_index=33,angular_iou=0)])
    dest=tmp_path/"output.png"
    overlay_frame(source,dest,active,edges,det,{"R":"D"},scores,
                  sx=128,sy=128,rsun_px=8,assume_ht_rsun=True)
    assert source.read_bytes()==original
    with Image.open(dest) as im:
        assert im.width==864 and im.height>840
        assert im.getpixel((330,428))[1]>0
