import pandas as pd
from scripts.validate_mvcc import validate_mvcc


def test_available_observation_not_detection_controls_denominator():
    ref=pd.DataFrame([("R","2010-12-12T05:24:00Z",90,30),
                      ("R","2010-12-12T05:39:00Z",90,30),
                      ("R","2010-12-12T05:54:00Z",90,30)],
                     columns=["catalog_event_id","timestamp_utc","cpa_deg","angular_width_deg"])
    obs=pd.DataFrame([(1,"2010-12-12T05:24:24Z"),(2,"2010-12-12T05:54:24Z")],
                     columns=["pair_index","end_time"])
    det=pd.DataFrame([("D",1,"2010-12-12T05:24:24Z",90,30)],
                     columns=["event_id","pair_index","timestamp","cpa_deg","angular_width_deg"])
    events=pd.DataFrame({"catalog_event_id":["R","unmeasured"]})
    p,f,a,s,summary=validate_mvcc(ref,det,obs,events)
    assert summary["aligned_frames"]==2
    assert summary["no_observation_frames"]==1
    assert summary["mean_matched_angular_iou"]==0.5
    assert list(f.angular_iou)==[1.,0.]
    assert s.loc[s.event_id.eq("unmeasured"),"status"].iloc[0]=="NOT_EVALUABLE"
    assert p.iloc[0].matched


def test_no_detection_still_aligned():
    ref=pd.DataFrame([("R","2010-12-12T05:24:00Z",90,30)],
                     columns=["catalog_event_id","timestamp_utc","cpa_deg","angular_width_deg"])
    obs=pd.DataFrame([(1,"2010-12-12T05:24:24Z")],columns=["pair_index","end_time"])
    det=pd.DataFrame(columns=["event_id","pair_index","timestamp","cpa_deg","angular_width_deg"])
    _,_,_,s,summary=validate_mvcc(ref,det,obs,pd.DataFrame({"catalog_event_id":["R"]}))
    assert summary["aligned_frames"]==1
    assert summary["matched_events"]==0
    assert summary["mean_matched_angular_iou"] is None
    assert s.iloc[0].status=="UNMATCHED_REFERENCE"
