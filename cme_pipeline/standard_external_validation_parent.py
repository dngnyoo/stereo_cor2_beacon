"""Stage 09 multi-catalog validation of immutable Stage 08 products."""
from __future__ import annotations
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from cme_pipeline.core.run_layout import initialize_run, complete_stage, _hash_file
from cme_pipeline.standard_runner import _revision, _context, _params, _stage_outputs
from cme_pipeline.standard_seeds_validation_parent import resolve_catalog_parent
from scripts.validate_seeds import _read_seeds, validate_seeds
from scripts.validate_cactus import validate_cactus
from scripts.validate_mvcc import validate_mvcc

NAMES = ("SEEDS", "CACTus", "MVCC")
KEYS = {"SEEDS": ("seeds_frames_csv",), "CACTus": ("cactus_events_csv",),
        "MVCC": ("mvcc_frames_csv", "mvcc_events_csv")}
OPTIONS = {"method", "catalogs", "visualization", "seeds_frames_csv",
           "cactus_events_csv", "mvcc_frames_csv", "mvcc_events_csv",
           "max_time_delta_sec", "min_mean_iou", "min_common_frames",
           "cactus_min_iou", "mvcc_max_time_delta_sec", "mvcc_min_mean_iou"}
MATCH_COLUMNS = ["catalog", "catalog_event_id", "event_id", "angular_iou", "metric"]


def parameters(config):
    settings = _params(config, "validation")
    if set(settings) - OPTIONS:
        raise ValueError(f"Unknown validation settings: {sorted(set(settings)-OPTIONS)}")
    catalogs = settings.get("catalogs")
    if not isinstance(catalogs, (list,tuple)) or not catalogs or any(x not in NAMES for x in catalogs) or len(set(catalogs)) != len(catalogs):
        raise ValueError("Specify unique validation.catalogs from SEEDS, CACTus, MVCC")
    date = str(config.data["experiment"]["date"])
    datetime.strptime(date,"%Y%m%d")
    common = settings.get("min_common_frames",1)
    if isinstance(common,bool) or not isinstance(common,int) or common<1:
        raise ValueError("Invalid min_common_frames")
    for key, default, low, high in (
        ("max_time_delta_sec",900,0,None), ("min_mean_iou",.1,0,1),
        ("cactus_min_iou",0,0,1), ("mvcc_max_time_delta_sec",60,0,None),
        ("mvcc_min_mean_iou",.1,0,1)):
        x=float(settings.get(key,default))
        if not math.isfinite(x) or x<low or (high is not None and x>high) or (key in ("min_mean_iou","mvcc_min_mean_iou") and x==0):
            raise ValueError(f"Invalid {key}")
    return settings, catalogs, date


def references(settings, name, date):
    if name=="MVCC" and not "200703"<=date[:6]<="201409":
        return None, "outside published MVCC coverage (200703–201409)"
    paths={}
    for key in KEYS[name]:
        source=settings.get(key)
        if source is None:
            return None, f"missing explicit validation.{key}"
        if not isinstance(source,str) or not source.strip():
            raise ValueError(f"Invalid validation.{key}")
        path=Path(source).expanduser().resolve()
        if not path.is_file():
            return None, f"reference file absent: {path}"
        paths[key]=path
    return paths,None


def evaluate(name, paths, settings, date, frames_file, frames, observations):
    """Reuse scientific evaluators unchanged; normalize only matched summaries."""
    raw=pd.read_csv(frames_file)
    extras={}
    if name=="SEEDS":
        seeds=_read_seeds(paths["seeds_frames_csv"])
        if not seeds.parent_event_id.astype(str).str.startswith(f"SEEDS_COR2A_{date}T").all():
            raise ValueError("SEEDS reference date/instrument mismatch")
        pairs,status,summary=validate_seeds(seeds,frames,
            tolerance_sec=float(settings.get("max_time_delta_sec",900)),
            min_iou=float(settings.get("min_mean_iou",.1)),
            min_common_frames=settings.get("min_common_frames",1))
        chosen=pairs.loc[pairs.assigned_match.astype(bool)]
        metric="mean_frame_angular_iou"
        unified=chosen.rename(columns={"seeds_event_id":"catalog_event_id",
            "detected_event_id":"event_id","mean_angular_iou":"angular_iou"})
        summary["limitations"]=[x for x in summary["limitations"] if "2010-12-12" not in x]
    elif name=="CACTus":
        ref=pd.read_csv(paths["cactus_events_csv"])
        if not ref.catalog_event_id.astype(str).str.startswith(f"CACTUS_COR2A_{date[:6]}_").all():
            raise ValueError("CACTus reference month/instrument mismatch")
        pairs,status,summary=validate_cactus(ref,raw,time_mode="liftoff",
            evaluation_date=date,min_iou=float(settings.get("cactus_min_iou",0)))
        chosen=pairs.loc[pairs.matched.astype(bool)]
        metric="liftoff_window_angular_sector_union_iou"
        unified=chosen
    else:
        ref=pd.read_csv(paths["mvcc_frames_csv"])
        events=pd.read_csv(paths["mvcc_events_csv"])
        day=f"{date[:4]}-{date[4:6]}-{date[6:]}"
        if not events.date.astype(str).eq(day).all() or not events.spacecraft.eq("A").all():
            raise ValueError("MVCC event date/instrument mismatch")
        if not set(ref.catalog_event_id).issubset(set(events.loc[events.measurement_status.eq("CORSET"),"catalog_event_id"])):
            raise ValueError("MVCC frames not from declared CORSET events")
        pairs,scores,alignment,status,summary=validate_mvcc(ref,raw,observations,events,
            tolerance_sec=float(settings.get("mvcc_max_time_delta_sec",60)),
            min_iou=float(settings.get("mvcc_min_mean_iou",.1)))
        extras={"frame_scores":scores,"observation_alignment":alignment}
        chosen=pairs.loc[pairs.matched.astype(bool)]
        metric="observation_aligned_mean_frame_angular_iou"
        unified=chosen.rename(columns={"mean_angular_iou":"angular_iou"})
    unified=unified.assign(catalog=name,metric=metric)[MATCH_COLUMNS]
    summary.update(status="COMPLETED",metric=metric,
        mean_matched_angular_iou=float(unified.angular_iou.mean()) if len(unified) else None)
    return {"event_pair_scores":pairs,"matched_events":chosen,"event_status":status,**extras},unified,summary


def execute_external(config, *, input_run, until, run_dir, visualize=None):
    if until!="validation":
        raise ValueError("Catalog input supports --until validation only")
    if config.vis_mode("validation",visualize)!="off":
        raise NotImplementedError("Use --visualize off for explicit Stage 09; standalone overlays remain available")
    settings,catalogs,date=parameters(config)
    output=Path(run_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    parent,meta,seg_root,frames_file,events,frames,observations=resolve_catalog_parent(
        config,input_run,include_observations=True)
    results={}
    reports={}
    combined=[]
    sources={}
    for name in catalogs:
        paths,reason=references(settings,name,date)
        if paths is None:
            reports[name]={"status":"REFERENCE_UNAVAILABLE","reason":reason,
                           "matched_events":None,"mean_matched_angular_iou":None}
            continue
        sources[name]={key:{"path":str(p),"sha256":_hash_file(p)} for key,p in paths.items()}
        tables,unified,report=evaluate(name,paths,settings,date,frames_file,frames,observations)
        report["references"]=sources[name]
        results[name]=tables
        reports[name]=report
        combined.append(unified)
    # No scientific output is written before all available references validate.
    initialize_run(output,config.data,_revision(),{
        "mode":"explicit_input_run","input_run":str(parent),"input_stage":"catalog",
        "segmentation_ancestor":str(seg_root),"catalog_outputs":meta["outputs"],
        "external_references":sources,"upstream_recomputed":False})
    ctx=_context(output,"validation")
    outputs={}
    if "MVCC" in results:
        obs_path=ctx.tables_dir/"observation_manifest.csv"
        observations.to_csv(obs_path,index=False)
        outputs["observation_manifest"]=obs_path
    for name,tables in results.items():
        folder=ctx.tables_dir/name.lower()
        folder.mkdir(parents=True,exist_ok=False)
        for label,table in tables.items():
            path=folder/(label+".csv")
            table.to_csv(path,index=False)
            outputs[name.lower()+"_"+label]=path
        path=folder/"summary.json"
        path.write_text(json.dumps(reports[name],indent=2,allow_nan=False)+"\n")
        outputs[name.lower()+"_summary"]=path
    merged=pd.concat(combined,ignore_index=True) if combined else pd.DataFrame(columns=MATCH_COLUMNS)
    match_path=ctx.tables_dir/"catalog_matches.csv"
    merged.to_csv(match_path,index=False)
    summary_path=ctx.run_dir/"summary.json"
    summary={"date":date,"input_catalog_run":str(parent),"upstream_recomputed":False,
             "catalogs":reports,"note":"Independent catalog metrics; no pooled mean. Unavailable is not zero."}
    summary_path.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    outputs.update(catalog_matches=match_path,evaluation_metrics=summary_path)
    complete_stage(output,"validation",{"parent_run":str(parent),
        "height_time":meta["outputs"]["height_time"]["path"]},
        _stage_outputs(output,outputs),"external_catalogs:independent_angular_iou",settings)
    return output
