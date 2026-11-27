"""Opt-in preparation of official external references, only after Stage 08."""
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
import pandas as pd

KEYS = {"SEEDS": ("seeds_frames_csv",), "CACTus": ("cactus_events_csv",),
        "MVCC": ("mvcc_frames_csv", "mvcc_events_csv")}


def acquisition_settings(settings):
    value=settings.get("acquisition",{})
    if not isinstance(value,dict) or set(value)-{"mode","reuse_existing","cache_root"}:
        raise ValueError("Invalid validation.acquisition mapping")
    mode=value.get("mode","explicit")
    if mode not in ("explicit","auto"):
        raise ValueError("Acquisition mode must be explicit or auto")
    if value.get("reuse_existing",True) is not True:
        raise ValueError("Only non-destructive reuse_existing: true is supported")
    root=value.get("cache_root","data/catalogs")
    if not isinstance(root,str) or not root.strip():
        raise ValueError("Invalid cache_root")
    return mode,Path(root).expanduser().resolve()


def prepare(name,date,settings):
    """Return effective paths and acquisition status; never overwrite normalized CSVs."""
    mode,root=acquisition_settings(settings)
    if mode!="auto":
        return dict(settings),{"status":"EXPLICIT_ONLY"}
    if name=="MVCC" and not "200703"<=date[:6]<="201409":
        return dict(settings),{"status":"OUTSIDE_COVERAGE"}
    keys=KEYS[name]
    supplied=[k for k in keys if k in settings]
    if supplied:
        if len(supplied)!=len(keys):
            raise ValueError(f"Provide all or none of {keys}")
        # Explicit locations remain authoritative; no silent replacement.
        return dict(settings),{"status":"EXPLICIT_PATHS"}
    effective=dict(settings)
    if name=="SEEDS":
        folder=root/"seeds"/date
        paths={keys[0]:folder/"seeds_frames.csv"}
    elif name=="CACTus":
        folder=root/"cactus"/"normalized"/date
        paths={keys[0]:folder/"cactus_events.csv"}
    else:
        folder=root/"mvcc"/"normalized"/date
        paths={keys[0]:folder/"mvcc_frames.csv",keys[1]:folder/"mvcc_events.csv"}
    effective.update({k:str(p) for k,p in paths.items()})
    if all(p.is_file() for p in paths.values()):
        return effective,{"status":"REUSED_NORMALIZED","paths":{k:str(p) for k,p in paths.items()}}
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f"Incomplete normalized cache: {folder}; inspect before retry")
    try:
        if name=="SEEDS":
            from scripts.fetch_seeds_monthly import acquire
            from scripts.fetch_seeds_details import fetch_details
            from scripts.parse_seeds_monthly import parse_monthly
            from scripts.parse_seeds_details import parse_directory, EVENT_KEYS, FRAME_KEYS
            raw=root/"seeds"/"raw"
            details=root/"seeds"/"details"
            month=raw/date[:4]/date[4:6]/"monthly.txt"
            if not month.is_file():
                acquire(date,output_root=raw)
            expected=parse_monthly(month.read_text(encoding="utf-8-sig"),date=date)
            if expected.empty:
                events=pd.DataFrame(columns=["event_id",*EVENT_KEYS,"source_filename","provenance"])
                frames=pd.DataFrame(columns=[*FRAME_KEYS,"parent_event_id","source_filename"])
            else:
                fetch_details(date,raw_root=raw,output_root=details)
                events,frames=parse_directory(details/date,date)
                if frames.empty or set(events.event_id)!=set(expected.event_id):
                    raise ValueError("Incomplete SEEDS details")
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/"seeds_events.csv").open("x") as f:
                events.to_csv(f,index=False)
            with paths[keys[0]].open("x") as f:
                frames.to_csv(f,index=False)
        elif name=="CACTus":
            from scripts.fetch_cactus_monthly import acquire
            from scripts.parse_cactus_monthly import main as parse
            base=root/"cactus"
            raw=base/"raw"/date[:4]/date[4:6]/"cmecat.txt"
            acquire(date,output_root=base/"raw",reuse_existing=raw.is_file())
            parse(["--date",date,"--output-root",str(base),"--day-only"])
        else:
            from scripts.fetch_mvcc import main as fetch
            from scripts.parse_mvcc import main as parse
            raw=root/"mvcc"/"raw"/date[:6]
            fetch(["--date",date,"--raw-dir",str(raw),"--reuse-existing"])
            parse(["--date",date,"--raw-dir",str(raw),"--output-dir",str(folder)])
    except HTTPError as exc:
        # A missing server asset is distinct from an empty valid catalog.
        if exc.code in (404,410):
            return effective,{"status":"SOURCE_UNAVAILABLE","reason":f"HTTP {exc.code}"}
        return effective,{"status":"ACQUISITION_FAILED","reason":f"HTTP {exc.code}"}
    except (URLError,TimeoutError,ConnectionError) as exc:
        return effective,{"status":"ACQUISITION_FAILED","reason":str(exc)}
    if not all(p.is_file() for p in paths.values()):
        raise ValueError("Acquisition produced incomplete normalized output")
    return effective,{"status":"PREPARED","paths":{k:str(p) for k,p in paths.items()}}
