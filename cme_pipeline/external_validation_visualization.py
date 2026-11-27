"""Render Stage 09 reference geometry on saved RD and family masks, never rerun science."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from cme_pipeline.core.run_layout import verify_stage, _hash_file
from cme_pipeline.family_colors import family_color
from cme_pipeline.stages.segmentation.visualization import get_rd_cmap
from scripts.visualize_seeds_overlay import load_seeds, select_nearest_observations
from scripts.visualize_cactus_overlay import load_cactus, active_events
from scripts.visualize_mvcc_overlay import edge_segments


def ancestor(root, stage):
    seen=set()
    root=Path(root).resolve()
    while root not in seen:
        seen.add(root)
        m=json.loads((root/"run_manifest.json").read_text())
        if m["stages"].get(stage)=="COMPLETED":
            return root,verify_stage(root,stage)
        p=m.get("input_provenance") or {}
        if not p.get("input_run"):
            break
        root=Path(p["input_run"]).resolve()
    raise ValueError(f"No verified {stage} ancestor")


def artifact(root,meta,key):
    return root/meta["outputs"][key]["path"]


def load_geometry(seg_root, observations):
    from astropy.io import fits
    from astropy.wcs import WCS
    root,meta=ancestor(seg_root,"preprocessing")
    manifest=pd.read_csv(artifact(root,meta,"manifest"))
    rd=np.load(artifact(root,meta,"rd"),mmap_mode="r")
    valid=np.load(artifact(root,meta,"valid_mask"),mmap_mode="r")
    if rd.shape!=valid.shape or rd.shape!=(len(manifest),256,256):
        raise ValueError("RD geometry mismatch")
    manifest["end_time"]=pd.to_datetime(manifest.end_time,utc=True)
    obs=observations.copy()
    obs["end_time"]=pd.to_datetime(obs.end_time,utc=True)
    if manifest.pair_index.duplicated().any() or not manifest[["pair_index","end_time"]].reset_index(drop=True).equals(obs[["pair_index","end_time"]].reset_index(drop=True)):
        raise ValueError("Preprocessing and catalog observation manifests disagree")
    wcs=WCS(fits.Header.fromstring(artifact(root,meta,"geometry").read_text(),sep="\n"))
    center=np.asarray(wcs.world_to_pixel_values(0.,0.),float)
    n=np.asarray(wcs.world_to_pixel_values(0.,.1))-center
    e=np.asarray(wcs.world_to_pixel_values(-.1,0.))-center
    scales=np.array([s.to_value("arcsec") for s in wcs.proj_plane_pixel_scales()])
    if not np.isfinite([*center,*n,*e,*scales]).all() or not (n[1]>0 and e[0]<0 and abs(n[0])<.01*abs(n[1]) and abs(e[1])<.01*abs(e[0]) and min(scales)>0 and np.isclose(*scales,rtol=1e-4)):
        raise ValueError("Expected isotropic north-up east-left WCS")
    sm=verify_stage(seg_root,"segmentation")
    with np.load(artifact(seg_root,sm,"family_pixel_masks"),allow_pickle=False) as z:
        families={k.removeprefix("mask_"):z[k] for k in z.files if k.startswith("mask_")}
    if any(m.shape!=rd.shape or m.dtype!=np.bool_ for m in families.values()):
        raise ValueError("Invalid saved family masks")
    return manifest,rd,valid,center,families,root


def choose(indices,mode):
    indices=list(indices)
    if mode=="all" or len(indices)<=6:
        return indices
    return [indices[i] for i in np.unique(np.linspace(0,len(indices)-1,6,dtype=int))]


def xy(center,pa,r):
    a=np.deg2rad(pa)
    return center[0]-r*np.sin(a),center[1]+r*np.cos(a)


def draw_sector(ax,center,cpa,width,r,inner=22.,color="magenta"):
    if not np.isfinite([cpa,width,r]).all() or not 0<width<=360:
        raise ValueError("Invalid reference sector")
    r=min(r,120.)
    if r<=inner:
        return
    a=cpa-width/2+np.linspace(0,width,max(24,int(width)+1))
    x,y=xy(center,a,r)
    xx,yy=xy(center,a[::-1],inner)
    ax.fill(np.r_[x,xx],np.r_[y,yy],facecolor=color,alpha=.10)
    ax.plot(np.r_[x,xx,x[0]],np.r_[y,yy,y[0]],color=color,lw=1.2)


def gif(paths,dest):
    images=[]
    try:
        for p in paths:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        if images:
            images[0].save(dest,save_all=True,append_images=images[1:],duration=500,loop=0)
    finally:
        for im in images:
            im.close()


def render(config,seg_root,observations,results,reports,sources,frames_file,destination,mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    destination.mkdir(parents=True,exist_ok=False)
    outputs={}
    record={"mode":mode,"catalogs":{},"read_only_saved_stage":True}
    fig,ax=plt.subplots(figsize=(8,4))
    names=[n for n,r in reports.items() if r.get("mean_matched_angular_iou") is not None]
    if names:
        ax.bar(names,[reports[n]["mean_matched_angular_iou"] for n in names])
    else:
        ax.text(.5,.5,"No matched events",ha="center",transform=ax.transAxes)
    ax.set(ylim=(0,1),ylabel="Matched angular IoU",title="Distinct catalog metrics (not pooled)")
    fig.text(.02,.02,"; ".join(f"{n}: {r['status']}" for n,r in reports.items()),fontsize=8)
    p=destination/"validation_summary.png"
    fig.savefig(p,dpi=120);plt.close(fig)
    outputs["validation_summary_png"]=p
    if results:
        manifest,rd,valid,center,families,pre=load_geometry(seg_root,observations)
        rsun=float(config.data["catalog"]["rsun_px"])
        if not np.isfinite(rsun) or rsun<=0:
            raise ValueError("Invalid catalog solar radius")
        record.update(preprocessing_run=str(pre),segmentation_run=str(seg_root),
                      rsun_px=rsun, radius_policy="same verified catalog.rsun_px used in Stage 08")
        settings=config.data["validation"]
        assume=settings.get("mvcc_assume_ht_rsun_deg",False)
        date=str(config.data["experiment"]["date"])
        stamps=manifest.end_time
        det=pd.read_csv(frames_file)
        for name,tables in results.items():
            paths={}
            for key,item in sources[name].items():
                path=Path(item["path"])
                if _hash_file(path)!=item["sha256"]:
                    raise ValueError(f"Reference changed: {path}")
                paths[key]=path
            active={}
            edges=None
            if name=="SEEDS":
                ref=load_seeds(paths["seeds_frames_csv"])
                tol=float(settings.get("max_time_delta_sec",900))
                # Visualization selects closest reference to each RD; scoring selects
                # closest detection per reference, so the displayed sector is not a frame score.
                for i,t in enumerate(stamps):
                    if t.strftime("%Y%m%d")==date:
                        rows=select_nearest_observations(ref,t,tol)
                        if len(rows):active[i]=rows
            elif name=="CACTus":
                ref=load_cactus(paths["cactus_events_csv"])
                ref=ref[ref.onset_utc.dt.strftime("%Y%m%d").eq(date)]
                for i,t in enumerate(stamps):
                    rows=active_events(ref,t)
                    if len(rows):active[i]=rows
            else:
                ref=pd.read_csv(paths["mvcc_frames_csv"])
                align=tables["observation_alignment"].copy()
                for table in (ref,align):
                    table["timestamp_utc"]=pd.to_datetime(table.timestamp_utc,utc=True)
                align=align[align.status.eq("ALIGNED")].merge(ref[["catalog_event_id","timestamp_utc","front_rsun"]],on=["catalog_event_id","timestamp_utc"],validate="one_to_one")
                if assume:
                    edgepath=paths["mvcc_frames_csv"].with_name("mvcc_leading_edges.csv")
                    edges=pd.read_csv(edgepath)
                    edges["timestamp_utc"]=pd.to_datetime(edges.timestamp_utc,utc=True)
                    record["mvcc_edge_source"]={"path":str(edgepath),"sha256":_hash_file(edgepath),"units_assumed":True}
                for i,pair in enumerate(manifest.pair_index):
                    rows=align[align.pair_index.eq(pair)]
                    if len(rows):active[i]=rows
            indices=choose(sorted(active),mode)
            info={"eligible_frames":len(active),"rendered_frames":len(indices),"status":"RENDERED" if indices else "NO_ACTIVE_REFERENCE"}
            record["catalogs"][name]=info
            if not indices:continue
            folder=destination/name.lower()
            (folder/"frames").mkdir(parents=True)
            rendered=[]
            matches=tables["matched_events"]
            if name=="SEEDS":
                mapping=dict(zip(matches.seeds_event_id,matches.detected_event_id))
            else:
                mapping=dict(zip(matches.catalog_event_id,matches.event_id))
            for i in indices:
                pair=int(manifest.iloc[i].pair_index)
                fig,ax=plt.subplots(figsize=(8,8))
                fig.subplots_adjust(left=.10,right=.95,bottom=.22,top=.91)
                image=np.ma.masked_where(~valid[i].astype(bool)|~np.isfinite(rd[i]),rd[i])
                ax.set_facecolor("black")
                ax.imshow(image,origin="lower",cmap=get_rd_cmap(),vmin=-600,vmax=600,interpolation="nearest")
                for fid,stack in sorted(families.items()):
                    mask=stack[i]
                    if not mask.any():continue
                    rgba=np.zeros((256,256,4),dtype=np.float32)
                    rgba[mask]=to_rgba(family_color(fid),alpha=.30)
                    ax.imshow(rgba,origin="lower",interpolation="nearest")
                    ax.contour(mask.astype(float),levels=[.5],colors=[family_color(fid)],linewidths=.8)
                    yy,xx=np.nonzero(mask)
                    ax.text(xx.mean(),yy.mean(),fid,color=family_color(fid),fontsize=8,bbox={"facecolor":"black","alpha":.5,"edgecolor":"none"})
                lines=[]
                for row in active[i].itertuples(index=False):
                    if name=="SEEDS":
                        ident=row.parent_event_id
                        cpa,width=row.seeds_bpa,row.seeds_width
                        start=getattr(row,"seeds_pa_start",cpa-width/2)
                        end=getattr(row,"seeds_pa_end",cpa+width/2)
                        width=(end-start)%360 or width
                        draw_sector(ax,center,(start+width/2)%360,width,row.seeds_height*rsun)
                        for key,color in (("seeds_lead","yellow"),("seeds_follow","cyan")):
                            h=getattr(row,key,np.nan)*rsun
                            if np.isfinite(h) and 22<h<=120:
                                ax.plot(*xy(center,start+np.linspace(0,width,100),h),color=color,lw=1.5)
                        lines.append(f"{ident} | {mapping.get(ident,'unmatched')} | dt={row.delta_sec:.1f}s")
                    elif name=="CACTus":
                        ident=row.catalog_event_id
                        draw_sector(ax,center,row.cpa_deg,row.angular_width_deg,120.)
                        lines.append(f"{ident} | {mapping.get(ident,'unmatched')} | liftoff sector only")
                    else:
                        ident=row.catalog_event_id
                        draw_sector(ax,center,row.cpa_deg,row.angular_width_deg,row.front_rsun*rsun)
                        if edges is not None:
                            selected=edges[(edges.catalog_event_id==ident)&(edges.timestamp_utc==row.timestamp_utc)]
                            for group in edge_segments(selected,row.cpa_deg):
                                a,h=np.array(group).T
                                r=h*rsun
                                r[(r<22)|(r>120)]=np.nan
                                ax.plot(*xy(center,a,r),color="cyan",lw=2,marker=".",ms=2)
                        event=mapping.get(ident)
                        score=tables["frame_scores"]
                        score=score[(score.catalog_event_id==ident)&(score.event_id==event)&(score.pair_index==pair)]
                        own=det[(det.event_id==event)&(det.pair_index==pair)]
                        v=f"{score.iloc[0].angular_iou:.4f}" if len(score)==1 else "N/A"
                        ours=f"CPA {own.iloc[0].cpa_deg:.1f} W {own.iloc[0].angular_width_deg:.1f}" if len(own)==1 else "not detected"
                        lines.append(f"{ident}: CPA {row.cpa_deg:.1f} W {row.angular_width_deg:.1f} | {event or 'unmatched'}")
                        lines.append(f"Our {ours} | IoU {v} | dt={row.delta_sec:.1f}s")
                note={"SEEDS":"Magenta: height sector; yellow: lead; cyan: follow. Nearest display, not frame score.",
                      "CACTus":"Magenta: angular extent; display radius only, not measured height.",
                      "MVCC":"Magenta: morphology; cyan: HT (assumed Rsun/deg)." if assume else "Magenta: morphology; HT disabled."}[name]
                ax.set(xlim=(-.5,255.5),ylim=(-.5,255.5),aspect="equal",xlabel="X (pixel)",ylabel="Y (pixel)",title=f"{name} | Pair {pair} | {stamps.iloc[i]:%Y-%m-%d %H:%M:%S} UTC")
                fig.text(.02,.19,note,fontsize=8)
                fig.text(.02,.17,"\n".join(lines[:8]),fontsize=7,va="top")
                if len(lines)>8:fig.text(.02,.01,f"+ {len(lines)-8} additional lines; see saved tables",fontsize=7)
                p=folder/"frames"/f"pair_{pair:03d}.png"
                fig.savefig(p,dpi=110);plt.close(fig)
                rendered.append(p)
                outputs[f"{name.lower()}_overlay_{pair}"]=p
            if mode=="all":
                p=folder/f"{name.lower()}_overlay.gif"
                gif(rendered,p)
                outputs[f"{name.lower()}_gif"]=p
    p=destination/"visualization_manifest.json"
    p.write_text(json.dumps(record,indent=2)+"\n")
    outputs["visualization_manifest"]=p
    return outputs
