"""Post-hoc MVCC overlay using saved evaluation alignment; never recompute matching."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from scripts.visualize_seeds_overlay import (
    FILE_PATTERN, axes_bounds, load_target_geometry, nearest_fits_radius,
    png_xy, sector_points,
)

MAGENTA = (255, 85, 245, 220)
CYAN = (70, 220, 255, 255)


def edge_segments(rows, cpa, *, max_gap_deg=1.5):
    """Order around CPA, preserve gaps, reject ambiguous duplicate PA."""
    points = []
    for row in rows.itertuples(index=False):
        pa, h = float(row.position_angle), float(row.leading_edge_height)
        if not math.isfinite(pa) or not math.isfinite(h) or h <= 0 or not 0 <= pa <= 360:
            raise ValueError("Invalid leading edge geometry")
        points.append(((pa-cpa+180)%360-180, h))
    points.sort()
    groups = []
    previous = None
    for offset, h in points:
        if previous is not None and abs(offset-previous)<1e-9:
            raise ValueError("Duplicate leading edge PA")
        if previous is None or offset-previous > max_gap_deg:
            groups.append([])
        groups[-1].append((cpa+offset,h))
        previous = offset
    return groups


def overlay_frame(source, destination, active, edges, detections, matches, scores, *,
                  sx, sy, rsun_px, inner_px=22., outer_px=120., assume_ht_rsun=False):
    if source.resolve() == destination.resolve():
        raise ValueError("Output cannot overwrite source")
    with Image.open(source) as original:
        base = original.convert("RGBA")
    bounds = axes_bounds(base.size)
    layer = Image.new("RGBA", base.size)
    painter = ImageDraw.Draw(layer)
    lines = []
    for row in active.itertuples(index=False):
        cpa, width, height = float(row.cpa_deg), float(row.angular_width_deg), float(row.front_rsun)
        if not np.isfinite([cpa,width,height,rsun_px]).all() or not (0<width<=360 and height>0 and rsun_px>0):
            raise ValueError("Invalid morphology geometry")
        radius = min(height*rsun_px, outer_px)
        if radius > inner_px:
            vertices = sector_points(cpa,width,inner_px,radius,sx,sy,bounds)
            painter.polygon(vertices,fill=(224,64,251,24))
            painter.line(vertices+[vertices[0]],fill=MAGENTA,width=2)
        if assume_ht_rsun:
            selected=edges[(edges.catalog_event_id==row.catalog_event_id)&
                           (edges.timestamp_utc==row.timestamp_utc)]
            for group in edge_segments(selected,cpa):
                # Never project off-FOV points onto a fictitious edge.
                previous=None
                for pa,h in group:
                    r=h*rsun_px
                    if not inner_px <= r <= outer_px:
                        previous=None
                        continue
                    a=math.radians(pa)
                    point=png_xy(sx-r*math.sin(a),sy+r*math.cos(a),bounds)
                    if previous is not None:
                        painter.line([previous,point],fill=CYAN,width=3)
                    else:
                        x,y=point
                        painter.ellipse((x-1,y-1,x+1,y+1),fill=CYAN)
                    previous=point
        family=matches.get(row.catalog_event_id)
        d=detections[(detections.event_id==family)&(detections.pair_index==row.pair_index)]
        s=scores[(scores.catalog_event_id==row.catalog_event_id)&
                 (scores.event_id==family)&(scores.pair_index==row.pair_index)]
        own=(f"CPA {d.iloc[0].cpa_deg:.1f} / W {d.iloc[0].angular_width_deg:.1f}"
             if len(d)==1 else "not detected")
        iou=f"{s.iloc[0].angular_iou:.4f}" if len(s)==1 else "N/A"
        lines.append(f"MVCC {row.catalog_event_id}: CPA {cpa:.1f} / W {width:.1f} / H {height:.2f} Rsun")
        lines.append(f"Our {family or 'unmatched'}: {own} | IoU {iou} | dt {row.delta_sec:.1f}s")
    combined=Image.alpha_composite(base,layer).convert("RGB")
    # Append footer rather than covering existing legends.
    footer=["MVCC morphology = magenta; geometric sector, not a pixel mask",
            "Cyan = HT leading edge (ASSUMED Rsun / degrees)" if assume_ht_rsun
            else "HT leading edge disabled: units not independently verified"]
    footer += lines or ["No aligned MVCC observation"]
    result=Image.new("RGB",(combined.width,combined.height+18*len(footer)+12),(20,20,20))
    result.paste(combined,(0,0))
    draw=ImageDraw.Draw(result)
    for i,line in enumerate(footer):
        draw.text((14,combined.height+6+18*i),line,fill=(245,245,245))
    destination.parent.mkdir(parents=True,exist_ok=True)
    result.save(destination)
    for im in (base,layer,combined,result):
        im.close()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ("run-dir","reference-dir","evaluation-dir","raw-fits-root","output-dir"):
        ap.add_argument("--"+arg,type=Path,required=True)
    ap.add_argument("--assume-ht-rsun-deg",action="store_true")
    ap.add_argument("--gif",action="store_true")
    a=ap.parse_args(argv)
    ref=a.reference_dir
    ev=a.evaluation_dir
    frames_path=ref/"mvcc_frames.csv"
    det_path=a.run_dir/"tables/cme_frame_measurements.csv"
    report=json.loads((ev/"summary.json").read_text())
    for path in (frames_path,det_path):
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if report["inputs"].get(str(path.resolve()))!=actual:
            raise ValueError(f"Evaluation input mismatch: {path}")
    frames=pd.read_csv(frames_path)
    edges=pd.read_csv(ref/"mvcc_leading_edges.csv")
    alignment=pd.read_csv(ev/"observation_alignment.csv")
    det=pd.read_csv(det_path)
    scores=pd.read_csv(ev/"frame_scores.csv")
    matched=pd.read_csv(ev/"matched_events.csv")
    matches=dict(zip(matched.catalog_event_id,matched.event_id))
    for table in (frames,edges,alignment):
        table["timestamp_utc"]=pd.to_datetime(table.timestamp_utc,utc=True)
    active=alignment[alignment.status.eq("ALIGNED")].merge(
        frames[["catalog_event_id","timestamp_utc","front_rsun"]],
        on=["catalog_event_id","timestamp_utc"],validate="one_to_one")
    active["beacon_timestamp"]=pd.to_datetime(active.beacon_timestamp,utc=True)
    if active.empty:
        raise ValueError("No aligned MVCC frames")
    sources={}
    for path in (a.run_dir/"visualizations/segmentation_overlays").glob("pair_*_families.png"):
        m=FILE_PATTERN.fullmatch(path.name)
        if m:
            idx=int(m.group(1))
            if idx in sources:
                raise ValueError("Duplicate PNG pair index")
            sources[idx]=(path,pd.to_datetime(m.group(2),format="%Y%m%dT%H%M%SZ",utc=True))
    sx,sy,scale=load_target_geometry(a.run_dir)
    prepared=[]
    for idx,rows in active.groupby("pair_index",sort=True):
        if idx not in sources:
            raise FileNotFoundError(f"No segmentation PNG for aligned pair {idx}")
        source,stamp=sources[idx]
        if ((rows.beacon_timestamp-stamp).abs().dt.total_seconds()>1).any():
            raise ValueError(f"PNG timestamp mismatch: {source}")
        with Image.open(source) as im:
            axes_bounds(im.size)
        radius,fits=nearest_fits_radius(stamp,a.raw_fits_root,scale,max_delta_sec=60.)
        prepared.append((source,rows,radius,fits))
    a.output_dir.mkdir(parents=True,exist_ok=False)
    outputs=[]
    for source,rows,radius,fits in prepared:
        target=a.output_dir/"frames"/source.name.replace("_families.png","_mvcc.png")
        overlay_frame(source,target,rows,edges,det,matches,scores,sx=sx,sy=sy,
                      rsun_px=radius,assume_ht_rsun=a.assume_ht_rsun_deg)
        outputs.append(target)
        print(f"{target.name}: {len(rows)} aligned MVCC observation(s)")
    if a.gif:
        images=[]
        try:
            for path in outputs:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
            height=max(im.height for im in images)
            for i,im in enumerate(images):
                if im.height!=height:
                    padded=Image.new("RGB",(im.width,height),(20,20,20))
                    padded.paste(im,(0,0))
                    im.close()
                    images[i]=padded
            images[0].save(a.output_dir/"mvcc_overlay.gif",save_all=True,
                           append_images=images[1:],duration=500,loop=0)
        finally:
            for im in images:
                im.close()
    provenance={"evaluation_dir":str(ev.resolve()),"ht_units_assumed":a.assume_ht_rsun_deg,
                "frames":[{"source":str(src),"fits":str(fits),"rsun_px":r}
                          for src,rows,r,fits in prepared]}
    (a.output_dir/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__=="__main__":
    main()
