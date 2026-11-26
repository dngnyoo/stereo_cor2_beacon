"""Fetch official MVCC monthly JSON and daily COR2-A CORSET details."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from scripts.parse_mvcc import daily_events, detail_id, safe_name, parse_date, parse_morphology, parse_edges

MONTHLY = "https://solar.jhuapl.edu/Data-Products/dataProcessing.php"
BASE = "http://sd-www.jhuapl.edu/COR2CMECatalog/catalog"


def fetch(url, path, *, body=None, reuse=False):
    if path.exists():
        if not reuse:
            raise FileExistsError(f"{path}; use --reuse-existing")
        return dict(path=path.name, requested_url=url, resolved_url=None,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), reused=True)
    with urlopen(Request(url, data=body, headers={"User-Agent": "COR2 research catalog client"}), timeout=60) as response:
        data = response.read()
        final_url = response.geturl()
    if not data or data.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise ValueError(f"Unexpected response: {url}")
    with path.open("xb") as handle:
        handle.write(data)
    return dict(path=path.name, requested_url=url, resolved_url=final_url,
                sha256=hashlib.sha256(data).hexdigest(), reused=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args(argv)
    parse_date(args.date)
    raw = args.raw_dir or Path("data/catalogs/mvcc/raw")/args.date[:6]
    raw.mkdir(parents=True, exist_ok=True)
    month = args.date[:4]+"-"+args.date[4:6]
    manifest = [fetch(MONTHLY, raw/"monthly.json",
                      body=urlencode({"yearMonth":month}).encode(), reuse=args.reuse_existing)]
    rows = daily_events(json.loads((raw/"monthly.json").read_text()), args.date)
    seen = set()
    for row in rows:
        if row["measurement_status"] != "CORSET":
            continue
        ident = detail_id(row)
        folder = PurePosixPath(row["dataPath"])
        if folder.is_absolute() or ".." in folder.parts or folder.parts != (
            "A", args.date[:4], args.date[4:6], args.date[2:], ident
        ):
            raise ValueError(f"Unexpected CORSET directory: {folder}")
        txt = safe_name(row["txtFile"])
        targets = [(BASE+"/indivFiles/"+quote(txt), txt)]
        targets += [(BASE+"/"+str(folder)+"/"+quote(ident+suffix), ident+suffix)
                    for suffix in (".corset.ht", ".info")]
        for url, name in targets:
            if name not in seen:
                manifest.append(fetch(url, raw/name, reuse=args.reuse_existing))
                seen.add(name)
        parse_morphology(raw/txt, row)
        parse_edges(raw/(ident+".corset.ht"), row)
    report = {"date":args.date, "monthly_post": {"yearMonth":month},
              "files":manifest, "catalog_events":len(rows)}
    target = raw/("fetch_"+args.date+".json")
    with target.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
