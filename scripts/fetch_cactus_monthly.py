"""Download official NRL CACTus COR2-A monthly TXT, retaining original bytes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from scripts.parse_cactus_monthly import parse_date, parse_monthly

NRL_BASE = "https://secchi.nrl.navy.mil/cactus/SECCHI-A"
MAX_BYTES = 25 * 1024 * 1024


def month_url(date: str, base: str = NRL_BASE) -> str:
    dt = parse_date(date)
    return f"{base.rstrip('/')}/{dt:%Y/%m}/out/cmecat.txt"


def fetch_asset(url: str, *, timeout: float = 30) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": "StereoCOR2-CACTus-research/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return response.read(MAX_BYTES + 1), response.geturl()


def acquire(date: str, *, output_root: Path, base: str = NRL_BASE,
            timeout: float = 30, force: bool = False, reuse_existing: bool = False,
            fetch=fetch_asset) -> dict:
    dt = parse_date(date)
    dest = Path(output_root) / dt.strftime("%Y/%m")
    raw = dest / "cmecat.txt"
    manifest = dest / "source_provenance.json"
    if force and reuse_existing:
        raise ValueError("Choose either force or reuse_existing")
    url = month_url(date, base)
    if reuse_existing:
        blob = raw.read_bytes()
        resolved = None
        mode = "existing_local_file"
    else:
        if not force and (raw.exists() or manifest.exists()):
            raise FileExistsError(f"CACTus assets already exist: {dest}; use --reuse-existing or --force")
        blob, resolved = fetch(url, timeout=timeout)
        mode = "download"
    if not blob.strip() or len(blob) > MAX_BYTES:
        raise ValueError("Empty or oversized CACTus asset")
    parse_monthly(blob.decode("utf-8-sig"), dt.strftime("%Y%m"))
    digest = hashlib.sha256(blob).hexdigest()
    if reuse_existing and manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous["files"]["cmecat.txt"]["sha256"] != digest:
            raise ValueError("Existing CACTus raw checksum differs from provenance")
        return {"directory": str(dest), "provenance": str(manifest)}
    metadata = {
        "source": "NRL CACTus STEREO-A COR2",
        "archive_month": dt.strftime("%Y-%m"),
        "requested_date": date,
        "acquisition_mode": mode,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {"cmecat.txt": {
            "requested_url": url, "resolved_url": resolved,
            "bytes": len(blob), "sha256": digest,
        }},
        "note": "For reused files the original retrieval URL/time is not independently verified.",
    }
    dest.mkdir(parents=True, exist_ok=True)
    if not reuse_existing:
        raw.write_bytes(blob)
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"directory": str(dest), "provenance": str(manifest)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--output-root", type=Path, default=Path("data/catalogs/cactus/raw"))
    ap.add_argument("--base-url", default=NRL_BASE)
    ap.add_argument("--timeout", type=float, default=30)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--force", action="store_true")
    group.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args(argv)
    if args.timeout <= 0:
        ap.error("--timeout must be positive")
    result = acquire(args.date, output_root=args.output_root, base=args.base_url,
                     timeout=args.timeout, force=args.force, reuse_existing=args.reuse_existing)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
