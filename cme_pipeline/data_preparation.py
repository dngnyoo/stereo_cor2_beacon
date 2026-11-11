"""Input acquisition and FITS inventory for reproducible CME data preparation.

This module deliberately DOES NOT treat raw FITS as equivalent to frozen
north-up COR2 RD DNs. The raw-to-RD calibration/reprojection and causal
C3 detector are separate, not-yet-integrated scientific steps.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote
import json
import re

from astropy.io import fits
from bs4 import BeautifulSoup
import numpy as np

SOURCE_URLS = {
    "beacon_raw": "https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/secchi/img/cor2/",
    "science_raw": "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0/a/img/cor2/",
    "beacon_seq": "https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/secchi/seq/cor2/",
    "science_seq": "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0/a/seq/cor2/",
}
FITS_SUFFIXES = (".fts", ".fts.gz", ".fits", ".fits.gz", ".fit", ".fit.gz")


def validate_date(value: str) -> str:
    if not re.fullmatch(r"\d{8}", str(value)):
        raise ValueError("Date must be YYYYMMDD (eight digits)")
    datetime.strptime(str(value), "%Y%m%d")
    return str(value)


def list_fits(directory: Path, *, recursive: bool = True) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    paths = directory.rglob("*") if recursive else directory.iterdir()
    files = sorted(
        (p for p in paths if p.is_file() and p.name.lower().endswith(FITS_SUFFIXES)),
        key=lambda x: str(x),
    )
    if not files:
        raise ValueError(f"No FITS files found in {directory}")
    return files


def download_cor2(date: str, product: str, output_dir: Path,
                  *, session=None) -> list[Path]:
    """Download one explicit STEREO-A/COR2 product/date without remote code execution.

    URLs are constructed from a fixed allowlist, and filenames are filtered
    to exclude path traversal and arbitrary links. Existing nonempty files
    are kept; network failures raise rather than silently yielding partial runs.
    """
    import requests

    date = validate_date(date)
    if product not in SOURCE_URLS:
        raise ValueError(f"Unknown COR2 product: {product}; choose {sorted(SOURCE_URLS)}")
    client = session if session is not None else requests.Session()
    base_url = f"{SOURCE_URLS[product]}{date}/"
    response = client.get(base_url, timeout=(10, 60))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    names = set()
    for link in soup.find_all("a", href=True):
        candidate = unquote(urlsplit(link["href"]).path).split("/")[-1]
        if candidate and candidate == link["href"] and candidate.lower().endswith(FITS_SUFFIXES):
            if candidate not in {".", ".."} and "/" not in candidate and "\\" not in candidate:
                names.add(candidate)
    if not names:
        raise ValueError(f"No COR2 FITS links found at {base_url}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for name in sorted(names):
        target = output_dir / name
        if not target.is_file() or target.stat().st_size == 0:
            temporary = target.with_name(target.name + ".part")
            try:
                with client.get(urljoin(base_url, name), stream=True, timeout=(10, 120)) as stream:
                    stream.raise_for_status()
                    with temporary.open("wb") as handle:
                        for block in stream.iter_content(chunk_size=1024 * 1024):
                            if block:
                                handle.write(block)
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise ValueError(f"Empty downloaded file: {name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        result.append(target)
    return result


def inspect_fits(paths: list[Path], *, expected_date: str | None = None) -> list[dict]:
    """Read image metadata without interpreting or resampling solar measurements."""
    if expected_date is not None:
        validate_date(expected_date)
    records = []
    for source in paths:
        source = Path(source).resolve()
        with fits.open(source, memmap=False) as hdul:
            image_hdus = [hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim == 2]
            if not image_hdus:
                raise ValueError(f"No 2D FITS image found: {source}")
            image = image_hdus[0]
            hdr = image.header
            shape = tuple(int(d) for d in image.data.shape)
            observed = str(hdr.get("DATE-OBS", hdul[0].header.get("DATE-OBS", "")))
            if not observed:
                raise ValueError(f"Missing DATE-OBS header: {source}")
            observation_date = observed[:10].replace("-", "")
            if expected_date is not None and observation_date != expected_date:
                raise ValueError(
                    f"FITS observation date mismatch: {source.name} has {observed}, "
                    f"expected {expected_date}"
                )
            records.append({
                "file": str(source),
                "date_obs": observed,
                "shape": list(shape),
                "instrument": str(hdr.get("DETECTOR", hdr.get("INSTRUME", ""))),
                "observatory": str(hdr.get("OBSRVTRY", "")),
                "exptime": float(hdr.get("EXPTIME", 0) or 0),
                "polar": str(hdr.get("POLAR", "")),
                "has_wcs": all(key in hdr for key in ("CRPIX1", "CRPIX2", "CDELT1", "CDELT2")),
            })
    records.sort(key=lambda x: (x["date_obs"], x["file"]))
    return records


def prepare_source(*, date: str, source: str, product: str,
                   input_dir: str | Path | None, output_dir: str | Path) -> Path:
    """Prepare a provenance-preserving image inventory, not a detection result."""
    date = validate_date(date)
    output_dir = Path(output_dir).resolve()
    if source == "download":
        if input_dir is not None:
            raise ValueError("Do not pass input_dir for download source")
        files = download_cor2(date, product, output_dir / "raw" / product / date)
    elif source == "local":
        if input_dir is None:
            raise ValueError("local source requires --input-dir")
        files = list_fits(Path(input_dir))
    else:
        raise ValueError("source must be 'local' or 'download'")
    records = inspect_fits(files, expected_date=date)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "date": date,
        "source": source,
        "product": product,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(records),
        "files": records,
        "status": "INVENTORIED_NOT_PREPROCESSED",
        "next_step": "Validated COR2 calibration, north-up RD and C3 detection are required.",
    }, indent=2), encoding="utf-8")
    return manifest_path
