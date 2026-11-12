"""Input acquisition and FITS inventory for reproducible CME data preparation.

This module deliberately DOES NOT treat raw FITS as equivalent to frozen
north-up COR2 RD DNs. The raw-to-RD calibration/reprojection and causal
C3 detector are separate, not-yet-integrated scientific steps.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote
import hashlib
import json
import re
import time

from astropy.io import fits
from bs4 import BeautifulSoup
import numpy as np

D7_SOURCE_URL = "https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/secchi/img/cor2/"

SOURCE_URLS = {
    "d7": D7_SOURCE_URL,
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
    output_dir = Path(output_dir)
    completion = output_dir / ".download_complete.json"
    # A completed day/source listing is authoritative for offline replay.
    # Verify SHA-256 before reusing; a missing or changed file invalidates
    # the marker and triggers an incremental refresh from the archive.
    stale_names = set()
    if completion.is_file():
        try:
            recorded = json.loads(completion.read_text(encoding="utf-8"))
            if (recorded.get("date") == date and recorded.get("product") == product
                    and recorded.get("files")):
                verified = []
                for item in recorded["files"]:
                    name = item["name"]
                    if Path(name).name != name or not name.lower().endswith(FITS_SUFFIXES):
                        raise ValueError("Invalid raw cache filename")
                    target = output_dir / name
                    if not target.is_file() or target.stat().st_size != item["size"]:
                        stale_names.add(name)
                        raise ValueError("Missing or changed raw cache file")
                    digest = hashlib.sha256()
                    with target.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != item["sha256"]:
                        stale_names.add(name)
                        raise ValueError("Raw cache checksum mismatch")
                    verified.append(target)
                print(f"Reused cached {product} {date}: {len(verified)} FITS", flush=True)
                return verified
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            print(f"Raw cache incomplete/changed for {product} {date}; refreshing", flush=True)
    client = session if session is not None else requests.Session()
    base_url = f"{SOURCE_URLS[product]}{date}/"
    # NASA may occasionally close connections without sending a response.
    # Retry transient failures without masking permanent HTTP errors.
    def request_with_retry(url: str, *, stream: bool = False):
        for attempt in range(5):
            try:
                response = client.get(
                    url, stream=stream,
                    timeout=(15, 120) if stream else (15, 60),
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                transient = not isinstance(exc, requests.HTTPError) or (
                    exc.response is not None
                    and exc.response.status_code in (408, 429, 500, 502, 503, 504)
                )
                if not transient or attempt == 4:
                    raise
                delay = min(2 ** attempt, 8)
                print(f"Transient NASA request error ({exc}); retrying in {delay}s "
                      f"({attempt + 1}/4): {url}", flush=True)
                time.sleep(delay)

    with request_with_retry(base_url) as response:
        soup = BeautifulSoup(response.text, "html.parser")
    names = set()
    for link in soup.find_all("a", href=True):
        candidate = unquote(urlsplit(link["href"]).path).split("/")[-1]
        if candidate and candidate == link["href"] and candidate.lower().endswith(FITS_SUFFIXES):
            if product == "d7" and "_d7c2" not in candidate.lower():
                continue
            if candidate not in {".", ".."} and "/" not in candidate and "\\" not in candidate:
                names.add(candidate)
    if not names:
        raise ValueError(f"No COR2 FITS links found at {base_url}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for name in sorted(names):
        target = output_dir / name
        if name in stale_names or not target.is_file() or target.stat().st_size == 0:
            temporary = target.with_name(target.name + ".part")
            file_url = urljoin(base_url, name)
            for attempt in range(5):
                try:
                    # Always restart failed streams from byte zero; never
                    # promote an incomplete .part file to a valid FITS.
                    with request_with_retry(file_url, stream=True) as stream:
                        with temporary.open("wb") as handle:
                            for block in stream.iter_content(chunk_size=1024 * 1024):
                                if block:
                                    handle.write(block)
                    if not temporary.is_file() or temporary.stat().st_size == 0:
                        raise IOError(f"Empty downloaded file: {name}")
                    temporary.replace(target)
                    print(f"Downloaded {name}", flush=True)
                    break
                except (requests.RequestException, OSError) as exc:
                    if attempt == 4:
                        raise
                    delay = min(2 ** attempt, 8)
                    print(f"Interrupted FITS download ({exc}); retrying in {delay}s "
                          f"({attempt + 1}/4): {name}", flush=True)
                    time.sleep(delay)
                finally:
                    temporary.unlink(missing_ok=True)
        result.append(target)
    # Commit the day-level inventory only after every expected FITS succeeds.
    # The atomic marker makes complete dates reusable without network access.
    inventory = []
    for target in result:
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        inventory.append({
            "name": target.name,
            "size": target.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    marker = completion.with_name(completion.name + ".tmp")
    marker.write_text(json.dumps({
        "schema_version": 1,
        "date": date,
        "product": product,
        "files": inventory,
    }, indent=2) + "\n", encoding="utf-8")
    marker.replace(completion)
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
                   input_dir: str | Path | None, output_dir: str | Path,
                   phase_sec: float | None = None,
                   warmup_days: int = 0,
                   frozen_reference_pairs: bool = False,
                   raw_root: str | Path | None = None) -> Path:
    """Acquire the analysis date and optional preceding UTC warmup days."""
    date = validate_date(date)
    if not 0 <= warmup_days <= 7:
        raise ValueError("warmup_days must be between 0 and 7")
    target_date = datetime.strptime(date, "%Y%m%d").date()
    requested_days = [(target_date - timedelta(days=i)).strftime("%Y%m%d")
                      for i in range(warmup_days, -1, -1)]
    output_dir = Path(output_dir).resolve()
    if source == "download":
        if input_dir is not None:
            raise ValueError("Do not pass input_dir for download source")
        files = []
        if product == "d7":
            # General, date-independent selection: download both complete
            # archives, retain provenance, and choose Science-first d7.
            from cme_pipeline.dual_archive import collect_dual_archive
            files, raw_inventory_path, selected_sources_path = collect_dual_archive(
                requested_days, output_dir,
                raw_root=Path(raw_root) if raw_root is not None else None,
            )
        else:
            for observed_date in requested_days:
                files.extend(download_cor2(
                    observed_date, product, output_dir / "raw" / product / observed_date
                ))
    elif source == "local":
        if input_dir is None:
            raise ValueError("local source requires --input-dir")
        files = list_fits(Path(input_dir))
        if product == "d7":
            # Reusing a dual-archive raw directory can expose science and
            # beacon copies of the same frame. Match download's science-first
            # policy without changing or deleting any downloaded FITS.
            by_name = {}
            for file in files:
                if "_d7c2" not in file.name.lower():
                    continue
                parts = {part.lower() for part in file.parts}
                rank = 0 if "science" in parts else (1 if "beacon" in parts else 2)
                previous = by_name.get(file.name)
                if previous is None or (rank, str(file)) < previous[0]:
                    by_name[file.name] = ((rank, str(file)), file)
            files = sorted((item[1] for item in by_name.values()), key=lambda p: p.name)
    else:
        raise ValueError("source must be 'local' or 'download'")
    if product == "d7":
        from cme_pipeline.cadence_selection import is_d7, select_d7_pairs, DEFAULT_PHASE_SEC
        files = [p for p in files if is_d7(p.name)]
        if not files:
            raise ValueError("No STEREO-A COR2 d7 (_d7c2) FITS files found")
    records = inspect_fits(files, expected_date=date if warmup_days == 0 else None)
    permitted = set(requested_days)
    records = [r for r in records if r["date_obs"][:10].replace("-", "") in permitted]
    if not any(r["date_obs"][:10].replace("-", "") == date for r in records):
        raise ValueError(f"No observations on requested analysis date {date}")
    cadence = None
    if frozen_reference_pairs:
        if date != "20101212" or product != "d7":
            raise ValueError("Frozen reference pair selection is 20101212 d7 only")
        import csv
        from cme_pipeline.cadence_selection import parse_timestamp, FILENAME_TIME_RE
        reference = Path(__file__).resolve().parents[1] / (
            "validation_20101212/experiments/01_d7_baseline_segmentation/"
            "results/00_manifest/01_rd_manifest.csv"
        )
        with reference.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 60:
            raise ValueError("Frozen reference must contain 60 pairs")

        # Frozen mode bypasses the modern phase-selected frame list. The
        # historical source manifest, not a phase heuristic, defines the
        # exact FITS frames to calibrate and pair.
        file_lookup = {}
        for record in records:
            name = Path(record["file"]).name
            if name in file_lookup:
                raise ValueError(f"Duplicate FITS filename in source: {name}")
            file_lookup[name] = record
        exact_pairs, selected_names = [], set()
        for item in rows:
            first, second = item["prev_file"], item["curr_file"]
            absent = [name for name in (first, second) if name not in file_lookup]
            if absent:
                raise ValueError(
                    f"Frozen reference FITS genuinely absent from downloaded inventory: {absent}. "
                    f"Check the raw/d7 daily directories and source URL."
                )
            start, end = (parse_timestamp(file_lookup[name]["date_obs"])
                          for name in (first, second))
            for name, stamp in ((first, start), (second, end)):
                match = FILENAME_TIME_RE.search(name)
                if match is None:
                    raise ValueError(f"Missing timestamp in FITS filename: {name}")
                named = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
                if abs((named - stamp).total_seconds()) > 1.:
                    raise ValueError(f"Frozen FITS filename/DATE-OBS mismatch: {name}")
            dt = (end - start).total_seconds()
            if abs(dt - 1800.) > 2.:
                raise ValueError(
                    f"Frozen pair not separated by 30 minutes: {first}, {second}: {dt}s"
                )
            exact_pairs.append({
                "pair_index": len(exact_pairs) + 1,
                "start_file": file_lookup[first]["file"],
                "end_file": file_lookup[second]["file"],
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "delta_t_sec": dt, "delta_t_min": dt / 60.,
            })
            selected_names.update((first, second))
        selected_frames = [
            {"file": rec["file"], "date_obs": rec["date_obs"]}
            for rec in records if Path(rec["file"]).name in selected_names
        ]
        cadence = {
            "mode": "frozen_reference_pairs",
            "phase_sec": phase_sec,
            "cadence_sec": 1800.,
            "selected_frames": selected_frames,
            "pairs": exact_pairs,
            "gaps": [],
            "n_input": len(records),
            "n_selected": len(selected_frames),
            "n_pairs": len(exact_pairs),
            "n_gaps": 0,
            "reference_pair_manifest": str(reference),
        }
    elif product == "d7":
        from cme_pipeline.cadence_selection import select_d7_pairs, infer_d7_phase
        selected_phase = infer_d7_phase(records) if phase_sec is None else phase_sec
        cadence = select_d7_pairs(records, phase_sec=selected_phase)
        cadence["phase_source"] = "inferred_from_DATE_OBS" if phase_sec is None else "explicit"
    elif phase_sec is not None:
        raise ValueError("--phase-sec is only supported with --product d7")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "input_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "date": date,
        "warmup_days_requested": warmup_days,
        "frozen_reference_pairs": frozen_reference_pairs,
        "included_dates": requested_days,
        "source": source,
        "product": product,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_files": len(records),
        "files": records,
        "cadence_selection": cadence,
        "raw_inventory_manifest": str(raw_inventory_path) if source == "download" and product == "d7" else None,
        "raw_data_root": str(Path(raw_root).resolve()) if source == "download" and raw_root is not None else None,
        "selected_d7_sources_manifest": str(selected_sources_path) if source == "download" and product == "d7" else None,
        "status": "INVENTORIED_NOT_PREPROCESSED",
        "next_step": "Validated COR2 calibration, north-up RD and C3 detection are required.",
    }, indent=2), encoding="utf-8")
    return manifest_path
