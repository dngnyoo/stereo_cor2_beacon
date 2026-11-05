#!/usr/bin/env python3
"""NASA STEREO-A COR2 Data Downloader.

Downloads raw Beacon (d7) and Science (d4) FITS files directly from the
NASA STEREO Science Center (SSC) public HTTP archive.

Examples:
  # Download Beacon data for a single date:
  python download_cor2.py --date 20240510 --type beacon

  # Download Beacon data for a date range:
  python download_cor2.py --start-date 20240501 --end-date 20240505 --type beacon

  # Download Science (d4) data to custom directory:
  python download_cor2.py --date 20240510 --type science --output-dir ./data/science_20240510
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# NASA STEREO Science Center archive endpoints
URL_MAP = {
    "beacon": "https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/secchi/img/cor2/",
    "science": "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0/a/img/cor2/",
}


def download_day(
    date_str: str,
    data_type: str = "beacon",
    output_dir: Path | str = "./data",
    decompress_gzip: bool = True,
    timeout: int = 15,
) -> list[Path]:
    """Downloads all COR2 FITS files for a specific date from NASA SSC."""
    base_url = f"{URL_MAP[data_type]}{date_str}/"
    target_dir = Path(output_dir) / date_str
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{data_type.upper()}] Checking archive at: {base_url}")
    try:
        resp = requests.get(base_url, timeout=timeout)
        if resp.status_code != 200:
            print(f"  Warning: Directory not found on NASA server (HTTP {resp.status_code}). Skipping.")
            return []
    except Exception as e:
        print(f"  Connection error querying {base_url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    # Identify FITS files (.fts, .fits, .fts.gz, .fits.gz)
    links = [
        a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].endswith((".fts", ".fits", ".fts.gz", ".fits.gz"))
    ]

    # Filter for d7 (beacon-style) if beacon requested
    if data_type == "beacon":
        links = [f for f in links if "_d7c2" in f.lower() or f.endswith((".fts", ".fts.gz"))]

    if not links:
        print(f"  No matching FITS files found for {date_str}.")
        return []

    print(f"  Found {len(links)} observations for {date_str}. Downloading to: {target_dir}")
    downloaded: list[Path] = []

    for idx, fname in enumerate(links, start=1):
        file_url = urljoin(base_url, fname)
        save_path = target_dir / fname

        # Final destination if decompressed
        dest_path = save_path
        if decompress_gzip and fname.endswith(".gz"):
            dest_path = target_dir / fname[:-3]

        if dest_path.is_file() and dest_path.stat().st_size > 0:
            print(f"  [{idx:02d}/{len(links):02d}] Already exists: {dest_path.name}")
            downloaded.append(dest_path)
            continue

        print(f"  [{idx:02d}/{len(links):02d}] Downloading: {fname} ...", end=" ", flush=True)
        try:
            with requests.get(file_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)

            # Automatically decompress .gz if requested
            if decompress_gzip and fname.endswith(".gz"):
                with gzip.open(save_path, "rb") as f_in, open(dest_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                save_path.unlink()  # remove .gz archive

            size_mb = dest_path.stat().st_size / 1e6
            print(f"Done ({size_mb:.2f} MB)")
            downloaded.append(dest_path)
        except Exception as e:
            print(f"Failed ({e})")
            if save_path.exists():
                save_path.unlink()

    return downloaded


def parse_args():
    parser = argparse.ArgumentParser(description="Download STEREO-A COR2 observation data from NASA SSC.")
    parser.add_argument("--date", type=str, help="Single observation date (YYYYMMDD, e.g. 20240510)")
    parser.add_argument("--start-date", type=str, help="Start date for range download (YYYYMMDD)")
    parser.add_argument("--end-date", type=str, help="End date for range download (YYYYMMDD)")
    parser.add_argument(
        "--type",
        choices=["beacon", "science"],
        default="beacon",
        help="Data product: 'beacon' (d7 30-min cadence) or 'science' (d4 high-res)",
    )
    parser.add_argument("--output-dir", type=str, default="./raw_data", help="Local directory to store raw FITS files")
    parser.add_argument("--keep-gzip", action="store_true", help="Keep raw .gz compressed files without decompressing")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.date and not (args.start_date and args.end_date):
        print("Error: Specify either --date YYYYMMDD or both --start-date and --end-date.")
        sys.exit(1)

    dates: list[str] = []
    if args.date:
        dates.append(args.date.strip())
    else:
        d_start = datetime.strptime(args.start_date.strip(), "%Y%m%d")
        d_end = datetime.strptime(args.end_date.strip(), "%Y%m%d")
        curr = d_start
        while curr <= d_end:
            dates.append(curr.strftime("%Y%m%d"))
            curr += timedelta(days=1)

    print("=" * 78)
    print(f"  STEREO-A COR2 NASA Downloader ({args.type.upper()})")
    print(f"  Target Date(s) : {dates[0]} to {dates[-1]} ({len(dates)} day(s))")
    print(f"  Save Directory : {args.output_dir}")
    print("=" * 78)

    total_files = 0
    for d in dates:
        files = download_day(
            date_str=d,
            data_type=args.type,
            output_dir=args.output_dir,
            decompress_gzip=(not args.keep_gzip),
        )
        total_files += len(files)

    print("\n" + "=" * 78)
    print(f"  Download process finished! Total files: {total_files}")
    print("=" * 78)


if __name__ == "__main__":
    main()

