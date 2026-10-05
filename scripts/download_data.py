#!/usr/bin/env python3
"""Download the 30 MCMRT workbooks from Science Data Bank into ``data/raw/``.

The dataset page (DOI 10.57760/sciencedb.15823) lists its files only through
JavaScript, but it also embeds a schema.org description with a direct link,
size and MD5 checksum for every file. This script reads that description,
downloads each file and checks it. Files already present with the right
checksum are skipped.

The data are CC0 (public domain). Please cite Zhang et al., Scientific Data 11,
946 (2024), doi:10.1038/s41597-024-03780-5.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --out data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
import urllib.request
from pathlib import Path

DATASET_PAGE = "https://www.scidb.cn/en/detail?dataSetId=7b1a8766e1e947e1884707f6a1016e4f"
EXPECTED_FILES = 30

FILE_PATTERN = re.compile(
    r'"@type":"cr:FileObject","contentSize":"(?P<size>\d+) B",'
    r'"contentUrl":"(?P<url>[^"]+)",[^{}]*?"md5":"(?P<md5>[0-9a-f]{32})",'
    r'"name":"(?P<name>[^"]+\.xlsx)"'
)


def fetch(url: str, attempts: int = 3) -> bytes:
    """Download a URL, retrying on network errors."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError as exc:
            if attempt == attempts:
                raise
            print(f"  retrying ({exc})", file=sys.stderr)
            time.sleep(3 * attempt)
    raise AssertionError("unreachable")


def list_files(page: str) -> list[dict[str, str]]:
    """File entries (name, url, size, md5) embedded in the dataset page."""
    files = {m["name"]: m.groupdict() for m in FILE_PATTERN.finditer(page)}
    return [files[name] for name in sorted(files)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    files = list_files(fetch(DATASET_PAGE).decode("utf-8", errors="replace"))
    if len(files) != EXPECTED_FILES:
        print(
            f"error: found {len(files)} files on the dataset page, expected "
            f"{EXPECTED_FILES}. The page layout may have changed; download the "
            f"files by hand from {DATASET_PAGE}",
            file=sys.stderr,
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for entry in files:
        dest = args.out / entry["name"]
        if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == entry["md5"]:
            print(f"ok      {entry['name']} (already present)")
            continue
        data = fetch(entry["url"])
        if len(data) != int(entry["size"]) or hashlib.md5(data).hexdigest() != entry["md5"]:
            failed.append(entry["name"])
            print(f"FAILED  {entry['name']} (checksum mismatch, not saved)")
            continue
        dest.write_bytes(data)
        print(f"ok      {entry['name']}")

    if failed:
        print(f"\n{len(failed)} file(s) failed; run again to retry.", file=sys.stderr)
        return 1
    print(f"\nall {len(files)} files in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
