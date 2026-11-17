"""Download the registered NIST "Overhang Part X4" data (dataset mds2-3761).

    python scripts/download_data.py --parts 1          # start with one part
    python scripts/download_data.py --parts 1 2 3 4    # everything

Each part is one zip file. It is saved in data/raw/ and unpacked into
data/raw/part<N>/. Nothing in data/raw/ is tracked by Git.

Data description: https://doi.org/10.6028/NIST.AMS.100-69
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# File names as listed in the NIST data catalogue (note: "part1", then "part02").
URLS = {
    1: "https://data.nist.gov/od/ds/mds2-3761/part1.zip",
    2: "https://data.nist.gov/od/ds/mds2-3761/part02.zip",
    3: "https://data.nist.gov/od/ds/mds2-3761/part03.zip",
    4: "https://data.nist.gov/od/ds/mds2-3761/part04.zip",
}


def download(url: str, target: Path, chunk: int = 1 << 20) -> None:
    """Download `url` to `target`, showing progress. Skips finished files."""
    request = urllib.request.Request(url, headers={"User-Agent": "am-process-anomaly-detection"})
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length") or 0)
        if target.exists() and total and target.stat().st_size == total:
            print(f"  already downloaded: {target}")
            return
        print(f"  {url} -> {target}  ({total / 1e6:.0f} MB)" if total else f"  {url} -> {target}")
        tmp = target.with_suffix(target.suffix + ".part")
        done = 0
        with open(tmp, "wb") as fh:
            while True:
                block = response.read(chunk)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if total:
                    print(f"\r  {done / 1e6:8.0f} / {total / 1e6:.0f} MB", end="", file=sys.stderr)
        print(file=sys.stderr)
        tmp.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", type=int, nargs="+", default=[1], choices=sorted(URLS))
    parser.add_argument("--dest", default="data/raw")
    parser.add_argument("--keep-zip", action="store_true", help="do not delete the zip after unpacking")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for part in args.parts:
        zip_path = dest / f"part{part}.zip"
        folder = dest / f"part{part}"
        if folder.exists() and any(folder.rglob("*.csv")):
            print(f"part {part}: already unpacked in {folder}")
            continue
        print(f"part {part}:")
        download(URLS[part], zip_path)
        print(f"  unpacking to {folder}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(folder)
        if not args.keep_zip:
            zip_path.unlink()
    print("Done. Next: python scripts/inspect_data.py data/raw/part1")


if __name__ == "__main__":
    main()
