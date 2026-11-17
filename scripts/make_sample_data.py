"""Make the small FAKE sample data in data/sample/.

The sample has the same file layout as the NIST registered data, but the
numbers are invented (see src/synthetic.py). It exists so the tests and the
quick start run without the big download.

    python scripts/make_sample_data.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import PartSpec, make_dataset, write_part  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data/sample")
    parser.add_argument("--parts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dest = Path(args.dest)
    for part, points in make_dataset(args.parts, PartSpec(), args.seed).items():
        folder = dest / f"part{part}"
        shutil.rmtree(folder, ignore_errors=True)
        write_part(points, folder)
        print(f"part {part}: {len(points):,} rows -> {folder}")


if __name__ == "__main__":
    main()
