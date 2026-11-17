"""Run the whole experiment.

    python scripts/run_pipeline.py --config configs/sample.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--no-figures", action="store_true", help="only write tables")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = run(cfg, make_figures=not args.no_figures)
    print()
    print(Path(cfg["output_dir"], "results.md").read_text(encoding="utf-8"))
    print(f"{len(results['traceability'])} flagged segments listed in "
          f"{Path(cfg['output_dir']) / 'traceability.csv'}")


if __name__ == "__main__":
    main()
