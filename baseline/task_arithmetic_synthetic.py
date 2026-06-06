"""Run the shared synthetic benchmark and emit baseline-only outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/adarank_synthetic/predictions.json")
    parser.add_argument("--output-dir", default="outputs/baseline_synthetic")
    args = parser.parse_args()
    with Path(args.source).open() as f:
        raw = json.load(f)
    exp = raw["experiments"]["synthetic_4task_merge"]
    exp["methods"] = {k: v for k, v in exp["methods"].items() if k in {"task_arithmetic", "fixed_topk_svd", "individual"}}
    out = {"config": raw.get("config", {}), "experiments": {"synthetic_4task_merge": exp}}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.json").open("w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
