"""Run the shared synthetic benchmark and emit baseline-only outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


BASELINE_METHODS = {"task_arithmetic", "fixed_topk_svd", "individual"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/adarank_synthetic/predictions.json")
    parser.add_argument("--output-dir", default="outputs/baseline_synthetic")
    args = parser.parse_args()
    source = Path(args.source)
    with source.open() as f:
        raw = json.load(f)
    exp = raw["experiments"]["synthetic_4task_merge"]
    exp["methods"] = {k: v for k, v in exp["methods"].items() if k in BASELINE_METHODS}
    out = {"config": raw.get("config", {}), "experiments": {"synthetic_4task_merge": exp}}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.json").open("w") as f:
        json.dump(out, f, indent=2)

    source_artifact = source.with_name("artifact.pt")
    if source_artifact.exists():
        try:
            artifact = torch.load(source_artifact, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(source_artifact, map_location="cpu")
        artifact["methods"] = {k: v for k, v in artifact["methods"].items() if k in BASELINE_METHODS}
        torch.save(artifact, out_dir / "artifact.pt")


if __name__ == "__main__":
    main()
