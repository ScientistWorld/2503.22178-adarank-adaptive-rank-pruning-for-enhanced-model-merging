"""Evaluate saved model-merging artifacts without importing method code."""

from __future__ import annotations

import argparse
from pathlib import Path

import json

from eval.artifact_scoring import build_scores, load_artifact, reference_method_filter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/adarank_synthetic/artifact.pt")
    parser.add_argument("--output", default="scoring/scores.json")
    parser.add_argument("--reference", default="scoring/reference.json")
    parser.add_argument(
        "--include-extra-methods",
        action="store_true",
        help="Also write diagnostic methods that are absent from reference.json.",
    )
    args = parser.parse_args()

    artifact = load_artifact(args.input)
    reference_methods = reference_method_filter(args.reference, args.include_extra_methods)
    scores = build_scores(artifact, reference_methods)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
