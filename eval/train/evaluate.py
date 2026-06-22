"""Visible train-slice scorer for saved model-merging artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval.artifact_scoring import build_scores, load_artifact


HASH_SALT = "researchgym-adarank-disjoint-v3"


def slice_indices(num_examples: int) -> list[int]:
    """Use one side of a stable hash partition as the visible slice."""

    selected = []
    for idx in range(num_examples):
        digest = hashlib.blake2b(f"{HASH_SALT}:{idx}".encode(), digest_size=2).digest()
        if int.from_bytes(digest, "big") % 100 < 50:
            selected.append(idx)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/adarank_synthetic/artifact.pt")
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    artifact = load_artifact(args.input)
    scores = build_scores(artifact, indices_for_task=slice_indices)
    scores["slice"] = "train"
    scores["split_style"] = "visible side of a per-task disjoint salted hash partition from saved model states"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2)
        f.write("\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
