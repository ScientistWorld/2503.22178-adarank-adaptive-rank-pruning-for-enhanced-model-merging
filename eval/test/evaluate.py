"""Held-out test-slice scorer for saved model-merging artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval.artifact_scoring import build_scores as build_artifact_scores
from eval.artifact_scoring import load_artifact


HASH_SALT = "researchgym-adarank-disjoint-v3"


def slice_indices(num_examples: int) -> list[int]:
    """Use the held-out side of the same stable hash partition as train."""

    selected = []
    for idx in range(num_examples):
        digest = hashlib.blake2b(f"{HASH_SALT}:{idx}".encode(), digest_size=2).digest()
        if int.from_bytes(digest, "big") % 100 >= 50:
            selected.append(idx)
    return selected


def load_expected_methods(train_scores_path: Path) -> dict[str, set[str]] | None:
    if not train_scores_path.exists():
        return None
    with train_scores_path.open() as f:
        train_scores = json.load(f)
    expected: dict[str, set[str]] = {}
    for exp_name, exp_payload in train_scores.get("experiments", {}).items():
        expected[exp_name] = set(exp_payload.get("results", {}).keys())
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/adarank_synthetic/artifact.pt")
    parser.add_argument("--output", default="scoring/scores_test.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    artifact = load_artifact(args.input)
    expected_methods = load_expected_methods(Path(args.train_scores))
    scores = build_artifact_scores(artifact, expected_methods, indices_for_task=slice_indices)
    scores["slice"] = "test"
    scores["split_style"] = "held-out side of the same per-task disjoint salted hash partition from saved model states"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2)
        f.write("\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
