"""Held-out test-slice scorer for saved model-merging predictions.

This evaluator is independent from eval/train/. It reads the same candidate
prediction artifact, scores the complementary held-out half of each task, and
can enforce the method set previously written by scoring/scores_train.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


SLICE_NAME = "test"


def slice_indices(num_examples: int) -> list[int]:
    """Use odd-indexed examples, complementary to the visible train slice."""

    return list(range(1, num_examples, 2))


def accuracy(pred: list[int], label: list[int], indices: Iterable[int]) -> float:
    selected = list(indices)
    if not selected:
        raise ValueError("Cannot score an empty split.")
    correct = sum(int(pred[idx] == label[idx]) for idx in selected)
    return correct / len(selected)


def summarize_method(payload: dict) -> dict[str, float]:
    per_task: dict[str, float] = {}
    for record in payload["predictions"]:
        pred = record["pred"]
        label = record["label"]
        if len(pred) != len(label):
            raise ValueError(f"Prediction/label length mismatch for {record['task']}.")
        indices = slice_indices(len(label))
        per_task[record["task"]] = accuracy(pred, label, indices)

    if not per_task:
        raise ValueError("Method payload contained no task predictions.")

    return {
        "avg_accuracy": sum(per_task.values()) / len(per_task),
        "param_multiplier_vs_single": float(payload["param_multiplier_vs_single"]),
        "learnable_param_fraction": float(payload.get("learnable_param_fraction", 0.0)),
    }


def load_expected_methods(train_scores_path: Path) -> dict[str, set[str]] | None:
    if not train_scores_path.exists():
        return None
    with train_scores_path.open() as f:
        train_scores = json.load(f)
    expected: dict[str, set[str]] = {}
    for exp_name, exp_payload in train_scores.get("experiments", {}).items():
        expected[exp_name] = set(exp_payload.get("results", {}).keys())
    return expected


def build_scores(raw: dict, expected_methods: dict[str, set[str]] | None) -> dict:
    scores = {
        "slice": SLICE_NAME,
        "split_style": "per-task odd-index examples from saved predictions",
        "experiments": {},
    }

    if expected_methods is None:
        experiment_names = sorted(raw["experiments"].keys())
    else:
        experiment_names = sorted(expected_methods.keys())
        missing_experiments = sorted(set(expected_methods) - set(raw["experiments"]))
        if missing_experiments:
            joined = ", ".join(missing_experiments)
            raise ValueError(f"Train-scored experiments missing from test input: {joined}")

    for exp_name in experiment_names:
        raw_methods = raw["experiments"][exp_name]["methods"]
        if expected_methods is None:
            methods_to_score = sorted(raw_methods.keys())
        else:
            methods_to_score = sorted(expected_methods.get(exp_name, set()))
            missing = [name for name in methods_to_score if name not in raw_methods]
            if missing:
                joined = ", ".join(missing)
                raise ValueError(f"Train-scored methods missing from test input for {exp_name}: {joined}")

        scores["experiments"][exp_name] = {"results": {}}
        for method_name in methods_to_score:
            scores["experiments"][exp_name]["results"][method_name] = summarize_method(raw_methods[method_name])

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/adarank_synthetic/predictions.json")
    parser.add_argument("--output", default="scoring/scores_test.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    args = parser.parse_args()

    with Path(args.input).open() as f:
        raw = json.load(f)

    expected_methods = load_expected_methods(Path(args.train_scores))
    scores = build_scores(raw, expected_methods)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2)
        f.write("\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
