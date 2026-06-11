"""Visible train-slice scorer for saved model-merging predictions.

This evaluator is method-agnostic: it reads the candidate's prediction JSON
artifact and computes scores on a deterministic half of each task's examples.
It does not import from method/ or run inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


SLICE_NAME = "train"


def slice_indices(num_examples: int) -> list[int]:
    """Use even-indexed examples, preserving each task's class/domain mix."""

    return list(range(0, num_examples, 2))


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


def build_scores(raw: dict) -> dict:
    scores = {
        "slice": SLICE_NAME,
        "split_style": "per-task even-index examples from saved predictions",
        "experiments": {},
    }
    for exp_name, exp_payload in raw["experiments"].items():
        scores["experiments"][exp_name] = {"results": {}}
        for method_name, method_payload in exp_payload["methods"].items():
            scores["experiments"][exp_name]["results"][method_name] = summarize_method(method_payload)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/adarank_synthetic/predictions.json")
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    with Path(args.input).open() as f:
        raw = json.load(f)

    scores = build_scores(raw)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2)
        f.write("\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
