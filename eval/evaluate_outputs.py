"""Evaluate saved model-merging outputs.

This module intentionally does not import method code. It reads predictions
written by any candidate method and computes the benchmark scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def accuracy(pred: list[int], label: list[int]) -> float:
    if not label:
        return 0.0
    return sum(int(p == y) for p, y in zip(pred, label)) / len(label)


def summarize_method(payload: dict) -> dict[str, float]:
    per_task = {}
    for record in payload["predictions"]:
        per_task[record["task"]] = accuracy(record["pred"], record["label"])
    avg = sum(per_task.values()) / len(per_task)
    result = {
        "avg_accuracy": avg,
        "param_multiplier_vs_single": float(payload["param_multiplier_vs_single"]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/adarank_synthetic/predictions.json")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    with Path(args.input).open() as f:
        raw = json.load(f)

    scores = {"experiments": {}}
    for exp_name, exp_payload in raw["experiments"].items():
        scores["experiments"][exp_name] = {"results": {}}
        for method_name, method_payload in exp_payload["methods"].items():
            scores["experiments"][exp_name]["results"][method_name] = summarize_method(method_payload)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
