"""Evaluate saved model-merging outputs.

This module intentionally does not import method code. It reads predictions
written by any candidate method and computes the benchmark scores. Labels are
regenerated from the method-agnostic benchmark generator rather than trusted
from the prediction artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.synthetic_tasks import make_tasks


EXPECTED_TASKS = 4
EXPECTED_INPUT_DIM = 32
EXPECTED_TRAIN_PER_TASK = 768
EXPECTED_ADAPT_PER_TASK = 256
EXPECTED_TEST_PER_TASK = 512
EXPECTED_SEED = 7


def accuracy(pred: list[int], label: list[int]) -> float:
    if not label:
        return 0.0
    return sum(int(p == y) for p, y in zip(pred, label)) / len(label)


def macro_recall(pred: list[int], label: list[int], num_classes: int) -> float:
    recalls = []
    for class_id in range(num_classes):
        class_indices = [idx for idx, target in enumerate(label) if target == class_id]
        if not class_indices:
            continue
        correct = sum(int(pred[idx] == label[idx]) for idx in class_indices)
        recalls.append(correct / len(class_indices))
    if not recalls:
        raise ValueError("Cannot compute balanced accuracy without represented classes.")
    return sum(recalls) / len(recalls)


def expected_labels() -> dict[str, tuple[list[int], int]]:
    tasks = make_tasks(
        num_tasks=EXPECTED_TASKS,
        input_dim=EXPECTED_INPUT_DIM,
        train_per_task=EXPECTED_TRAIN_PER_TASK,
        adapt_per_task=EXPECTED_ADAPT_PER_TASK,
        test_per_task=EXPECTED_TEST_PER_TASK,
        seed=EXPECTED_SEED,
    )
    return {task.name: (task.test_y.cpu().tolist(), task.num_classes) for task in tasks}


def summarize_method(payload: dict, labels_by_task: dict[str, tuple[list[int], int]]) -> dict[str, float]:
    per_task: dict[str, float] = {}
    per_task_balanced: dict[str, float] = {}
    seen_tasks: set[str] = set()
    for record in payload["predictions"]:
        task_name = record["task"]
        if task_name not in labels_by_task:
            raise ValueError(f"Unknown task in prediction artifact: {task_name}.")
        if task_name in seen_tasks:
            raise ValueError(f"Duplicate task predictions for {task_name}.")
        seen_tasks.add(task_name)

        pred = record["pred"]
        label, num_classes = labels_by_task[task_name]
        if len(pred) != len(label):
            raise ValueError(f"Prediction length mismatch for {task_name}.")
        invalid = [value for value in pred if not isinstance(value, int) or value < 0 or value >= num_classes]
        if invalid:
            raise ValueError(f"Invalid class id in predictions for {task_name}.")

        per_task[task_name] = accuracy(pred, label)
        per_task_balanced[task_name] = macro_recall(pred, label, num_classes)

    missing_tasks = sorted(set(labels_by_task) - seen_tasks)
    if missing_tasks:
        joined = ", ".join(missing_tasks)
        raise ValueError(f"Method payload missing task predictions: {joined}")

    avg = sum(per_task.values()) / len(per_task)
    result = {
        "avg_accuracy": avg,
        "balanced_accuracy": sum(per_task_balanced.values()) / len(per_task_balanced),
        "worst_task_accuracy": min(per_task.values()),
        "param_multiplier_vs_single": float(payload["param_multiplier_vs_single"]),
        "learnable_param_fraction": float(payload.get("learnable_param_fraction", 0.0)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/adarank_synthetic/predictions.json")
    parser.add_argument("--output", default="scoring/scores.json")
    parser.add_argument("--reference", default="scoring/reference.json")
    parser.add_argument(
        "--include-extra-methods",
        action="store_true",
        help="Also write diagnostic methods that are absent from reference.json.",
    )
    args = parser.parse_args()

    with Path(args.input).open() as f:
        raw = json.load(f)
    reference_methods: dict[str, set[str]] = {}
    reference_path = Path(args.reference)
    if reference_path.exists() and not args.include_extra_methods:
        with reference_path.open() as f:
            reference = json.load(f)
        for exp_name, exp_payload in reference.get("experiments", {}).items():
            reference_methods[exp_name] = set(exp_payload.get("results", {}).keys())

    labels_by_task = expected_labels()
    scores = {"experiments": {}}
    for exp_name, exp_payload in raw["experiments"].items():
        scores["experiments"][exp_name] = {"results": {}}
        for method_name, method_payload in exp_payload["methods"].items():
            if reference_methods and method_name not in reference_methods.get(exp_name, set()):
                print(f"Skipping diagnostic method absent from reference.json: {exp_name}.{method_name}")
                continue
            scores["experiments"][exp_name]["results"][method_name] = summarize_method(method_payload, labels_by_task)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
