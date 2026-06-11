"""Held-out test-slice scorer for saved model-merging predictions.

This evaluator is independent from eval/train/. It reads the same candidate
prediction artifact, scores a separately salted held-out sample of each task,
and can enforce the method set previously written by scoring/scores_train.json.
Labels are regenerated from the benchmark data code instead of trusted from the
candidate artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from data.synthetic_tasks import make_tasks


SLICE_NAME = "test"
EXPECTED_TASKS = 4
EXPECTED_INPUT_DIM = 32
EXPECTED_TRAIN_PER_TASK = 768
EXPECTED_ADAPT_PER_TASK = 256
EXPECTED_TEST_PER_TASK = 512
EXPECTED_SEED = 7
HASH_SALT = "researchgym-adarank-heldout-test-v2"


def slice_indices(num_examples: int) -> list[int]:
    """Use a stable hidden hash split rather than a positional complement."""

    selected = []
    for idx in range(num_examples):
        digest = hashlib.blake2b(f"{HASH_SALT}:{idx}".encode(), digest_size=2).digest()
        if int.from_bytes(digest, "big") % 100 < 50:
            selected.append(idx)
    return selected


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


def accuracy(pred: list[int], label: list[int], indices: Iterable[int]) -> float:
    selected = list(indices)
    if not selected:
        raise ValueError("Cannot score an empty split.")
    correct = sum(int(pred[idx] == label[idx]) for idx in selected)
    return correct / len(selected)


def macro_recall(pred: list[int], label: list[int], indices: Iterable[int], num_classes: int) -> float:
    selected = list(indices)
    recalls = []
    for class_id in range(num_classes):
        class_indices = [idx for idx in selected if label[idx] == class_id]
        if not class_indices:
            continue
        correct = sum(int(pred[idx] == label[idx]) for idx in class_indices)
        recalls.append(correct / len(class_indices))
    if not recalls:
        raise ValueError("Cannot score balanced accuracy with no represented classes.")
    return sum(recalls) / len(recalls)


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
        indices = slice_indices(len(label))
        per_task[task_name] = accuracy(pred, label, indices)
        per_task_balanced[task_name] = macro_recall(pred, label, indices, num_classes)

    missing_tasks = sorted(set(labels_by_task) - seen_tasks)
    if missing_tasks:
        joined = ", ".join(missing_tasks)
        raise ValueError(f"Method payload missing task predictions: {joined}")

    return {
        "avg_accuracy": sum(per_task.values()) / len(per_task),
        "balanced_accuracy": sum(per_task_balanced.values()) / len(per_task_balanced),
        "worst_task_accuracy": min(per_task.values()),
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
    labels_by_task = expected_labels()
    scores = {
        "slice": SLICE_NAME,
        "split_style": "per-task hidden salted hash sample from saved predictions with independently regenerated labels",
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
            scores["experiments"][exp_name]["results"][method_name] = summarize_method(raw_methods[method_name], labels_by_task)

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
