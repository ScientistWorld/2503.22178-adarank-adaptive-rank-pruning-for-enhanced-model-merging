"""Probe scorer for AdaRank configuration searches.

This is intentionally separate from the benchmark train/test scorers. It reads
candidate prediction artifacts, regenerates the matching synthetic labels from
the artifact config, and reports train/test deltas for every emitted method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from data.synthetic_tasks import make_tasks


SPLIT_SALT = "researchgym-adarank-disjoint-v3"


def slice_indices(num_examples: int, split: str) -> list[int]:
    selected = []
    for idx in range(num_examples):
        digest = hashlib.blake2b(f"{SPLIT_SALT}:{idx}".encode(), digest_size=2).digest()
        bucket = int.from_bytes(digest, "big") % 100
        if (split == "train" and bucket < 50) or (split == "test" and bucket >= 50):
            selected.append(idx)
    return selected


def labels_from_config(config: dict) -> dict[str, tuple[list[int], int]]:
    tasks = make_tasks(
        num_tasks=int(config.get("tasks", 4)),
        input_dim=int(config.get("input_dim", 32)),
        train_per_task=int(config.get("train_per_task", 768)),
        adapt_per_task=int(config.get("adapt_per_task", 256)),
        test_per_task=int(config.get("test_per_task", 512)),
        seed=int(config.get("seed", 7)),
    )
    return {task.name: (task.test_y.cpu().tolist(), task.num_classes) for task in tasks}


def accuracy(pred: list[int], label: list[int], indices: Iterable[int]) -> float:
    selected = list(indices)
    return sum(int(pred[idx] == label[idx]) for idx in selected) / len(selected)


def macro_recall(pred: list[int], label: list[int], indices: Iterable[int], num_classes: int) -> float:
    selected = list(indices)
    recalls = []
    for class_id in range(num_classes):
        class_indices = [idx for idx in selected if label[idx] == class_id]
        if not class_indices:
            continue
        recalls.append(sum(int(pred[idx] == label[idx]) for idx in class_indices) / len(class_indices))
    if not recalls:
        raise ValueError("Cannot score balanced accuracy with no represented classes.")
    return sum(recalls) / len(recalls)


def summarize_method(payload: dict, labels_by_task: dict[str, tuple[list[int], int]], split: str) -> dict[str, float]:
    per_task = {}
    per_task_balanced = {}
    for record in payload["predictions"]:
        label, num_classes = labels_by_task[record["task"]]
        indices = slice_indices(len(label), split)
        per_task[record["task"]] = accuracy(record["pred"], label, indices)
        per_task_balanced[record["task"]] = macro_recall(record["pred"], label, indices, num_classes)
    return {
        "avg_accuracy": sum(per_task.values()) / len(per_task),
        "balanced_accuracy": sum(per_task_balanced.values()) / len(per_task_balanced),
        "worst_task_accuracy": min(per_task.values()),
        "param_multiplier_vs_single": float(payload["param_multiplier_vs_single"]),
        "learnable_param_fraction": float(payload.get("learnable_param_fraction", 0.0)),
    }


def score_artifact(path: Path) -> dict:
    with path.open() as f:
        raw = json.load(f)
    config = raw.get("config", {})
    labels_by_task = labels_from_config(config)
    methods = raw["experiments"]["synthetic_4task_merge"]["methods"]
    train = {name: summarize_method(payload, labels_by_task, "train") for name, payload in methods.items()}
    test = {name: summarize_method(payload, labels_by_task, "test") for name, payload in methods.items()}
    fixed_train = train["fixed_topk_svd"]
    fixed_test = test["fixed_topk_svd"]
    ta_train = train["task_arithmetic"]
    ta_test = test["task_arithmetic"]

    rows = {}
    for name, payload in methods.items():
        trace = payload.get("adaptation_trace", {})
        rows[name] = {
            "train": train[name],
            "test": test[name],
            "delta_vs_fixed_train": {
                key: train[name][key] - fixed_train[key]
                for key in ("avg_accuracy", "balanced_accuracy", "worst_task_accuracy")
            },
            "delta_vs_fixed_test": {
                key: test[name][key] - fixed_test[key]
                for key in ("avg_accuracy", "balanced_accuracy", "worst_task_accuracy")
            },
            "delta_vs_task_arithmetic_test": {
                key: test[name][key] - ta_test[key]
                for key in ("avg_accuracy", "balanced_accuracy", "worst_task_accuracy")
            },
            "mask_density": trace.get("mask_density"),
            "adapt_steps": trace.get("adapt_steps"),
            "entropy_start": trace.get("entropy_step_0"),
            "learned_lambda_mean": trace.get("learned_lambda_mean"),
        }
    return {"config": config, "methods": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--output", default="outputs/core_claim_probe/summary.json")
    args = parser.parse_args()

    summary = {}
    for artifact in args.artifacts:
        path = Path(artifact)
        summary[path.parent.name] = score_artifact(path)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
