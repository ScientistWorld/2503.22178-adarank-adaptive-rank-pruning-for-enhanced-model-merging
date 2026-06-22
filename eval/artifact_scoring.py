"""Shared, method-agnostic scoring from saved model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from data.synthetic_model import Backbone, functional_backbone, tensor_count
from data.synthetic_tasks import make_tasks


EXPECTED_TASKS = 4
EXPECTED_INPUT_DIM = 32
EXPECTED_TRAIN_PER_TASK = 768
EXPECTED_ADAPT_PER_TASK = 256
EXPECTED_TEST_PER_TASK = 512
EXPECTED_SEED = 7
EXPECTED_HIDDEN_DIM = 64


def expected_tasks():
    return make_tasks(
        num_tasks=EXPECTED_TASKS,
        input_dim=EXPECTED_INPUT_DIM,
        train_per_task=EXPECTED_TRAIN_PER_TASK,
        adapt_per_task=EXPECTED_ADAPT_PER_TASK,
        test_per_task=EXPECTED_TEST_PER_TASK,
        seed=EXPECTED_SEED,
    )


def resolve_artifact_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.suffix == ".json":
        sibling = path.with_name("artifact.pt")
        if sibling.exists():
            return sibling
    return path


def load_artifact(path: str | Path) -> dict:
    artifact_path = resolve_artifact_path(str(path))
    try:
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(artifact_path, map_location="cpu")
    if artifact.get("schema_version") != 1:
        raise ValueError("Unsupported or missing artifact schema_version.")
    config = artifact.get("config", {})
    expected = {
        "tasks": EXPECTED_TASKS,
        "hidden_dim": EXPECTED_HIDDEN_DIM,
        "train_per_task": EXPECTED_TRAIN_PER_TASK,
        "adapt_per_task": EXPECTED_ADAPT_PER_TASK,
        "test_per_task": EXPECTED_TEST_PER_TASK,
        "seed": EXPECTED_SEED,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Artifact config mismatch for {key}: expected {value}, found {config.get(key)}.")
    return artifact


def head_logits(features: torch.Tensor, head_state: dict[str, torch.Tensor]) -> torch.Tensor:
    return F.linear(features, head_state["weight"], head_state["bias"])


def validate_head_states(head_states: list[dict[str, torch.Tensor]], num_classes: list[int]) -> None:
    if len(head_states) != len(num_classes):
        raise ValueError("Head count does not match task count.")
    for idx, (head_state, classes) in enumerate(zip(head_states, num_classes)):
        weight = head_state.get("weight")
        bias = head_state.get("bias")
        if weight is None or bias is None:
            raise ValueError(f"Head {idx} is missing weight or bias.")
        if tuple(weight.shape) != (classes, EXPECTED_HIDDEN_DIM) or tuple(bias.shape) != (classes,):
            raise ValueError(f"Head {idx} shape does not match the benchmark task.")


def method_predictions(method_payload: dict, tasks) -> list[dict[str, object]]:
    kind = method_payload.get("kind")
    head_states = method_payload.get("head_states")
    if not isinstance(head_states, list):
        raise ValueError("Method artifact is missing task heads.")
    validate_head_states(head_states, [task.num_classes for task in tasks])

    records = []
    with torch.no_grad():
        if kind == "merged":
            backbone_state = method_payload.get("backbone_state")
            if not isinstance(backbone_state, dict):
                raise ValueError("Merged method is missing backbone_state.")
            for task, head_state in zip(tasks, head_states):
                logits = head_logits(functional_backbone(task.test_x, backbone_state), head_state)
                records.append({"task": task.name, "pred": logits.argmax(dim=1).tolist()})
        elif kind == "individual":
            backbone_states = method_payload.get("backbone_states")
            if not isinstance(backbone_states, list) or len(backbone_states) != len(tasks):
                raise ValueError("Individual method must provide one backbone per task.")
            for task, backbone_state, head_state in zip(tasks, backbone_states, head_states):
                logits = head_logits(functional_backbone(task.test_x, backbone_state), head_state)
                records.append({"task": task.name, "pred": logits.argmax(dim=1).tolist()})
        else:
            raise ValueError(f"Unknown method artifact kind: {kind}.")
    return records


def accuracy(pred: list[int], label: list[int], indices: Iterable[int]) -> float:
    selected = list(indices)
    if not selected:
        raise ValueError("Cannot score an empty split.")
    return sum(int(pred[idx] == label[idx]) for idx in selected) / len(selected)


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
        raise ValueError("Cannot compute balanced accuracy without represented classes.")
    return sum(recalls) / len(recalls)


def count_heads(head_states: list[dict[str, torch.Tensor]]) -> int:
    return sum(tensor_count(head_state) for head_state in head_states)


def expected_parameter_budget(tasks) -> tuple[int, int, int]:
    backbone = Backbone(EXPECTED_INPUT_DIM, EXPECTED_HIDDEN_DIM)
    backbone_params = sum(int(param.numel()) for param in backbone.parameters())
    head_params = sum((EXPECTED_HIDDEN_DIM * task.num_classes) + task.num_classes for task in tasks)
    single_params = backbone_params + head_params
    individual_params = backbone_params * len(tasks) + head_params
    return backbone_params, single_params, individual_params


def method_parameter_count(method_payload: dict) -> int:
    head_states = method_payload["head_states"]
    head_params = count_heads(head_states)
    if method_payload["kind"] == "merged":
        return tensor_count(method_payload["backbone_state"]) + head_params
    if method_payload["kind"] == "individual":
        return sum(tensor_count(state) for state in method_payload["backbone_states"]) + head_params
    raise ValueError(f"Unknown method artifact kind: {method_payload['kind']}.")


def learnable_adaptation_params(method_payload: dict) -> int:
    adaptation = method_payload.get("adaptation", {"type": "none"})
    if adaptation.get("type") == "none":
        return 0
    if adaptation.get("type") == "adarank_mask":
        return int(adaptation["num_tasks"]) * sum(int(rank) for rank in adaptation["matrix_ranks"])
    raise ValueError(f"Unsupported adaptation state: {adaptation.get('type')}.")


def summarize_method(
    method_payload: dict,
    tasks,
    single_params: int,
    individual_params: int,
    indices_for_task: Callable[[int], Iterable[int]],
) -> dict[str, float]:
    predictions = method_predictions(method_payload, tasks)
    per_task = {}
    per_task_balanced = {}
    labels_by_task = {task.name: (task.test_y.tolist(), task.num_classes) for task in tasks}
    seen = set()
    for record in predictions:
        task_name = record["task"]
        if task_name in seen:
            raise ValueError(f"Duplicate predictions for {task_name}.")
        seen.add(task_name)
        if task_name not in labels_by_task:
            raise ValueError(f"Unknown task in artifact predictions: {task_name}.")
        pred = record["pred"]
        label, num_classes = labels_by_task[task_name]
        if len(pred) != len(label):
            raise ValueError(f"Prediction length mismatch for {task_name}.")
        invalid = [value for value in pred if not isinstance(value, int) or value < 0 or value >= num_classes]
        if invalid:
            raise ValueError(f"Invalid class id in predictions for {task_name}.")
        indices = list(indices_for_task(len(label)))
        per_task[task_name] = accuracy(pred, label, indices)
        per_task_balanced[task_name] = macro_recall(pred, label, indices, num_classes)

    missing = sorted(set(labels_by_task) - seen)
    if missing:
        raise ValueError(f"Method artifact missing task predictions: {', '.join(missing)}")

    method_params = method_parameter_count(method_payload)
    learnable_params = learnable_adaptation_params(method_payload)
    if method_payload["kind"] == "merged" and method_params != single_params:
        raise ValueError("Merged method parameter count does not match the benchmark single-model budget.")
    return {
        "avg_accuracy": sum(per_task.values()) / len(per_task),
        "balanced_accuracy": sum(per_task_balanced.values()) / len(per_task_balanced),
        "worst_task_accuracy": min(per_task.values()),
        "param_multiplier_vs_single": method_params / single_params,
        "learnable_param_fraction": learnable_params / individual_params,
    }


def reference_method_filter(reference_path: str | Path, include_extra_methods: bool) -> dict[str, set[str]]:
    path = Path(reference_path)
    if include_extra_methods or not path.exists():
        return {}
    with path.open() as f:
        reference = json.load(f)
    return {
        exp_name: set(exp_payload.get("results", {}).keys())
        for exp_name, exp_payload in reference.get("experiments", {}).items()
    }


def build_scores(
    artifact: dict,
    reference_methods: dict[str, set[str]] | None = None,
    indices_for_task: Callable[[int], Iterable[int]] | None = None,
) -> dict:
    tasks = expected_tasks()
    _, single_params, individual_params = expected_parameter_budget(tasks)
    if indices_for_task is None:
        indices_for_task = lambda n: range(n)
    reference_methods = reference_methods or {}
    scores = {"experiments": {"synthetic_4task_merge": {"results": {}}}}
    raw_methods = artifact["methods"]
    methods_to_score = sorted(raw_methods.keys())
    if reference_methods:
        methods_to_score = [
            method for method in methods_to_score if method in reference_methods.get("synthetic_4task_merge", set())
        ]
    for method_name in methods_to_score:
        if reference_methods and method_name not in reference_methods.get("synthetic_4task_merge", set()):
            print(f"Skipping diagnostic method absent from reference.json: synthetic_4task_merge.{method_name}")
            continue
        scores["experiments"]["synthetic_4task_merge"]["results"][method_name] = summarize_method(
            raw_methods[method_name], tasks, single_params, individual_params, indices_for_task
        )
    return scores
