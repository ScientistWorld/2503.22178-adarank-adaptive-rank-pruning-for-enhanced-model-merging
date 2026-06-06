"""Synthetic multi-task classification benchmark for model merging.

The data generator is method-agnostic. It creates several related but
conflicting tasks from fixed teacher networks. Labels are generated once from
the teachers; methods only receive labels during individual fine-tuning and
receive unlabeled validation inputs during test-time adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskData:
    name: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    adapt_x: torch.Tensor
    adapt_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    num_classes: int


def _teacher_labels(x: torch.Tensor, task_id: int, num_classes: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(10_000 + task_id)
    in_dim = x.shape[1]
    hidden = 48
    w1 = torch.randn(in_dim, hidden, generator=gen) / (in_dim**0.5)
    # Alternate sign structure creates task interference in shared weights.
    if task_id % 2:
        w1[: in_dim // 2] *= -1.0
    w2 = torch.randn(hidden, num_classes, generator=gen) / (hidden**0.5)
    bias = torch.randn(num_classes, generator=gen) * 0.15
    logits = torch.relu(x @ w1) @ w2 + bias
    return logits.argmax(dim=1)


def make_tasks(
    num_tasks: int = 4,
    input_dim: int = 32,
    train_per_task: int = 768,
    adapt_per_task: int = 256,
    test_per_task: int = 512,
    seed: int = 7,
) -> list[TaskData]:
    """Create a deterministic suite of tasks.

    Class counts differ across tasks to mimic the paper's heterogeneous
    benchmark and to make fixed-rank truncation less natural.
    """

    gen = torch.Generator().manual_seed(seed)
    class_counts = [3, 4, 5, 6, 3, 5, 4, 7]
    tasks: list[TaskData] = []
    total = train_per_task + adapt_per_task + test_per_task
    for task_id in range(num_tasks):
        x = torch.randn(total, input_dim, generator=gen)
        # Give each task a mild domain shift without making it linearly trivial.
        x = x + 0.18 * torch.sin(x.roll(shifts=task_id + 1, dims=1))
        x[:, task_id:: max(num_tasks, 1)] += 0.25 * ((task_id % 3) - 1)
        num_classes = class_counts[task_id % len(class_counts)]
        y = _teacher_labels(x, task_id, num_classes)
        train_end = train_per_task
        adapt_end = train_end + adapt_per_task
        tasks.append(
            TaskData(
                name=f"task_{task_id}",
                train_x=x[:train_end].clone(),
                train_y=y[:train_end].clone(),
                adapt_x=x[train_end:adapt_end].clone(),
                adapt_y=y[train_end:adapt_end].clone(),
                test_x=x[adapt_end:].clone(),
                test_y=y[adapt_end:].clone(),
                num_classes=num_classes,
            )
        )
    return tasks

