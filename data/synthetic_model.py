"""Method-agnostic model definition for the synthetic merging benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        # Matrix-only shared layers keep this benchmark aligned with SVD-based
        # task-vector pruning.
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class TaskModel(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.backbone = Backbone(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def functional_backbone(x: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
    x = F.relu(F.linear(x, state["fc1.weight"], state.get("fc1.bias")))
    return F.relu(F.linear(x, state["fc2.weight"], state.get("fc2.bias")))


def tensor_count(state: dict[str, torch.Tensor]) -> int:
    return sum(int(tensor.numel()) for tensor in state.values())
