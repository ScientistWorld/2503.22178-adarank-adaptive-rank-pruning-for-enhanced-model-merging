"""Scaled AdaRank reproduction on a synthetic multi-task benchmark.

This implements the paper's computational procedure:
1. train task-specific models from one pretrained initialization;
2. form task vectors between fine-tuned and pretrained weights;
3. decompose each matrix task vector with SVD;
4. merge with binary singular-component masks and layer coefficients;
5. adapt masks on unlabeled data by entropy minimization with an STE;
6. evaluate the resulting merged model on all tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.synthetic_tasks import TaskData, make_tasks


class Backbone(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

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


@dataclass
class SVDComponent:
    name: str
    base: torch.Tensor
    task_us: list[torch.Tensor]
    task_s: list[torch.Tensor]
    task_vh: list[torch.Tensor]
    task_delta: list[torch.Tensor]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def train_pretrained(tasks: list[TaskData], hidden_dim: int, epochs: int, lr: float) -> tuple[Backbone, list[nn.Linear]]:
    backbone = Backbone(tasks[0].train_x.shape[1], hidden_dim)
    heads = [nn.Linear(hidden_dim, task.num_classes) for task in tasks]
    params = list(backbone.parameters()) + [p for head in heads for p in head.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for task, head in zip(tasks, heads):
            logits = head(backbone(task.train_x))
            loss = loss + F.cross_entropy(logits, task.train_y)
        loss.backward()
        opt.step()
    return backbone, heads


def train_individuals(
    pretrained: Backbone,
    pretrained_heads: list[nn.Linear],
    tasks: list[TaskData],
    hidden_dim: int,
    epochs: int,
    lr: float,
) -> tuple[list[Backbone], list[nn.Linear]]:
    backbones: list[Backbone] = []
    heads: list[nn.Linear] = []
    for task, base_head in zip(tasks, pretrained_heads):
        model = TaskModel(task.num_classes, task.train_x.shape[1], hidden_dim)
        model.backbone.load_state_dict(deepcopy(pretrained.state_dict()))
        model.head.load_state_dict(deepcopy(base_head.state_dict()))
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model(task.train_x), task.train_y)
            loss.backward()
            opt.step()
        backbones.append(deepcopy(model.backbone))
        heads.append(deepcopy(model.head))
    return backbones, heads


def backbone_state(backbone: Backbone) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in backbone.state_dict().items()}


def build_svd_components(base: Backbone, task_backbones: list[Backbone]) -> tuple[list[SVDComponent], dict[str, torch.Tensor]]:
    base_state = backbone_state(base)
    task_states = [backbone_state(model) for model in task_backbones]
    matrix_components: list[SVDComponent] = []
    vector_deltas: dict[str, list[torch.Tensor]] = {}
    for name, base_tensor in base_state.items():
        deltas = [state[name] - base_tensor for state in task_states]
        if base_tensor.ndim == 2:
            us, ss, vhs = [], [], []
            for delta in deltas:
                u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                us.append(u.detach())
                ss.append(s.detach())
                vhs.append(vh.detach())
            matrix_components.append(SVDComponent(name, base_tensor, us, ss, vhs, deltas))
        else:
            vector_deltas[name] = [delta.detach() for delta in deltas]
    return matrix_components, {"base": base_state, "vectors": vector_deltas}


def ste_binary(logits: torch.Tensor, temperature: float = 10.0) -> torch.Tensor:
    prob = torch.sigmoid(logits / temperature)
    hard = (prob >= 0.5).to(prob.dtype)
    return hard.detach() - prob.detach() + prob


def mask_logits_for_initial(mask: torch.Tensor) -> torch.Tensor:
    # AdaRank initializes binary masks to a chosen baseline, then adapts the
    # continuous STE parameters. Small signed logits keep the initial hard mask
    # intact while allowing components to cross the 0.5 threshold during TTA.
    return torch.where(mask > 0, torch.full_like(mask, 1.0), torch.full_like(mask, -1.0))


def init_masks(components: list[SVDComponent], num_tasks: int, mode: str, top_fraction: float) -> nn.ParameterList:
    params = nn.ParameterList()
    for comp in components:
        rank = comp.task_s[0].numel()
        if mode == "full":
            mask = torch.ones(num_tasks, rank)
        else:
            k = max(1, math.ceil(rank * top_fraction))
            mask = torch.zeros(num_tasks, rank)
            mask[:, :k] = 1.0
        params.append(nn.Parameter(mask_logits_for_initial(mask)))
    return params


def construct_state(
    components: list[SVDComponent],
    vector_pack: dict[str, object],
    mask_params: nn.ParameterList | None,
    lambdas: torch.Tensor,
    mode: str,
    top_fraction: float,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for layer_idx, comp in enumerate(components):
        merged = comp.base.clone()
        for task_idx, (u, s, vh, full_delta) in enumerate(zip(comp.task_us, comp.task_s, comp.task_vh, comp.task_delta)):
            if mode == "full":
                delta = full_delta
            else:
                if mask_params is None:
                    rank = s.numel()
                    k = max(1, math.ceil(rank * top_fraction))
                    mask = torch.zeros_like(s)
                    mask[:k] = 1.0
                else:
                    mask = ste_binary(mask_params[layer_idx][task_idx])
                delta = (u * (s * mask).unsqueeze(0)) @ vh
            merged = merged + lambdas[layer_idx] * delta
        state[comp.name] = merged
    base_state: dict[str, torch.Tensor] = vector_pack["base"]  # type: ignore[assignment]
    vector_deltas: dict[str, list[torch.Tensor]] = vector_pack["vectors"]  # type: ignore[assignment]
    for name, base_tensor in base_state.items():
        if name in state:
            continue
        merged = base_tensor.clone()
        for delta in vector_deltas[name]:
            merged = merged + lambdas.mean() * delta
        state[name] = merged
    return state


def functional_backbone(x: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
    x = F.relu(F.linear(x, state["fc1.weight"], state["fc1.bias"]))
    return F.relu(F.linear(x, state["fc2.weight"], state["fc2.bias"]))


def evaluate_state(tasks: list[TaskData], heads: list[nn.Linear], state: dict[str, torch.Tensor]) -> tuple[dict[str, float], list[dict[str, object]]]:
    accuracies: dict[str, float] = {}
    predictions = []
    with torch.no_grad():
        for task, head in zip(tasks, heads):
            logits = head(functional_backbone(task.test_x, state))
            pred = logits.argmax(dim=1)
            acc = (pred == task.test_y).float().mean().item()
            accuracies[task.name] = acc
            predictions.append(
                {
                    "task": task.name,
                    "pred": pred.cpu().tolist(),
                    "label": task.test_y.cpu().tolist(),
                    "num_classes": task.num_classes,
                }
            )
    return accuracies, predictions


def evaluate_individuals(tasks: list[TaskData], backbones: list[Backbone], heads: list[nn.Linear]) -> tuple[dict[str, float], list[dict[str, object]]]:
    accuracies: dict[str, float] = {}
    predictions = []
    with torch.no_grad():
        for task, backbone, head in zip(tasks, backbones, heads):
            pred = head(backbone(task.test_x)).argmax(dim=1)
            accuracies[task.name] = (pred == task.test_y).float().mean().item()
            predictions.append({"task": task.name, "pred": pred.tolist(), "label": task.test_y.tolist(), "num_classes": task.num_classes})
    return accuracies, predictions


def adapt_adarank(
    tasks: list[TaskData],
    heads: list[nn.Linear],
    components: list[SVDComponent],
    vector_pack: dict[str, object],
    init_mode: str,
    top_fraction: float,
    steps: int,
    mask_lr: float,
    lambda_lr: float,
    learn_lambda: bool,
    max_lambda: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    num_tasks = len(tasks)
    masks = init_masks(components, num_tasks, init_mode, top_fraction)
    lambda_init = 0.3 if init_mode == "full" else 1.0
    lambdas = nn.Parameter(torch.full((len(components),), lambda_init), requires_grad=learn_lambda)
    opt_groups = [{"params": list(masks.parameters()), "lr": mask_lr}]
    if learn_lambda:
        opt_groups.append({"params": [lambdas], "lr": lambda_lr})
    opt = torch.optim.Adam(opt_groups)
    trace: dict[str, float] = {}
    for step in range(steps):
        opt.zero_grad()
        state = construct_state(components, vector_pack, masks, lambdas, "masked", top_fraction)
        loss = 0.0
        for task, head in zip(tasks, heads):
            logits = head(functional_backbone(task.adapt_x, state))
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
            loss = loss + entropy
        loss.backward()
        opt.step()
        if learn_lambda:
            with torch.no_grad():
                lambdas.clamp_(0.0, max_lambda)
        if step in {0, steps - 1}:
            trace[f"entropy_step_{step}"] = float(loss.detach().item())
    final_state = construct_state(components, vector_pack, masks, lambdas, "masked", top_fraction)
    selected = 0
    total = 0
    with torch.no_grad():
        for param in masks:
            binary = (torch.sigmoid(param / 10.0) >= 0.5)
            selected += int(binary.sum().item())
            total += binary.numel()
    trace["mask_density"] = selected / max(total, 1)
    trace["learnable_mask_params"] = float(total)
    trace["learned_lambda_mean"] = float(lambdas.detach().mean().item())
    return final_state, trace


def parameter_count(backbone: Backbone, heads: list[nn.Linear]) -> int:
    return sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for h in heads for p in h.parameters())


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    tasks = make_tasks(num_tasks=args.tasks, train_per_task=args.train_per_task, adapt_per_task=args.adapt_per_task, test_per_task=args.test_per_task, seed=args.seed)
    pretrained, pretrained_heads = train_pretrained(tasks, args.hidden_dim, args.pretrain_epochs, args.lr)
    task_backbones, task_heads = train_individuals(pretrained, pretrained_heads, tasks, args.hidden_dim, args.finetune_epochs, args.lr)
    components, vector_pack = build_svd_components(pretrained, task_backbones)

    base_lambdas = torch.full((len(components),), 0.3)
    ta_state = construct_state(components, vector_pack, None, base_lambdas, "full", args.top_fraction)
    topk_lambdas = torch.full((len(components),), 1.0)
    topk_state = construct_state(components, vector_pack, None, topk_lambdas, "topk", args.top_fraction)
    adarank_state, trace = adapt_adarank(
        tasks,
        task_heads,
        components,
        vector_pack,
        init_mode=args.init_mode,
        top_fraction=args.top_fraction,
        steps=args.adapt_steps,
        mask_lr=args.adapt_lr,
        lambda_lr=args.lambda_lr,
        learn_lambda=not args.freeze_lambda,
        max_lambda=args.max_lambda,
    )

    methods = {}
    for name, state in [("task_arithmetic", ta_state), ("fixed_topk_svd", topk_state), ("adarank", adarank_state)]:
        acc, preds = evaluate_state(tasks, task_heads, state)
        methods[name] = {"task_accuracy": acc, "predictions": preds}
    acc, preds = evaluate_individuals(tasks, task_backbones, task_heads)
    methods["individual"] = {"task_accuracy": acc, "predictions": preds}

    base_params = parameter_count(pretrained, task_heads)
    all_finetuned_params = base_params * len(tasks)
    for name, payload in methods.items():
        avg = sum(payload["task_accuracy"].values()) / len(tasks)
        payload["avg_accuracy"] = avg
        payload["merged_param_count"] = base_params if name != "individual" else all_finetuned_params
        payload["param_multiplier_vs_single"] = payload["merged_param_count"] / base_params
    methods["adarank"]["adaptation_trace"] = trace
    methods["adarank"]["learnable_param_fraction"] = trace["learnable_mask_params"] / all_finetuned_params

    output = {
        "config": vars(args),
        "experiments": {
            "synthetic_4task_merge": {
                "methods": methods,
                "paper_reference": "AdaRank Table 1/Table 4 style: average multi-task accuracy with constant merged parameter count.",
            }
        },
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.json").open("w") as f:
        json.dump(output, f, indent=2)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"adarank_state": adarank_state, "task_heads": [h.state_dict() for h in task_heads], "config": vars(args)}, ckpt_dir / "adarank_synthetic.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/adarank_synthetic")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--train-per-task", type=int, default=768)
    parser.add_argument("--adapt-per-task", type=int, default=256)
    parser.add_argument("--test-per-task", type=int, default=512)
    parser.add_argument("--pretrain-epochs", type=int, default=120)
    parser.add_argument("--finetune-epochs", type=int, default=180)
    parser.add_argument("--adapt-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adapt-lr", type=float, default=1e-2)
    parser.add_argument("--lambda-lr", type=float, default=1e-3)
    parser.add_argument("--max-lambda", type=float, default=3.0)
    parser.add_argument("--top-fraction", type=float, default=0.16)
    parser.add_argument("--init-mode", choices=["full", "topk"], default="full")
    parser.add_argument(
        "--freeze-lambda",
        dest="freeze_lambda",
        action="store_true",
        default=True,
        help="Keep the Task Arithmetic coefficient fixed and adapt only AdaRank masks.",
    )
    parser.add_argument(
        "--learn-lambda",
        dest="freeze_lambda",
        action="store_false",
        help="Also adapt layer coefficients as in the paper's full default setting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
