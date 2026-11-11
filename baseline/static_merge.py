"""Static (data-free) merging baselines written in the submission format.

* ``task_arithmetic``: theta_0 + 0.3 * sum_i tau_i (Ilharco et al., 2023)
* ``fixed_topk_svd``: like Task Arithmetic, but every attention/MLP weight matrix of each
  task vector is truncated to its top-16% singular components before summing.
* ``pretrained``: the unmerged pretrained encoder.

The scored baselines are recomputed independently by the evaluators; this script
exists to show the submission format and to sanity-check a pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data.clip_benchmark as cb  # noqa: E402


def merge(name: str, lam: float = 0.3, fraction: float = 0.16) -> dict[str, torch.Tensor]:
    base = cb.load_base_vision_state()
    if name == "pretrained":
        return base
    experts = [cb.load_expert_state(t) for t in cb.TASK_NAMES]
    state = {k: base[k] + lam * sum(e[k] - base[k] for e in experts) for k in base}
    if name == "task_arithmetic":
        return state
    layers = sum(1 for k in base if k.endswith("mlp.fc1.weight"))
    for layer in range(layers):
        pre = f"vision_model.encoder.layers.{layer}."
        for members in ([f"{pre}self_attn.{x}_proj.weight" for x in "qkv"], [f"{pre}self_attn.out_proj.weight"],
                        [f"{pre}mlp.fc1.weight"], [f"{pre}mlp.fc2.weight"]):
            w0 = torch.cat([base[m] for m in members], 0)
            acc = torch.zeros_like(w0)
            for e in experts:
                u, s, vh = torch.linalg.svd(torch.cat([e[m] for m in members], 0) - w0, full_matrices=False)
                k = max(1, int(s.numel() * fraction))
                acc += (u[:, :k] * s[:k]) @ vh[:k]
            merged = w0 + lam * acc
            for m, part in zip(members, merged.split([base[m].shape[0] for m in members], 0)):
                state[m] = part
    return state


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", choices=["task_arithmetic", "fixed_topk_svd", "pretrained"], default="task_arithmetic")
    p.add_argument("--output-root", default=str(ROOT / "outputs" / "baselines"))
    args = p.parse_args()
    out = cb.save_submission(merge(args.name), args.name, meta={"method": args.name}, root=args.output_root)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
