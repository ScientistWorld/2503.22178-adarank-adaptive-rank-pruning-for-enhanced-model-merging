"""Brute-force check: grid-search the Task Arithmetic coefficient on the visible labeled dev rows.

Uses only public data (``data/``). Writes the best encoder to
``outputs/variants/ta_lambda_grid_dev`` so it can be pushed through the hidden test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import data.clip_benchmark as cb  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = cb.pick_device() if args.device == "auto" else torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    base = cb.load_base_vision_state()
    total = {k: torch.zeros_like(v) for k, v in base.items()}
    for t in cb.TASK_NAMES:
        e = cb.load_expert_state(t)
        for k in total:
            total[k] += e[k] - base[k]
    module = cb.build_vision_module(device)
    heads = cb.load_heads(device)
    dev = {t: cb.load_split(t, "dev") for t in cb.TASK_NAMES}
    grid = [round(0.1 * i, 2) for i in range(1, 11)]
    table = {}
    for lam in grid:
        state = {k: base[k] + lam * total[k] for k in base}
        accs = []
        for t in cb.TASK_NAMES:
            pred = cb.predict(module, state, dev[t]["images"], heads, t, device)
            accs.append(float((pred == dev[t]["labels"]).float().mean()))
        table[lam] = sum(accs) / len(accs)
        print(f"lambda {lam}: dev avg acc {table[lam]:.4f}", flush=True)
    best = max(table, key=table.get)
    state = {k: base[k] + best * total[k] for k in base}
    cb.save_submission(state, "ta_lambda_grid_dev", meta={"method": "task arithmetic, lambda tuned on dev labels",
                                                            "lambda": best, "dev_table": table},
                       root=ROOT / "outputs" / "variants")
    print(json.dumps({"best_lambda": best, "table": table}))


if __name__ == "__main__":
    main()
