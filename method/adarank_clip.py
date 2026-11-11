"""AdaRank (Lee et al., 2025) on the 8-task CLIP ViT-B/32 model-merging benchmark.

Computational steps (paper Sec. 4 and Algorithm 1; official code ``method/official_adarank``):

1. Task vectors tau_i = theta_i - theta_origin. For Task Arithmetic (TA) the origin is
   the pretrained encoder; for CART it is the average of the fine-tuned encoders.
2. Every attention/MLP weight matrix of every task vector is decomposed with SVD.
   As in the official open_clip implementation, the q/k/v projections of a block
   form one stacked ``in_proj`` matrix (2304 x 768), so each block has 4 matrices
   and every matrix has rank 768: 12 * 4 * 768 * 8 = 294,912 mask entries.
3. Merged layer (Eq. 5): theta^l = theta_origin^l + sum_i lambda_i^l U_i diag(B_i (.) S_i) V_i^T.
   Non-matrix parameters (biases, norms, embeddings) are merged as
   theta_origin + sum_i lambda_i^l tau_i^l.
4. B are binary masks optimised with a straight-through estimator:
   b = 1{sigmoid(b~ / T) > 0.5} in the forward pass, gradient through the sigmoid,
   temperature T = 10, b~ initialised to +0.1 (kept) / -0.1 (pruned). TA starts from all
   components kept (= Task Arithmetic, lambda = 0.3); CART starts from the top-16%
   components (lambda = 2.3).
5. B and the task/layer-wise coefficients lambda are adapted jointly (paper default)
   by Adam (lr 1e-3, betas 0.9/0.999) on the summed Shannon entropy of each task's
   predictions on unlabeled inputs, 16 inputs per task per step, 10,000 steps.
6. The final merged encoder is materialised from the final B and lambda.

Modes: ``adarank`` (masks + lambda, default), ``mask_only`` (lambda fixed; Table 6),
``adamerging`` (lambda only, no masks; the AdaMerging baseline) and ``ce_oracle``
(masks + lambda trained with cross-entropy on the labeled dev rows -- the paper's
supervised oracle of Fig. 5; it uses labels and is NOT a valid submission, it is only
used to measure how much the visible labels are worth).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import data.clip_benchmark as cb  # noqa: E402

MATRIX_SUFFIXES = (".self_attn.out_proj.weight", ".mlp.fc1.weight", ".mlp.fc2.weight")
QKV = re.compile(r"^(vision_model\.encoder\.layers\.\d+\.self_attn\.)([qkv])_proj\.(weight|bias)$")


def build_groups(keys: list[str]) -> list[dict]:
    """One merge coefficient per group; q/k/v are stacked like open_clip's in_proj."""
    groups, seen = [], set()
    for key in keys:
        if key in seen:
            continue
        m = QKV.match(key)
        if m:
            prefix, _, kind = m.groups()
            members = [f"{prefix}{x}_proj.{kind}" for x in "qkv"]
            groups.append({"name": f"{prefix}in_proj_{kind}", "members": members, "matrix": kind == "weight"})
            seen.update(members)
        else:
            groups.append({"name": key, "members": [key], "matrix": key.endswith(MATRIX_SUFFIXES)})
            seen.add(key)
    return groups


def ste_mask(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    prob = torch.sigmoid(logits / temperature)
    hard = (prob > 0.5).to(prob.dtype)
    return prob + (hard - prob).detach()


class AdaptiveMerger(torch.nn.Module):
    def __init__(self, origin: dict, experts: list[dict], use_masks: bool, prior: float,
                 init_rank_ratio: float, temperature: float, device: torch.device):
        super().__init__()
        self.use_masks = use_masks
        self.temperature = temperature
        keys = list(cb.canonical_shapes().keys())
        self.groups = build_groups(keys)
        num_tasks = len(experts)
        self.lam = torch.nn.Parameter(torch.full((num_tasks, len(self.groups)), float(prior), device=device))
        self.masks = torch.nn.ParameterList()
        self.tensors: list[dict] = []
        for g in self.groups:
            members = g["members"]
            w0 = torch.cat([origin[m] for m in members], 0).to(device)
            delta = torch.stack([torch.cat([e[m] - origin[m] for m in members], 0) for e in experts]).to(device)
            entry = {"w0": w0, "splits": [origin[m].shape[0] for m in members]}
            if g["matrix"] and use_masks:
                u, s, vh = torch.linalg.svd(delta, full_matrices=False)  # (T,m,r) (T,r) (T,r,n)
                t, m, r = u.shape
                entry["u"] = u.permute(1, 0, 2).reshape(m, t * r).contiguous()
                entry["s"] = s.contiguous()
                entry["vh"] = vh.reshape(t * r, -1).contiguous()
                keep = int(r * init_rank_ratio)
                init = torch.full((t, r), -0.1, device=device)
                init[:, :keep] = 0.1
                entry["mask_index"] = len(self.masks)
                self.masks.append(torch.nn.Parameter(init))
                del delta
            else:
                entry["delta"] = delta
            self.tensors.append(entry)

    def num_mask_entries(self) -> int:
        return sum(int(p.numel()) for p in self.masks)

    def merged_state(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for gi, (g, entry) in enumerate(zip(self.groups, self.tensors)):
            lam = self.lam[:, gi]
            if "u" in entry:
                b = ste_mask(self.masks[entry["mask_index"]], self.temperature)
                coef = (lam[:, None] * entry["s"] * b).reshape(-1)
                w = entry["w0"] + (entry["u"] * coef) @ entry["vh"]
            else:
                w = entry["w0"] + torch.tensordot(lam, entry["delta"], dims=1)
            if len(g["members"]) == 1:
                state[g["members"][0]] = w
            else:
                for name, part in zip(g["members"], w.split(entry["splits"], 0)):
                    state[name] = part
        return state

    @torch.no_grad()
    def mask_summary(self) -> dict:
        if not self.use_masks:
            return {}
        kept, total, ranks = 0, 0, []
        for p in self.masks:
            hard = torch.sigmoid(p / self.temperature) > 0.5
            kept += int(hard.sum())
            total += hard.numel()
            ranks.append(hard.sum(dim=1).float())
        ranks_t = torch.stack(ranks)  # (matrices, tasks)
        return {
            "mask_density": kept / max(total, 1),
            "mean_rank_per_task": [round(float(x), 1) for x in ranks_t.mean(dim=0)],
            "mean_rank_first_block": round(float(ranks_t[:4].mean()), 1),
            "mean_rank_last_block": round(float(ranks_t[-4:].mean()), 1),
        }


class Sampler:
    """Deterministic epoch-wise shuffling, resumable from any step."""

    def __init__(self, sizes: list[int], batch: int, seed: int):
        self.sizes, self.batch, self.seed = sizes, batch, seed
        self.cache: dict[tuple[int, int], torch.Tensor] = {}

    def _perm(self, task: int, epoch: int) -> torch.Tensor:
        key = (task, epoch)
        if key not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + task * 10_007 + epoch)
            self.cache[key] = torch.randperm(self.sizes[task], generator=g)
        return self.cache[key]

    def indices(self, task: int, step: int) -> torch.Tensor:
        n = self.sizes[task]
        out = []
        for s in range(step * self.batch, (step + 1) * self.batch):
            epoch, pos = divmod(s, n)
            out.append(int(self._perm(task, epoch)[pos]))
        return torch.tensor(out, dtype=torch.long)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def run(args: argparse.Namespace) -> Path:
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    torch.manual_seed(args.seed)
    tasks = cb.TASK_NAMES
    out_dir = Path(args.output_root) / args.name
    if (out_dir / "model.safetensors").exists() and not args.overwrite:
        print(f"{out_dir}/model.safetensors exists; skipping")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base = cb.load_base_vision_state()
    experts = [cb.load_expert_state(t) for t in tasks]
    if args.merge == "cart":
        origin = {k: torch.stack([e[k] for e in experts]).mean(dim=0) for k in base}
    else:
        origin = base
    prior = args.prior if args.prior is not None else (2.3 if args.merge == "cart" else 0.3)
    ratio = args.init_rank_ratio if args.init_rank_ratio is not None else (0.16 if args.merge == "cart" else 1.0)
    use_masks = args.mode != "adamerging"
    merger = AdaptiveMerger(origin, experts, use_masks, prior, ratio, args.temperature, device)
    del experts, base, origin
    merger.lam.requires_grad_(args.mode != "mask_only")
    t_setup = time.time() - t0
    print(f"setup {t_setup:.1f}s; groups={len(merger.groups)} mask_entries={merger.num_mask_entries()}", flush=True)

    split = "dev" if args.mode == "ce_oracle" else "adapt"
    pools, labels = [], []
    for t in tasks:
        d = cb.load_split(t, split)
        pools.append(d["images"].to(device))
        labels.append(d["labels"].to(device) if d["labels"] is not None else None)
    sizes = [p.shape[0] for p in pools]
    print("pool sizes", dict(zip(tasks, sizes)), flush=True)

    module = cb.build_vision_module(device)
    heads = cb.load_heads(device)
    params = [p for p in [merger.lam, *merger.masks] if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    sampler = Sampler(sizes, args.batch_size, args.seed)

    ckpt_path = out_dir / "tta_state.pt"
    start = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        merger.lam.data.copy_(ck["lam"])
        for p, v in zip(merger.masks, ck["masks"]):
            p.data.copy_(v)
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed from step {start}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t_loop = time.time()
    history = []
    for step in range(start, args.steps):
        state = merger.merged_state()
        idx = [sampler.indices(i, step) for i in range(len(tasks))]
        x = torch.cat([pools[i][idx[i].to(device)] for i in range(len(tasks))])
        emb = cb.image_embeddings(module, state, cb.normalize(x), heads["projection"])
        loss = 0.0
        per_task = []
        for i, t in enumerate(tasks):
            logits = cb.task_logits(emb[i * args.batch_size:(i + 1) * args.batch_size], heads, t)
            if args.mode == "ce_oracle":
                lt = F.cross_entropy(logits, labels[i][idx[i].to(device)])
            else:
                lt = entropy(logits).mean()
            per_task.append(float(lt.detach()))
            loss = loss + lt
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            el = time.time() - t_loop
            rate = el / max(step - start + 1, 1)
            print(f"step {step} loss {float(loss):.4f} per_task {[round(v, 3) for v in per_task]} "
                  f"{rate:.3f}s/step eta {(args.steps - step - 1) * rate / 60:.1f}min", flush=True)
            history.append({"step": step, "loss": float(loss)})
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0 and step + 1 < args.steps:
            torch.save({"step": step + 1, "lam": merger.lam.detach(), "masks": [p.detach() for p in merger.masks],
                        "opt": opt.state_dict()}, ckpt_path)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with torch.no_grad():
        final = {k: v.detach().float().cpu() for k, v in merger.merged_state().items()}
    lam = merger.lam.detach().cpu()
    meta = {
        "method": "AdaRank" if args.mode == "adarank" else args.mode,
        "merge_base": args.merge,
        "mode": args.mode,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size_per_task": args.batch_size,
        "temperature": args.temperature,
        "prior_lambda": prior,
        "init_rank_ratio": ratio,
        "seed": args.seed,
        "inputs": split,
        "pool_sizes": dict(zip(tasks, sizes)),
        "learnable_mask_entries": merger.num_mask_entries() if args.mode != "adamerging" else 0,
        "learnable_lambdas": int(lam.numel()) if args.mode != "mask_only" else 0,
        "lambda_mean": float(lam.mean()),
        "lambda_min": float(lam.min()),
        "lambda_max": float(lam.max()),
        **merger.mask_summary(),
        "setup_seconds": round(t_setup, 1),
        "loop_seconds": round(time.time() - t_loop, 1),
        "history": history,
        "note": "informational only; the evaluator ignores this file",
    }
    cb.save_submission(final, args.name, meta=meta, root=args.output_root)
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}, indent=1), flush=True)
    return out_dir


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", default="proposed", help="submission directory name under --output-root")
    p.add_argument("--output-root", default=str(cb.SUBMISSION_ROOT))
    p.add_argument("--merge", choices=["ta", "cart"], default="ta")
    p.add_argument("--mode", choices=["adarank", "mask_only", "adamerging", "ce_oracle"], default="adarank")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument("--prior", type=float, default=None, help="initial lambda (TA 0.3, CART 2.3)")
    p.add_argument("--init-rank-ratio", type=float, default=None, help="initially kept fraction (TA 1.0, CART 0.16)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
