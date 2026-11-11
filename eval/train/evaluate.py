"""Visible dev-slice scorer for the 8-task image-encoder merging benchmark.

Same interface and metrics as the held-out scorer, on the public labeled ``dev`` rows
(``data/downloads/prepared/dev_<task>.pt``). Owns its own copy of the fixed heads,
encoder spec and baseline predictions (``eval/train/assets/``).

Every directory ``outputs/merged/<name>/`` is a submission and is scored; the
``proposed`` entry is always written, with the worst value of every metric when it is
missing or invalid. Only ``model.safetensors`` is read from a submission.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ASSETS = HERE / "assets"
SUBMISSIONS = ROOT / "outputs" / "merged"
EXPERIMENT = "vit_b32_8task_merge"
REQUIRED_SUBMISSIONS = ("proposed",)
BASELINES = ("pretrained", "task_arithmetic", "fixed_topk_svd", "individual")
WORST = {"avg_accuracy": 0.0, "balanced_accuracy": 0.0, "worst_task_accuracy": 0.0}
TA_LAMBDA = 0.3
TOPK_FRACTION = 0.16
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class SubmissionError(Exception):
    pass


def cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def pick_device() -> torch.device:
    """Visible CUDA device with the most free memory (nodes may expose busy GPUs), else CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [(torch.cuda.mem_get_info(i)[0], -i) for i in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{-max(free)[1]}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- data
def preprocess(raw: bytes) -> torch.Tensor:
    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = 224, int(224 * h / w)
    else:
        nw, nh = int(224 * w / h), 224
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = int(round((nw - 224) / 2.0)), int(round((nh - 224) / 2.0))
    img = img.crop((left, top, left + 224, top + 224))
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()


def load_dev(task: str, index: dict) -> tuple[torch.Tensor, torch.Tensor]:
    src = ROOT / "data" / "downloads" / "prepared" / f"dev_{task}.pt"
    if not src.exists():
        raise SystemExit(f"{src} missing: run scripts/download.sh, then python data/prepare_data.py")
    payload = torch.load(src, map_location="cpu", weights_only=False)
    if payload["rows"] != index[task]["rows"]:
        raise SystemExit(f"dev rows of {task} do not match data/splits.json")
    hashes = [sha256_bytes(b) for b in payload["image_bytes"]]
    cache = ROOT / "data" / "downloads" / "prepared" / f"evalcache_dev_{task}.pt"
    images = None
    if cache.exists():
        c = torch.load(cache, map_location="cpu", weights_only=False)
        if c.get("image_sha256") == hashes:
            images = c["images"]
    if images is None:
        with ThreadPoolExecutor(min(16, cpu_count())) as pool:
            images = torch.stack(list(pool.map(preprocess, payload["image_bytes"])))
        torch.save({"image_sha256": hashes, "images": images}, cache)
    return images, torch.tensor(list(payload["labels"]), dtype=torch.long)


# --------------------------------------------------------------------------- model
def load_assets(device):
    from safetensors.torch import load_file
    from transformers import CLIPVisionConfig, CLIPVisionModel

    cfg = json.loads((ASSETS / "vision_config.json").read_text())
    shapes = {k: tuple(v) for k, v in json.loads((ASSETS / "param_shapes.json").read_text()).items()}
    module = CLIPVisionModel(CLIPVisionConfig(**cfg)).to(device).eval()
    if {k: tuple(v.shape) for k, v in module.named_parameters()} != shapes:
        raise SystemExit("installed CLIPVisionModel does not match the recorded encoder spec")
    bundle = load_file(str(ASSETS / "zeroshot_heads.safetensors"))
    return module, shapes, bundle


def encode(module, state: dict, images_u8: torch.Tensor, projection: torch.Tensor, device, bs: int = 256) -> torch.Tensor:
    from torch.func import functional_call

    tensors = dict(state)
    for name, buf in module.named_buffers():
        tensors.setdefault(name, buf)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, images_u8.shape[0], bs):
            x = (images_u8[i:i + bs].to(device).float() / 255.0 - mean) / std
            pooled = functional_call(module, tensors, (x,), strict=True).pooler_output
            out.append(F.normalize(pooled @ projection.t(), dim=-1))
    return torch.cat(out)


def predict_all(module, state_for_task, data, bundle, device) -> dict[str, list[int]]:
    projection = bundle["visual_projection"].to(device)
    scale = float(bundle["logit_scale"].reshape(-1)[0])
    preds = {}
    on_device: dict = {}
    for task, (images, _) in data.items():
        key, state = state_for_task(task)
        if key not in on_device:
            on_device.clear()
            on_device[key] = {k: v.to(device) for k, v in state.items()}
        emb = encode(module, on_device[key], images, projection, device)
        logits = scale * emb @ bundle[f"head.{task}"].to(device).t()
        preds[task] = logits.argmax(dim=1).cpu().tolist()
    return preds


def read_submission(path: Path, shapes: dict) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    f = path / "model.safetensors"
    if not f.is_file():
        raise SubmissionError("model.safetensors missing")
    try:
        state = {}
        with safe_open(str(f), framework="pt") as fh:
            keys = list(fh.keys())
            if set(keys) != set(shapes):
                extra = sorted(set(keys) - set(shapes))[:3]
                missing = sorted(set(shapes) - set(keys))[:3]
                raise SubmissionError(f"tensor names differ from one image encoder (extra={extra}, missing={missing})")
            for k in keys:
                state[k] = fh.get_tensor(k)
    except SubmissionError:
        raise
    except Exception as exc:  # unreadable / corrupt file
        raise SubmissionError(f"cannot read model.safetensors: {type(exc).__name__}: {exc}") from None
    for k, v in state.items():
        if tuple(v.shape) != shapes[k]:
            raise SubmissionError(f"{k}: shape {tuple(v.shape)} != {shapes[k]}")
        if not v.is_floating_point():
            raise SubmissionError(f"{k}: dtype {v.dtype} is not floating point")
        v = v.float()
        if not torch.isfinite(v).all():
            raise SubmissionError(f"{k}: non-finite values")
        state[k] = v
    return state


# --------------------------------------------------------------------------- metrics
def metrics(preds: dict[str, list[int]], data: dict) -> tuple[dict, dict]:
    accs, bals = {}, {}
    for task, (_, labels) in data.items():
        p = torch.tensor(preds[task], dtype=torch.long)
        if p.shape != labels.shape:
            raise SubmissionError(f"prediction count mismatch for {task}")
        correct = (p == labels).float()
        accs[task] = float(correct.mean())
        recalls = [float(correct[labels == c].mean()) for c in torch.unique(labels).tolist()]
        bals[task] = sum(recalls) / len(recalls)
    result = {
        "avg_accuracy": sum(accs.values()) / len(accs),
        "balanced_accuracy": sum(bals.values()) / len(bals),
        "worst_task_accuracy": min(accs.values()),
    }
    for k, v in result.items():
        if not (0.0 <= v <= 1.0):
            raise SubmissionError(f"impossible {k}={v}")
    return result, accs


# --------------------------------------------------------------------------- baselines
def rebuild_baselines(module, shapes, bundle, data, index, device) -> dict:
    """Recompute the fixed baselines from hash-verified public checkpoints (one-off)."""
    from safetensors.torch import load_file

    src = json.loads((ASSETS / "source_hashes.json").read_text())
    models = ROOT / "data" / "downloads" / "models"
    base_file = models / "clip-vit-base-patch32" / "pytorch_model.bin"
    if sha256_file(base_file) != src["base"]:
        raise SystemExit("pretrained checkpoint hash mismatch")
    full = torch.load(base_file, map_location="cpu", weights_only=True)
    base = {k: full[k].float() for k in shapes}
    tasks = list(data)
    experts = {}
    for t in tasks:
        f = models / "experts" / t / "model.safetensors"
        if sha256_file(f) != src["experts"][t]:
            raise SystemExit(f"fine-tuned checkpoint hash mismatch for {t}")
        raw = load_file(str(f))
        experts[t] = {k: raw[k].float() for k in shapes}
    ta = {k: base[k] + TA_LAMBDA * sum(experts[t][k] - base[k] for t in tasks) for k in shapes}
    topk = dict(ta)
    for layer in range(int(json.loads((ASSETS / "vision_config.json").read_text())["num_hidden_layers"])):
        pre = f"vision_model.encoder.layers.{layer}."
        for members in ([f"{pre}self_attn.{x}_proj.weight" for x in "qkv"], [f"{pre}self_attn.out_proj.weight"],
                        [f"{pre}mlp.fc1.weight"], [f"{pre}mlp.fc2.weight"]):
            w0 = torch.cat([base[m] for m in members], 0)
            acc = torch.zeros_like(w0)
            for t in tasks:
                d = (torch.cat([experts[t][m] for m in members], 0) - w0).to(device).double()
                u, s, vh = torch.linalg.svd(d, full_matrices=False)
                k = max(1, int(s.numel() * TOPK_FRACTION))
                acc += ((u[:, :k] * s[:k]) @ vh[:k]).float().cpu()
            merged = w0 + TA_LAMBDA * acc
            for m, part in zip(members, merged.split([base[m].shape[0] for m in members], 0)):
                topk[m] = part
    specs = {
        "pretrained": lambda t: ("base", base),
        "task_arithmetic": lambda t: ("ta", ta),
        "fixed_topk_svd": lambda t: ("topk", topk),
        "individual": lambda t: (t, experts[t]),
    }
    out = {"rows_sha256": {t: sha256_bytes(json.dumps(index[t]["rows"]).encode()) for t in tasks}, "predictions": {}}
    for name, fn in specs.items():
        out["predictions"][name] = predict_all(module, fn, data, bundle, device)
        print(f"baseline {name}: {metrics(out['predictions'][name], data)[0]}", flush=True)
    (ASSETS / "baseline_predictions.json").write_text(json.dumps(out) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "scoring" / "scores_train.json"))
    parser.add_argument("--rebuild-baselines", action="store_true")
    parser.add_argument("--submissions", default=str(SUBMISSIONS))
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(cpu_count())
    device = pick_device()
    t0 = time.time()
    splits = json.loads((ROOT / "data" / "splits.json").read_text())["tasks"]
    index = {t: {"rows": splits[t]["dev"]} for t in splits}
    module, shapes, bundle = load_assets(device)
    data = {t: load_dev(t, index) for t in index}
    print(f"loaded dev data ({sum(v[1].numel() for v in data.values())} rows) in {time.time() - t0:.0f}s", flush=True)

    if args.rebuild_baselines:
        baseline = rebuild_baselines(module, shapes, bundle, data, index, device)
    else:
        baseline = json.loads((ASSETS / "baseline_predictions.json").read_text())
    for t in index:
        if baseline["rows_sha256"][t] != sha256_bytes(json.dumps(index[t]["rows"]).encode()):
            raise SystemExit("baseline predictions were computed on different rows")

    results, per_task, failures = {}, {}, {}
    for name in BASELINES:
        results[name], per_task[name] = metrics(baseline["predictions"][name], data)

    sub_root = Path(args.submissions)
    names = set(REQUIRED_SUBMISSIONS)
    if sub_root.is_dir():
        names |= {p.name for p in sub_root.iterdir() if p.is_dir() and not p.name.startswith(".")}
    for name in sorted(names):
        key = name
        while key in BASELINES or key in results:
            # never overwrite a fixed baseline or an entry that was already scored
            key = "submission_" + key
        if key != name:
            failures[key] = f"submission name {name!r} collides with an existing entry; scored as {key!r}"
        try:
            state = read_submission(sub_root / name, shapes)
            preds = predict_all(module, lambda t, s=state: ("sub", s), data, bundle, device)
            results[key], per_task[key] = metrics(preds, data)
        except SubmissionError as exc:
            results[key] = dict(WORST)
            failures[key] = str(exc)
        print(f"{key}: {results[key]}" + (f"  FAILED: {failures[key]}" if key in failures else ""), flush=True)

    scores = {
        "experiments": {EXPERIMENT: {"results": results}},
        "slice": "train",
        "per_task_accuracy": per_task,
        "failures": failures,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2) + "\n")
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
