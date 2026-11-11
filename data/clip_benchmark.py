"""Public, method-agnostic utilities for the 8-task CLIP ViT-B/32 merging benchmark.

Everything here is shared by solution code (method/, baseline/) and describes the
fixed problem: the task list, the pretrained and fine-tuned image encoders, the
fixed zero-shot classification heads, the image preprocessing, the public
adaptation/dev splits, and the submission format.

Submission format
-----------------
A submission is a directory ``outputs/merged/<name>/`` containing
``model.safetensors`` with exactly the parameters of one CLIP ViT-B/32 image
encoder (the ``vision_model.*`` tensors of ``transformers.CLIPVisionModel``,
same names and shapes as the pretrained encoder), all finite. The evaluator
attaches the fixed visual projection and the fixed per-task zero-shot heads.
"""

from __future__ import annotations

import io
import functools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOWNLOADS = DATA_DIR / "downloads"
MODELS_DIR = DOWNLOADS / "models"
BASE_DIR = MODELS_DIR / "clip-vit-base-patch32"
PREPARED_DIR = DOWNLOADS / "prepared"
ASSETS_DIR = DATA_DIR / "assets"
SUBMISSION_ROOT = ROOT / "outputs" / "merged"

MANIFEST = json.loads((DATA_DIR / "tasks.json").read_text())
TASK_NAMES: list[str] = [t["name"] for t in MANIFEST["tasks"]]
NUM_CLASSES: dict[str, int] = {t["name"]: t["num_classes"] for t in MANIFEST["tasks"]}

IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# --------------------------------------------------------------------------- images
def preprocess_image_bytes(raw: bytes) -> torch.Tensor:
    """CLIP preprocessing: RGB, bicubic resize of the short side to 224, center crop.

    Returns a uint8 tensor of shape (3, 224, 224). Normalisation happens in
    :func:`normalize`.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    if w <= h:
        nw, nh = IMAGE_SIZE, int(IMAGE_SIZE * h / w)
    else:
        nw, nh = int(IMAGE_SIZE * w / h), IMAGE_SIZE
    img = img.resize((nw, nh), Image.BICUBIC)
    left = int(round((nw - IMAGE_SIZE) / 2.0))
    top = int(round((nh - IMAGE_SIZE) / 2.0))
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def preprocess_many(raw_list: list[bytes]) -> torch.Tensor:
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 1
    workers = max(1, min(16, ncpu))
    with ThreadPoolExecutor(workers) as pool:
        tensors = list(pool.map(preprocess_image_bytes, raw_list))
    return torch.stack(tensors) if tensors else torch.zeros(0, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)


def normalize(images_u8: torch.Tensor) -> torch.Tensor:
    x = images_u8.float().div_(255.0)
    mean = torch.tensor(CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def load_split(task: str, split: str) -> dict:
    """Load a public split: ``adapt`` (unlabeled inputs) or ``dev`` (inputs + labels).

    Returns ``{"rows": [...], "images": uint8 (N,3,224,224), "labels": LongTensor | None}``.
    Decoded images are cached next to the prepared file.
    """
    if split not in {"adapt", "dev"}:
        raise ValueError("public splits are 'adapt' and 'dev'")
    src = PREPARED_DIR / f"{split}_{task}.pt"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing: run scripts/download.sh and data/prepare_data.py")
    payload = torch.load(src, map_location="cpu", weights_only=False)
    cache = PREPARED_DIR / f"{split}_{task}.u8.pt"
    images = None
    if cache.exists():
        cached = torch.load(cache, map_location="cpu", weights_only=False)
        if cached.get("rows") == payload["rows"]:
            images = cached["images"]
    if images is None:
        images = preprocess_many(payload["image_bytes"])
        tmp = cache.with_suffix(f".tmp{os.getpid()}")
        torch.save({"rows": payload["rows"], "images": images}, tmp)
        os.replace(tmp, cache)
    labels = payload.get("labels")
    return {
        "rows": payload["rows"],
        "images": images,
        "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
    }


# --------------------------------------------------------------------------- models
@functools.lru_cache(maxsize=1)
def vision_config():
    from transformers import CLIPConfig

    return CLIPConfig.from_pretrained(str(BASE_DIR)).vision_config


def build_vision_module(device: str | torch.device = "cpu"):
    """An (unused-weight) CLIPVisionModel used as a functional template."""
    from transformers import CLIPVisionModel

    module = CLIPVisionModel(vision_config()).to(device)
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


@functools.lru_cache(maxsize=1)
def _canonical_shapes() -> tuple:
    module = build_vision_module("cpu")
    return tuple((k, tuple(v.shape)) for k, v in module.named_parameters())


def canonical_shapes() -> dict[str, tuple[int, ...]]:
    return dict(_canonical_shapes())


def load_base_full() -> dict[str, torch.Tensor]:
    return torch.load(BASE_DIR / "pytorch_model.bin", map_location="cpu", weights_only=True)


def load_base_vision_state() -> dict[str, torch.Tensor]:
    """Pretrained image-encoder weights, restricted to the canonical parameter names."""
    full = load_base_full()
    shapes = canonical_shapes()
    state = {k: full[k].float().clone() for k in shapes}
    return state


def load_expert_state(task: str) -> dict[str, torch.Tensor]:
    """Fine-tuned image-encoder weights of one task (canonical parameter names)."""
    from safetensors.torch import load_file

    raw = load_file(str(MODELS_DIR / "experts" / task / "model.safetensors"))
    shapes = canonical_shapes()
    missing = [k for k in shapes if k not in raw]
    if missing:
        raise KeyError(f"expert {task} misses {missing[:3]}")
    return {k: raw[k].float().clone() for k in shapes}


def load_heads(device: str | torch.device = "cpu") -> dict:
    """Fixed zero-shot heads: ``{"heads": {task: (C,512)}, "projection": (512,768), "logit_scale": float}``."""
    from safetensors.torch import load_file

    bundle = load_file(str(ASSETS_DIR / "zeroshot_heads.safetensors"))
    heads = {t: bundle[f"head.{t}"].to(device) for t in TASK_NAMES}
    return {
        "heads": heads,
        "projection": bundle["visual_projection"].to(device),
        "logit_scale": float(bundle["logit_scale"].item()),
    }


def image_embeddings(module, state: dict[str, torch.Tensor], pixels: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """L2-normalised image embeddings of ``pixels`` under the encoder weights ``state``."""
    from torch.func import functional_call

    tensors = dict(state)
    for name, buf in module.named_buffers():
        tensors.setdefault(name, buf)
    out = functional_call(module, tensors, (pixels,), strict=True)
    emb = out.pooler_output @ projection.t()
    return F.normalize(emb, dim=-1)


def task_logits(embeddings: torch.Tensor, heads: dict, task: str) -> torch.Tensor:
    return heads["logit_scale"] * embeddings @ heads["heads"][task].t()


# --------------------------------------------------------------------------- submissions
def check_state(state: dict[str, torch.Tensor]) -> None:
    shapes = canonical_shapes()
    if set(state) != set(shapes):
        extra = sorted(set(state) - set(shapes))[:3]
        missing = sorted(set(shapes) - set(state))[:3]
        raise ValueError(f"state keys differ from the encoder (extra={extra}, missing={missing})")
    for k, v in state.items():
        if tuple(v.shape) != shapes[k]:
            raise ValueError(f"{k}: shape {tuple(v.shape)} != {shapes[k]}")
        if not torch.isfinite(v).all():
            raise ValueError(f"{k}: non-finite values")


def save_submission(state: dict[str, torch.Tensor], name: str, meta: dict | None = None,
                    root: Path | str = SUBMISSION_ROOT) -> Path:
    """Write ``<root>/<name>/model.safetensors`` (+ optional informational meta.json)."""
    from safetensors.torch import save_file

    check_state(state)
    out_dir = Path(root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().float().cpu().contiguous() for k, v in state.items()}, str(out_dir / "model.safetensors"))
    if meta is not None:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


@torch.no_grad()
def predict(module, state: dict[str, torch.Tensor], images_u8: torch.Tensor, heads: dict, task: str,
            device: str | torch.device, batch_size: int = 256) -> torch.Tensor:
    preds = []
    st = {k: v.to(device) for k, v in state.items()}
    for i in range(0, images_u8.shape[0], batch_size):
        x = normalize(images_u8[i:i + batch_size].to(device))
        emb = image_embeddings(module, st, x, heads["projection"])
        preds.append(task_logits(emb, heads, task).argmax(dim=1).cpu())
    return torch.cat(preds)


def pick_device() -> torch.device:
    """Visible CUDA device with the most free memory (shared nodes may expose busy GPUs)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [(torch.cuda.mem_get_info(i)[0], -i) for i in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{-max(free)[1]}")
