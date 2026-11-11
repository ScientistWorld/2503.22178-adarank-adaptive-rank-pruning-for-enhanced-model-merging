"""Download the public benchmark files listed in data/tasks.json (stdlib only).

Fetches, from pinned Hugging Face revisions:
  * the pretrained CLIP ViT-B/32 checkpoint and tokenizer files,
  * the eight task-specific fine-tuned image encoders,
  * the public test-split parquet files of the eight image-classification tasks
    (only when the prepared splits do not exist yet; they are deleted again by
    ``data/prepare_data.py --scrub-raw`` after the splits are built).

Every LFS file is checked against the sha256 recorded in data/tasks.json.
Idempotent: files that already exist with the right digest are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = ROOT / "data" / "downloads"
MODELS = DOWNLOADS / "models"
RAW = DOWNLOADS / "raw"
PREPARED = DOWNLOADS / "prepared"
HF = "https://huggingface.co"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, sha256: str | None) -> None:
    if dest.exists() and (sha256 is None or sha256_file(dest) == sha256):
        print(f"  ok   {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            print(f"  get  {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "benchmark-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 22)
            if sha256 is not None and sha256_file(tmp) != sha256:
                raise IOError(f"sha256 mismatch for {url}")
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001 - retry any network error
            print(f"  retry {attempt + 1}/5 after error: {exc}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"failed to download {url}")


def prepared_complete(manifest: dict) -> bool:
    names = [t["name"] for t in manifest["tasks"]]
    need = [PREPARED / f"{split}_{n}.pt" for n in names for split in ("adapt", "dev")]
    hidden_prep = ROOT / "eval" / "test" / "prepare_hidden.py"
    if hidden_prep.exists():
        need += [ROOT / "eval" / "test" / "downloads" / f"hidden_{n}.pt" for n in names]
    return all(p.exists() for p in need)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--force-raw", action="store_true", help="fetch raw parquet files even if prepared splits exist")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data" / "tasks.json").read_text())

    base = manifest["base_model"]
    print("pretrained CLIP:", base["repo"])
    for fname in base["files"]:
        fetch(f"{HF}/{base['repo']}/resolve/{base['revision']}/{fname}",
              MODELS / "clip-vit-base-patch32" / fname, base["sha256"].get(fname))

    for task in manifest["tasks"]:
        print("fine-tuned encoder:", task["expert_repo"])
        for fname in task["expert_files"]:
            fetch(f"{HF}/{task['expert_repo']}/resolve/{task['expert_revision']}/{fname}",
                  MODELS / "experts" / task["name"] / fname, task["expert_sha256"].get(fname))

    if args.models_only:
        return
    if prepared_complete(manifest) and not args.force_raw:
        print("prepared splits already exist; skipping raw test-split download")
        return
    for task in manifest["tasks"]:
        print("public test split:", task["dataset_repo"])
        for fname, digest in zip(task["test_files"], task["test_file_sha256"]):
            fetch(f"{HF}/datasets/{task['dataset_repo']}/resolve/{task['dataset_revision']}/{fname}",
                  RAW / task["name"] / os.path.basename(fname), digest)


if __name__ == "__main__":
    main()
