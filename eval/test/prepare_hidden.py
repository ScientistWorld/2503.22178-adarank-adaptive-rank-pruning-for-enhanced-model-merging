"""Build the hidden evaluation split from the downloaded public test-split parquet files.

Reads the rows listed in ``assets/hidden_rows.json`` (disjoint from the public
adaptation/dev rows) from ``data/downloads/raw/`` and writes
``downloads/hidden_<task>.pt`` (encoded image bytes + labels). On first build it
records per-image sha256 digests and a label digest in ``assets/hidden_index.json``;
afterwards every rebuild is verified against that index. Optionally copies the
encoder/head assets built by ``data/build_heads.py`` (``--copy-assets``) and records
the public checkpoint digests used to rebuild the baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ASSETS = HERE / "assets"
DOWNLOADS = HERE / "downloads"
RAW = ROOT / "data" / "downloads" / "raw"

# Public test-split files (pinned revisions) in row order; identical to the public manifest.
SOURCES = {
    "sun397": ([f"test-{i:05d}-of-00018.parquet" for i in range(18)], 19850),
    "cars": ([f"test-{i:05d}-of-00002.parquet" for i in range(2)], 8041),
    "resisc45": (["test-00000-of-00001.parquet"], 6300),
    "eurosat": (["test-00000-of-00001.parquet"], 2700),
    "svhn": (["test-00000-of-00001.parquet"], 26032),
    "gtsrb": (["test-00000-of-00001.parquet"], 12630),
    "mnist": (["test-00000-of-00001.parquet"], 10000),
    "dtd": (["test-00000-of-00001.parquet"], 1880),
}


def extract_rows(files: list[Path], wanted: set[int], expected_rows: int) -> dict[int, tuple[bytes, int]]:
    import pyarrow.parquet as pq

    out: dict[int, tuple[bytes, int]] = {}
    offset = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            local = [r - offset for r in wanted if offset <= r < offset + n]
            if local:
                table = pf.read_row_group(rg, columns=["image", "label"])
                images, labels = table.column("image"), table.column("label")
                for i in local:
                    img = images[i].as_py()
                    raw = img["bytes"] if isinstance(img, dict) else img
                    out[offset + i] = (bytes(raw), int(labels[i].as_py()))
            offset += n
    if offset != expected_rows or len(out) != len(wanted):
        raise SystemExit(f"unexpected parquet contents in {files[0].parent}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-assets", default=None, help="directory produced by data/build_heads.py")
    args = parser.parse_args()
    rows_spec = json.loads((ASSETS / "hidden_rows.json").read_text())["tasks"]
    index_path = ASSETS / "hidden_index.json"
    index = json.loads(index_path.read_text())["tasks"] if index_path.exists() else {}
    DOWNLOADS.mkdir(parents=True, exist_ok=True)

    new_index = {}
    for task, (files, n_rows) in SOURCES.items():
        rows = rows_spec[task]["rows"]
        dest = DOWNLOADS / f"hidden_{task}.pt"
        if dest.exists() and task in index:
            new_index[task] = index[task]
            print(f"{task}: hidden split exists")
            continue
        paths = [RAW / task / f for f in files]
        if not all(p.exists() for p in paths):
            raise SystemExit(f"{task}: raw parquet files missing under {RAW / task}; run scripts/download.sh")
        got = extract_rows(paths, set(rows), n_rows)
        image_bytes = [got[r][0] for r in rows]
        labels = [got[r][1] for r in rows]
        entry = {
            "rows": rows,
            "image_sha256": [hashlib.sha256(b).hexdigest() for b in image_bytes],
            "label_sha256": hashlib.sha256(json.dumps(labels).encode()).hexdigest(),
        }
        if task in index and index[task] != entry:
            raise SystemExit(f"{task}: rebuilt hidden split differs from the recorded index")
        torch.save({"task": task, "rows": rows, "image_bytes": image_bytes, "labels": labels}, dest)
        new_index[task] = entry
        print(f"{task}: wrote {dest.relative_to(ROOT)} ({len(rows)} rows)")
    if not index:
        index_path.write_text(json.dumps({"tasks": new_index}) + "\n")
        print(f"recorded {index_path.relative_to(ROOT)}")

    if args.copy_assets:
        src = Path(args.copy_assets)
        for name in ("zeroshot_heads.safetensors", "vision_config.json", "param_shapes.json"):
            shutil.copy2(src / name, ASSETS / name)
        print("copied encoder/head assets")


if __name__ == "__main__":
    main()
