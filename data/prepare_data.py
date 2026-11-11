"""Build the public adaptation and dev splits from the downloaded test-split parquet files.

Writes ``data/downloads/prepared/{adapt,dev}_<task>.pt`` with the encoded image
bytes of the rows listed in ``data/splits.json`` (and labels for ``dev`` only).
Row ids follow the order of the parquet files listed in ``data/tasks.json``.

``--scrub-raw`` deletes ``data/downloads/raw/`` once every prepared file (and,
when the evaluation side is present, its own prepared files) exists, so that no
label beyond the public dev rows stays in the workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "downloads" / "raw"
PREPARED = DATA / "downloads" / "prepared"


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
                images = table.column("image")
                labels = table.column("label")
                for i in local:
                    img = images[i].as_py()
                    raw = img["bytes"] if isinstance(img, dict) else img
                    out[offset + i] = (bytes(raw), int(labels[i].as_py()))
            offset += n
    if offset != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {offset} in {files[0].parent}")
    if len(out) != len(wanted):
        raise ValueError("some requested rows were not found")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrub-raw", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((DATA / "tasks.json").read_text())
    splits = json.loads((DATA / "splits.json").read_text())["tasks"]
    PREPARED.mkdir(parents=True, exist_ok=True)

    for task in manifest["tasks"]:
        name = task["name"]
        targets = {s: PREPARED / f"{s}_{name}.pt" for s in ("adapt", "dev")}
        if all(p.exists() for p in targets.values()):
            print(f"{name}: prepared splits exist")
            continue
        files = [RAW / name / Path(f).name for f in task["test_files"]]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"{name}: raw files missing ({missing[:2]}); run scripts/download.sh")
        rows = {s: list(splits[name][s]) for s in ("adapt", "dev")}
        got = extract_rows(files, set(rows["adapt"]) | set(rows["dev"]), task["num_test_rows"])
        for s, path in targets.items():
            payload = {
                "task": name,
                "split": s,
                "rows": rows[s],
                "image_bytes": [got[r][0] for r in rows[s]],
                "image_sha256": [hashlib.sha256(got[r][0]).hexdigest() for r in rows[s]],
            }
            if s == "dev":
                payload["labels"] = [got[r][1] for r in rows[s]]
            torch.save(payload, path)
            print(f"{name}: wrote {path.relative_to(ROOT)} ({len(rows[s])} rows)")

    if args.scrub_raw:
        names = [t["name"] for t in manifest["tasks"]]
        need = [PREPARED / f"{s}_{n}.pt" for n in names for s in ("adapt", "dev")]
        hidden_prep = ROOT / "eval" / "test" / "prepare_hidden.py"
        if hidden_prep.exists():
            need += [ROOT / "eval" / "test" / "downloads" / f"hidden_{n}.pt" for n in names]
        if all(p.exists() for p in need):
            if RAW.exists():
                shutil.rmtree(RAW)
                print("deleted data/downloads/raw/")
        else:
            print("not scrubbing raw files: some prepared splits are missing")


if __name__ == "__main__":
    main()
