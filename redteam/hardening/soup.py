"""Weight-space average of several submissions (e.g. AdaRank runs with different seeds)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import data.clip_benchmark as cb  # noqa: E402
from safetensors.torch import load_file  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output-root", default=str(ROOT / "outputs" / "variants"))
    a = p.parse_args()
    states = [load_file(str(Path(x) / "model.safetensors")) for x in a.inputs]
    soup = {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}
    cb.save_submission(soup, a.name, meta={"method": "weight average", "inputs": a.inputs}, root=a.output_root)
    print("wrote", a.name)


if __name__ == "__main__":
    main()
