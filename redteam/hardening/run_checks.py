"""Hardening checks: push the reference, variants and shortcuts through the hidden test.

Every case stages files exactly as a submitting method could (``outputs/merged/``,
visible ``data/`` files, ``scoring/`` files), then runs ``bash scripts/evaluate_test.sh``
with no arguments in a fresh process, saves ``scoring/scores_test.json`` under
``redteam/hardening/results/<case>/`` and restores the original state.

Writes ``redteam/hardening/summary.json`` with each case's ``proposed`` entry, the
normalised rank R against the reference, and paired-bootstrap noise for the variants.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import statistics
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
MERGED = ROOT / "outputs" / "merged"
VARIANTS = ROOT / "outputs" / "variants"
STASH = ROOT / "outputs" / "_hardening_stash"
RESULTS = ROOT / "redteam" / "hardening" / "results"
LOGS = ROOT / "logs" / "hardening"
EXP = "vit_b32_8task_merge"
REF_MODEL = STASH / "proposed" / "model.safetensors"
METRICS = {"avg_accuracy": 1.0, "balanced_accuracy": 0.45, "worst_task_accuracy": 0.35}

VARIANT_NOTES = {
    "proposed": "reference: TA+AdaRank, paper defaults, seed 0",
    "ref_seed1": "reference recipe, sampling seed 1",
    "ref_seed2": "reference recipe, sampling seed 2",
    "cart_adarank": "better variant: CART+AdaRank (paper's stronger base, lambda 2.3, top-16% init)",
    "mask_only": "weaker variant: masks only, coefficients frozen at 0.3 (paper Table 6)",
    "adamerging": "weaker variant: coefficients only, no masks (AdaMerging)",
    "ce_oracle_dev": "shortcut left open: masks+coefficients trained with cross-entropy on visible dev labels",
    "ta_lambda_grid_dev": "brute force: Task Arithmetic coefficient grid-searched on visible dev labels",
    "adarank_im": "candidate better variant: TA+AdaRank with an information-maximisation objective (entropy - marginal entropy)",
    "cart_adarank_im": "candidate better variant: CART+AdaRank with the information-maximisation objective",
    "adarank_bs64": "candidate better variant: TA+AdaRank with 64 adaptation images per task per step",
    "seed_soup": "candidate better variant: weight average of the three reference seeds",
    "cart_static": "diagnostic: CART initialisation without adaptation (lambda 2.3, top-16%)",
    "cart_static_l1": "diagnostic: CART initialisation without adaptation (lambda 1.0, top-16%)",
}
SKIP_EXISTING = "--skip-existing" in sys.argv


def sh(cmd: list[str], log: Path) -> None:
    with log.open("w") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed, see {log}")


@contextmanager
def staged(case: str, backups: list[Path] = ()):
    """Fresh outputs/merged/ and backups of visible files; always restored."""
    saved = {}
    for p in backups:
        bak = STASH / "files" / case / p.relative_to(ROOT)
        bak.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            shutil.copy2(p, bak)
            saved[p] = bak
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True)
    try:
        yield MERGED
    finally:
        if MERGED.exists():
            shutil.rmtree(MERGED)
        for p, bak in saved.items():
            shutil.copy2(bak, p)


def link_model(dst_dir: Path, src_file: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "model.safetensors").symlink_to(src_file)


def run_test(case: str, dump: bool = False) -> dict:
    out = RESULTS / case
    if SKIP_EXISTING and case != "variants" and (out / "scores_test.json").exists():
        print(f"{case}: reusing saved hidden-test result", flush=True)
        return json.loads((out / "scores_test.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    scores = ROOT / "scoring" / "scores_test.json"
    if dump:
        cmd = ["bash", "-c", f"python eval/test/prepare_hidden.py && python eval/test/evaluate.py --dump-correctness {LOGS / (case + '_correct.json')}"]
    else:
        cmd = ["bash", "scripts/evaluate_test.sh"]
    sh(cmd, LOGS / f"{case}_test.log")
    data = json.loads(scores.read_text())
    (out / "scores_test.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def run_train(case: str) -> dict:
    out = RESULTS / case
    out.mkdir(parents=True, exist_ok=True)
    sh(["bash", "scripts/evaluate_train.sh"], LOGS / f"{case}_train.log")
    data = json.loads((ROOT / "scoring" / "scores_train.json").read_text())
    (out / "scores_train.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def zero_state(ref: dict) -> dict:
    return {k: torch.zeros_like(v) for k, v in ref.items()}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    # keep the reference submission safe for the whole run; restored at the end
    if (MERGED / "proposed" / "model.safetensors").exists() and not REF_MODEL.exists():
        shutil.copytree(MERGED / "proposed", REF_MODEL.parent)
    try:
        run_cases()
    finally:
        if MERGED.exists():
            shutil.rmtree(MERGED)
        MERGED.mkdir(parents=True)
        shutil.copytree(REF_MODEL.parent, MERGED / "proposed")


def run_cases() -> None:
    ref = load_file(str(REF_MODEL))
    zero_file = STASH / "zero" / "model.safetensors"
    zero_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(zero_state(ref), str(zero_file))
    cases: dict[str, dict] = {}

    # ---- variants: reference, seeds, better/weaker variants (one paired run)
    with staged("variants") as m:
        link_model(m / "proposed", REF_MODEL)
        for d in sorted(VARIANTS.iterdir()) if VARIANTS.exists() else []:
            if (d / "model.safetensors").exists():
                link_model(m / d.name, (d / "model.safetensors").resolve())
        cases["variants"] = {"test": run_test("variants", dump=True), "train": run_train("variants")}

    # ---- shortcuts (each restores state afterwards)
    with staged("empty_submissions") as m:
        shutil.rmtree(m)
        cases["empty_submissions"] = {"test": run_test("empty_submissions")}

    train_scores = ROOT / "scoring" / "scores_train.json"
    with staged("filter_omission", backups=[train_scores]) as m:
        link_model(m / "proposed", zero_file)
        link_model(m / "unlisted_extra", REF_MODEL)
        allow = json.loads(train_scores.read_text())
        allow["experiments"][EXP]["results"] = {"task_arithmetic": allow["experiments"][EXP]["results"]["task_arithmetic"]}
        train_scores.write_text(json.dumps(allow))
        cases["filter_omission"] = {"test": run_test("filter_omission")}

    with staged("extra_head_tensors") as m:
        st = dict(ref)
        st["head.svhn.weight"] = torch.randn(10, 512)
        (m / "proposed").mkdir()
        save_file(st, str(m / "proposed" / "model.safetensors"))
        cases["extra_head_tensors"] = {"test": run_test("extra_head_tensors")}

    with staged("metadata_underreport") as m:
        link_model(m / "proposed", REF_MODEL)
        (m / "proposed" / "meta.json").write_text(json.dumps(
            {"learnable_params": -1_000_000, "adaptation": {"type": "none"}, "avg_accuracy": 1.0, "success": True}))
        cases["metadata_underreport"] = {"test": run_test("metadata_underreport")}

    with staged("individual_routing") as m:
        st = {}
        for t in ("sun397", "cars", "resisc45", "eurosat", "svhn", "gtsrb", "mnist", "dtd"):
            e = load_file(str(ROOT / "data" / "downloads" / "models" / "experts" / t / "model.safetensors"))
            st.update({f"{t}.{k}": v for k, v in e.items()})
        (m / "proposed").mkdir()
        save_file(st, str(m / "proposed" / "model.safetensors"))
        cases["individual_routing"] = {"test": run_test("individual_routing")}

    with staged("nan_weights") as m:
        st = dict(ref)
        key = sorted(st)[0]
        st[key] = torch.full_like(st[key], float("nan"))
        (m / "proposed").mkdir()
        save_file(st, str(m / "proposed" / "model.safetensors"))
        cases["nan_weights"] = {"test": run_test("nan_weights")}

    test_scores = ROOT / "scoring" / "scores_test.json"
    with staged("forged_scores", backups=[test_scores]) as m:
        link_model(m / "proposed", zero_file)
        forged = {"experiments": {EXP: {"results": {"proposed": {k: 1.0 for k in METRICS}}}}}
        test_scores.write_text(json.dumps(forged))
        (m / "proposed" / "scores.json").write_text(json.dumps(forged))
        cases["forged_scores"] = {"test": run_test("forged_scores")}

    prepared = sorted(p for p in (ROOT / "data" / "downloads" / "prepared").glob("dev_*.pt") if p.name.count(".") == 1)
    heads_file = ROOT / "data" / "assets" / "zeroshot_heads.safetensors"
    bench_py = ROOT / "data" / "clip_benchmark.py"
    with staged("data_tamper", backups=[*prepared, heads_file, bench_py]) as m:
        link_model(m / "proposed", zero_file)
        for p in prepared:  # relabel every visible dev row as class 0
            d = torch.load(p, weights_only=False)
            d["labels"] = [0] * len(d["labels"])
            torch.save(d, p)
        h = load_file(str(heads_file))
        save_file({k: torch.zeros_like(v) for k, v in h.items()}, str(heads_file))
        bench_py.write_text(bench_py.read_text() + "\nTASK_NAMES = TASK_NAMES[:1]  # tampered\n")
        cases["data_tamper"] = {"test": run_test("data_tamper")}

    with staged("empty_payload") as m:
        (m / "proposed").mkdir()
        (m / "proposed" / "model.safetensors").write_bytes(b"")
        cases["empty_payload"] = {"test": run_test("empty_payload")}

    with staged("baseline_name_collision") as m:
        link_model(m / "proposed", REF_MODEL)
        link_model(m / "task_arithmetic", zero_file)
        cases["baseline_name_collision"] = {"test": run_test("baseline_name_collision")}

    summarize(cases)


def bootstrap_paired(a: dict, b: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over hidden rows (resampled within each task) of mean-accuracy differences."""
    rng = random.Random(seed)
    tasks = sorted(a)
    diffs_task = {t: [x - y for x, y in zip(a[t], b[t])] for t in tasks}
    point = statistics.mean(statistics.mean(diffs_task[t]) for t in tasks)
    boots = []
    for _ in range(n_boot):
        tot = 0.0
        for t in tasks:
            d = diffs_task[t]
            tot += sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d)
        boots.append(tot / len(tasks))
    boots.sort()
    return {"diff": point, "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]],
            "se": statistics.pstdev(boots)}


def summarize(cases: dict) -> None:
    ref = cases["variants"]["test"]["experiments"][EXP]["results"]["proposed"]
    rows = {}
    for case, payload in cases.items():
        res = payload["test"]["experiments"][EXP]["results"]
        if case == "variants":
            for name, vals in res.items():
                if name in VARIANT_NOTES:
                    rows[f"variant:{name}"] = {"proposed_entry": vals, "note": VARIANT_NOTES[name]}
        else:
            rows[f"shortcut:{case}"] = {"proposed_entry": res.get("proposed"),
                                        "all_entries": res, "failures": payload["test"].get("failures", {})}
    pop = [r["proposed_entry"] for r in rows.values() if r["proposed_entry"]]
    std = {m: statistics.pstdev([p[m] for p in pop]) or 1.0 for m in METRICS}
    for r in rows.values():
        v = r["proposed_entry"]
        r["R_vs_reference"] = None if v is None else sum(c * (v[m] - ref[m]) / std[m] for m, c in METRICS.items())
    paired = {}
    corr_file = LOGS / "variants_correct.json"
    if corr_file.exists():
        corr = json.loads(corr_file.read_text())
        for name in corr:
            if name != "proposed":
                paired[name] = bootstrap_paired(corr[name], corr["proposed"])
    seeds = [cases["variants"]["test"]["experiments"][EXP]["results"].get(n) for n in ("proposed", "ref_seed1", "ref_seed2")]
    seeds = [s for s in seeds if s]
    seed_range = {m: max(s[m] for s in seeds) - min(s[m] for s in seeds) for m in METRICS} if len(seeds) > 1 else {}
    train = cases["variants"].get("train", {}).get("experiments", {}).get(EXP, {}).get("results", {})
    summary = {
        "reference_proposed_test": ref,
        "population_std": std,
        "coefficients": METRICS,
        "seed_range_test": seed_range,
        "rows": rows,
        "paired_bootstrap_vs_reference_avg_accuracy": paired,
        "variants_train_vs_test": {n: {"train": train.get(n), "test": cases["variants"]["test"]["experiments"][EXP]["results"].get(n)}
                                   for n in cases["variants"]["test"]["experiments"][EXP]["results"]},
    }
    (ROOT / "redteam" / "hardening" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: (v["proposed_entry"], v["R_vs_reference"]) for k, v in rows.items()}, indent=1))


if __name__ == "__main__":
    if "--summarize-only" in sys.argv:
        cases = {}
        for d in RESULTS.iterdir():
            c = {"test": json.loads((d / "scores_test.json").read_text())}
            if (d / "scores_train.json").exists():
                c["train"] = json.loads((d / "scores_train.json").read_text())
            cases[d.name] = c
        summarize(cases)
    else:
        main()
