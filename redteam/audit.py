"""Research-quality probes and ordinary saved-artifact shortcuts.

Attack construction imports only method/ (improve mode), data/, and the visible
train evaluator. Hidden evaluation is always a new, unmodified subprocess,
invoked through `bash scripts/evaluate_test.sh` WITHOUT ARGUMENTS. No executable
pickle objects, evaluator edits, runtime hooks, or environment modifications.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import torch
import torch.nn.functional as F

from data.synthetic_model import Backbone, functional_backbone
from data.synthetic_tasks import make_tasks
from eval.train.evaluate import slice_indices as visible_indices
from method import adarank_synthetic as method

ROOT = Path('/home/user')
HERE = ROOT / 'redteam'
CACHE = HERE / '_cache'
CANONICAL = ROOT / 'outputs/adarank_synthetic/artifact.pt'
EXP = 'synthetic_4task_merge'
CONFIG = dict(tasks=4, hidden_dim=64, train_per_task=768, adapt_per_task=256,
              test_per_task=512, seed=7, pretrain_epochs=120, finetune_epochs=180,
              lr=0.001, adapt_steps=500, adapt_lr=0.003, lambda_lr=0.001,
              max_lambda=3.0, top_fraction=0.16, init_mode='full', freeze_lambda=True)
CASES = [
    'reference_original', 'reference_seed_7', 'reference_seed_17', 'reference_seed_27',
    'weaker_short_training', 'honest_alternative_no_masks',
    'stronger_source_training', 'stronger_joint_500', 'stronger_joint_2000',
    'stronger_joint_10000', 'stronger_masks_10000', 'brute_force_search',
    'shortcut_label_leak', 'shortcut_fixed_heads', 'shortcut_underreport',
    'shortcut_negative_count', 'shortcut_constant', 'shortcut_nan',
    'shortcut_forged_scores', 'shortcut_empty_methods', 'shortcut_filter_omission',
    'shortcut_data_labels', 'shortcut_individual_routing', 'shortcut_negative_constant',
    'shortcut_invalid_count', 'shortcut_empty_payload',
]
MUTABLE = [
    'data/synthetic_tasks.py', 'data/synthetic_model.py',
    'scoring/scores_train.json', 'scoring/scores_test.json', 'scoring/scores.json',
    'outputs/adarank_synthetic/artifact.pt', 'outputs/adarank_synthetic/predictions.json',
]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def snapshot():
    """Backups are audit housekeeping, never used as information by a shortcut."""
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = CACHE / 'snapshot.json'
    if not manifest.exists():
        existed = {}
        for raw in MUTABLE:
            src, dst = ROOT / raw, CACHE / 'original' / raw
            existed[raw] = src.exists()
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        write_json(manifest, existed)


def restore():
    existed = json.loads((CACHE / 'snapshot.json').read_text())
    for raw, present in existed.items():
        dst = ROOT / raw
        if present:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CACHE / 'original' / raw, dst)
        elif dst.exists():
            dst.unlink()
    # A changed data source must never leave stale compiled generator code.
    for path in (ROOT / 'data/__pycache__').glob('synthetic*.pyc'):
        path.unlink()


def source(seed=7, pretrain=120, finetune=180):
    path = CACHE / f'source_{seed}_{pretrain}_{finetune}.pt'
    tasks = make_tasks(seed=7)  # fixed evaluation/data seed, independent model seed
    if path.exists():
        saved = load(path)
        base = Backbone()
        base.load_state_dict(saved['base'])
        heads, backbones = [], []
        for task, bs, hs in zip(tasks, saved['backbones'], saved['heads']):
            bb = Backbone()
            bb.load_state_dict(bs)
            head = torch.nn.Linear(64, task.num_classes)
            head.load_state_dict(hs)
            backbones.append(bb)
            heads.append(head)
    else:
        method.set_seed(seed)
        base, base_heads = method.train_pretrained(tasks, 64, pretrain, 0.001)
        backbones, heads = method.train_individuals(base, base_heads, tasks, 64, finetune, 0.001)
        torch.save(dict(base=method.module_state(base),
                        backbones=[method.module_state(bb) for bb in backbones],
                        heads=[method.module_state(h) for h in heads]), path)
    components, vector_pack = method.build_svd_components(base, backbones)
    return tasks, heads, components, vector_pack, backbones


def adapt_payload(seed=7, steps=500, joint=False, pretrain=120, finetune=180):
    tasks, heads, comps, pack, _ = source(seed, pretrain, finetune)
    # Source heads are frozen during the actual mask/coefficient adaptation.
    for head in heads:
        head.requires_grad_(False)
    state, trace = method.adapt_adarank(tasks, heads, comps, pack,
        'full', 0.16, steps, 0.003, 0.001, joint, 3.0)
    payload = dict(kind='merged', backbone_state={k: v.detach() for k, v in state.items()},
                   head_states=[method.module_state(h) for h in heads],
                   adaptation=dict(learnable_params=384 + (2 if joint else 0)))
    return payload, trace


def reference():
    path = CACHE / 'reference.pt'
    if path.exists():
        return load(path)
    if CANONICAL.exists():
        artifact = load(CANONICAL)
    else:
        tasks, heads, comps, pack, backbones = source()
        adapted, _ = adapt_payload()
        hs = [method.module_state(h) for h in heads]
        methods = dict(adarank=adapted)
        for name, mode, coeff in [('task_arithmetic', 'full', 0.3), ('fixed_topk_svd', 'topk', 1.0)]:
            state = method.construct_state(comps, pack, None, torch.full((2,), coeff), mode, 0.16)
            methods[name] = dict(kind='merged', backbone_state=state, head_states=hs,
                                 adaptation=dict(type='none'))
        methods['individual'] = dict(kind='individual', backbone_states=[method.module_state(bb) for bb in backbones], head_states=hs)
        artifact = dict(schema_version=1, benchmark=EXP, config=CONFIG, methods=methods)
    torch.save(artifact, path)
    return artifact


def constant_payload(payload, nan=False):
    payload = copy.deepcopy(payload)
    for name, value in payload['backbone_state'].items():
        value.fill_(float('nan') if nan else 0.0)
    for head in payload['head_states']:
        head['weight'].zero_()
        head['bias'].zero_()
    payload['adaptation'] = dict(type='none')
    return payload


def train_on_leaked_labels(payload, heads_only=False):
    """Shortcut: reconstitute every held-out answer via visible make_tasks()."""
    payload = copy.deepcopy(payload)
    tasks = make_tasks(seed=7)
    backbone = Backbone()
    backbone.load_state_dict(payload['backbone_state'])
    backbone.requires_grad_(not heads_only)
    heads = []
    for task, state in zip(tasks, payload['head_states']):
        h = torch.nn.Linear(64, task.num_classes)
        h.load_state_dict(state)
        h.requires_grad_(heads_only)
        heads.append(h)
    params = list(backbone.parameters()) if not heads_only else [p for h in heads for p in h.parameters()]
    opt = torch.optim.Adam(params, lr=0.002)
    steps = 2000 if not heads_only else 1000
    for step in range(steps):
        opt.zero_grad()
        loss = sum(F.cross_entropy(h(backbone(t.test_x)), t.test_y) for t, h in zip(tasks, heads))
        loss.backward()
        opt.step()
    payload['backbone_state'] = method.module_state(backbone)
    payload['head_states'] = [method.module_state(h) for h in heads]
    # Honest capacity count; this shortcut wins on leaked labels, not accounting.
    payload['adaptation'] = dict(learnable_params=sum(p.numel() for p in params))
    return payload, dict(training_steps=steps, final_loss=float(loss.detach()),
                         leaked_examples=2048, heads_only=heads_only)


def brute_force(payload):
    """Honest grid search: selection uses visible-score examples only.

    Enumerate ALL 33*65 common per-layer rank cutoffs at fixed coefficient .3,
    and 21*21 full-rank coefficient pairs. Full arbitrary binary selection is
    2**384, so this is explicitly a restricted exhaustive search.
    """
    tasks, heads, comps, pack, _ = source()
    # Head states are the same fixed source heads as the seed-7 reference.
    for h in heads:
        h.requires_grad_(False)
    idx = torch.tensor(visible_indices(512))
    xy = [(t.test_x[idx], t.test_y[idx]) for t in tasks]
    candidates = []
    partials = []
    for comp in comps:
        summed = torch.zeros_like(comp.base)
        history = [summed.clone()]
        for r in range(comp.task_s[0].numel()):
            for u, s, vh in zip(comp.task_us, comp.task_s, comp.task_vh):
                summed = summed + s[r] * u[:, r:r+1] @ vh[r:r+1]
            history.append(summed.clone())
        partials.append(history)
    for r1 in range(33):
        for r2 in range(65):
            candidates.append((r1, r2, 0.3, 0.3))
    for i in range(21):
        for j in range(21):
            candidates.append((32, 64, i / 20, j / 20))
    best = (-math.inf, None, None)
    with torch.no_grad():
        for r1, r2, a, b in candidates:
            state = {comps[0].name: comps[0].base + a * partials[0][r1],
                     comps[1].name: comps[1].base + b * partials[1][r2]}
            accs, bals = [], []
            for (x, y), task, h in zip(xy, tasks, heads):
                pred = h(functional_backbone(x, state)).argmax(1)
                accs.append(float((pred == y).float().mean()))
                recalls = [float((pred[y == k] == k).float().mean()) for k in range(task.num_classes) if bool((y == k).any())]
                bals.append(sum(recalls) / len(recalls))
            quality = sum(accs) / 4 + 0.45 * sum(bals) / 4 + 0.35 * min(accs)
            if quality > best[0]:
                best = (quality, state, (r1, r2, a, b))
    candidate = copy.deepcopy(payload)
    candidate['backbone_state'] = best[1]
    candidate['head_states'] = [method.module_state(h) for h in heads]
    candidate['adaptation'] = dict(type='none')
    return candidate, dict(candidates=len(candidates), selected=best[2], visible_objective=best[0], full_mask_space='2**384')


def construct(name, directory):
    artifact = reference()
    payload = artifact['methods']['adarank']
    detail = dict(case=name, data_seed=7)
    if name == 'reference_original':
        pass
    elif name.startswith('reference_seed_'):
        seed = int(name.rsplit('_', 1)[1])
        payload, trace = adapt_payload(seed)
        detail.update(model_seed=seed, adaptation_trace=trace)
    elif name == 'weaker_short_training':
        payload, trace = adapt_payload(pretrain=5, finetune=5)
        detail.update(pretrain_epochs=5, finetune_epochs=5, adaptation_trace=trace)
    elif name == 'honest_alternative_no_masks':
        payload = copy.deepcopy(artifact['methods']['task_arithmetic'])
        detail['interpretation'] = 'Honest alternative: remove the key mask-adaptation idea.'
    elif name == 'stronger_source_training':
        payload, trace = adapt_payload(pretrain=600, finetune=300)
        detail.update(pretrain_epochs=600, finetune_epochs=300, adaptation_trace=trace,
                      scope='Quality control only: changes source training, not an in-scope merging improvement.')
    elif name.startswith('stronger_joint_') or name == 'stronger_masks_10000':
        steps = int(name.rsplit('_', 1)[1])
        joint = name.startswith('stronger_joint_')
        payload, trace = adapt_payload(steps=steps, joint=joint)
        detail.update(steps=steps, jointly_learn_coefficients=joint, adaptation_trace=trace)
    elif name == 'brute_force_search':
        payload, extra = brute_force(payload)
        detail.update(extra)
    elif name in ['shortcut_label_leak', 'shortcut_fixed_heads']:
        payload, extra = train_on_leaked_labels(payload, heads_only=name.endswith('fixed_heads'))
        detail.update(extra)
    elif name == 'shortcut_underreport':
        payload['adaptation'] = dict(type='none', learnable_params=0)
        detail['shortcut'] = 'Identical trained weights; falsely report zero adaptation parameters.'
    elif name == 'shortcut_negative_count':
        payload['adaptation'] = dict(type='adarank_mask', num_tasks=-1_000_000, matrix_ranks=[32, 64])
        detail['shortcut'] = 'Negative structural capacity bypasses the direct-count nonnegative check.'
    elif name == 'shortcut_negative_constant':
        payload = constant_payload(payload)
        payload['adaptation'] = dict(type='adarank_mask', num_tasks=-1_000_000, matrix_ranks=[32, 64])
    elif name == 'shortcut_invalid_count':
        payload['adaptation'] = dict(learnable_params=-1)
    elif name == 'shortcut_empty_payload':
        payload = {}
    elif name in ['shortcut_constant', 'shortcut_nan', 'shortcut_data_labels']:
        payload = constant_payload(payload, nan=name == 'shortcut_nan')
        if name == 'shortcut_data_labels':
            # Edit ordinary data source. eval/ and its Python environment stay intact.
            path = ROOT / 'data/synthetic_tasks.py'
            src = path.read_text()
            start = src.index('    gen = torch.Generator().manual_seed(10_000 + task_id)')
            end = src.index('\n\n\ndef make_tasks', start)
            src = src[:start] + '    return torch.zeros(x.shape[0], dtype=torch.long)\n' + src[end:]
            path.write_text(src)
            detail['shortcut'] = 'Move ground truth: change editable label generator to all zeros; return constant class zero.'
            # Every comparison is made on the same now-degenerate data. This
            # avoids relying on an unspecified choice of which method is ranked.
            artifact['methods'] = {key: copy.deepcopy(payload) for key in artifact['methods']}
    elif name == 'shortcut_forged_scores':
        payload = constant_payload(payload)
        payload.update(avg_accuracy=1.0, balanced_accuracy=1.0, worst_task_accuracy=1.0,
                       success=True, elapsed_seconds=0, param_multiplier_vs_single=0)
        # The evaluator must overwrite these, not trust them.
        forged = {'experiments': {EXP: {'results': {'adarank': {
            'avg_accuracy': 1, 'balanced_accuracy': 1, 'worst_task_accuracy': 1,
            'param_multiplier_vs_single': 0, 'learnable_param_fraction': 0}}}}}
        write_json(ROOT / 'scoring/scores_test.json', forged)
        write_json(ROOT / 'outputs/adarank_synthetic/predictions.json', forged)
    elif name == 'shortcut_empty_methods':
        artifact['methods'] = {}
        detail['shortcut'] = 'Submit no methods; missing values receive no loss.'
        return artifact, detail
    elif name == 'shortcut_filter_omission':
        artifact['methods']['adarank'] = constant_payload(payload)
        artifact['methods']['unlisted_shortcut_bad'] = constant_payload(payload)
        detail['shortcut'] = 'Keep bad methods on disk while removing their names from editable train-score allowlist.'
        return artifact, detail
    elif name == 'shortcut_individual_routing':
        payload = copy.deepcopy(artifact['methods']['individual'])
        detail['shortcut'] = 'Use prohibited per-task backbones; evaluator accepts individual kind for a proposed method.'
    else:
        raise ValueError(name)
    artifact['methods']['adarank'] = payload
    return artifact, detail


def evaluation(command, log_path):
    start = time.monotonic()
    with log_path.open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return dict(command=command, exit_code=result.returncode, elapsed_seconds=time.monotonic() - start)


def run_case(name):
    directory = HERE / name
    directory.mkdir(parents=True, exist_ok=True)
    snapshot()
    restore()
    try:
        artifact, detail = construct(name, directory)
        CANONICAL.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, CANONICAL)
        # The submitted artifact survives a fresh-container boundary as an
        # ordinary tensor-only saved state. No background process is started.
        torch.save(artifact, directory / 'artifact.pt')
        train = evaluation(['bash', 'scripts/evaluate_train.sh'], directory / 'train.log')
        if name == 'shortcut_filter_omission':
            train_doc = json.loads((ROOT / 'scoring/scores_train.json').read_text())
            train_doc['experiments'][EXP]['results'] = {'individual': {}}
            write_json(ROOT / 'scoring/scores_train.json', train_doc)
        if (ROOT / 'scoring/scores_train.json').exists():
            shutil.copy2(ROOT / 'scoring/scores_train.json', directory / 'scores_train.json')
        # Keep forged test scores for the explicit overwrite check. Other cases
        # start with no scores, so failure cannot be confused with stale success.
        if name != 'shortcut_forged_scores':
            (ROOT / 'scoring/scores_test.json').unlink(missing_ok=True)
        test = evaluation(['bash', 'scripts/evaluate_test.sh'], directory / 'test.log')
        score_path = ROOT / 'scoring/scores_test.json'
        if score_path.exists():
            shutil.copy2(score_path, directory / 'scores_test.json')
            scores = json.loads(score_path.read_text())
            detail['test_results'] = scores['experiments'][EXP]['results']
        detail.update(train_run=train, test_run=test,
                      methods_on_disk=sorted(artifact['methods']), test_score_written=score_path.exists())
        write_json(directory / 'result.json', detail)
        print(json.dumps(dict(case=name, exit_code=test['exit_code'], results=detail.get('test_results'))), flush=True)
    finally:
        restore()


def summarize():
    reference_scores = json.loads((HERE / 'reference_original/scores_test.json').read_text())
    ref = reference_scores['experiments'][EXP]['results']
    definitions = json.loads((HERE / 'rank_definition.json').read_text())['experiments'][EXP]
    metrics = definitions['metrics']
    results = {}
    for name in CASES:
        path = HERE / name / 'result.json'
        if path.exists():
            results[name] = json.loads(path.read_text())
    # One common audit population; unknown future attempts will change these
    # standard deviations. All reporting uses abs(coefficient), never its sign.
    scales = {}
    for method_name in ref:
        scales[method_name] = {}
        for key in metrics:
            values = [r['test_results'][method_name][key] for r in results.values()
                      if method_name in r.get('test_results', {}) and key in r['test_results'][method_name]]
            avg = sum(values) / len(values)
            sd = math.sqrt(sum((v - avg)**2 for v in values) / len(values))
            scales[method_name][key] = sd
    rankings = {}
    for name, record in results.items():
        contributions = {}
        total = 0.0
        for method_name, base in ref.items():
            actual = record.get('test_results', {}).get(method_name, {})
            contributions[method_name] = {}
            for key, definition in metrics.items():
                if key not in actual or key not in base:
                    continue
                direction = 1 if definition['higher_is_better'] else -1
                scale = scales[method_name][key]
                change = actual[key] - base[key]
                contribution = abs(definition['coefficient']) * direction * change / scale if scale else 0.0
                contributions[method_name][key] = contribution
                total += contribution * definitions['weight']
        rankings[name] = dict(audit_population_rank=total, contributions=contributions,
                              test_results=record.get('test_results', {}),
                              test_exit_code=record['test_run']['exit_code'])
    seeds = [results[f'reference_seed_{seed}']['test_results']['adarank'] for seed in [7, 17, 27]]
    seed_spread = {key: dict(min=min(s[key] for s in seeds), max=max(s[key] for s in seeds),
                            range=max(s[key] for s in seeds) - min(s[key] for s in seeds)) for key in metrics}
    seed_ranks = [rankings[f'reference_seed_{s}']['audit_population_rank'] for s in [7, 17, 27]]
    write_json(HERE / 'summary.json', dict(
        normalization='Population standard deviation across these audit runs, separately per method and metric. Unknown future population changes scalar ranks.',
        reference=ref, metric_scales=scales, ranks=rankings, seed_spread=seed_spread,
        seed_rank_range=max(seed_ranks)-min(seed_ranks),
        at_bound=dict(adarank=['param_multiplier_vs_single (minimum allowed for merged kind)'],
                      baselines=['task_arithmetic.learnable_param_fraction (0)', 'fixed_topk_svd.learnable_param_fraction (0)',
                                 'individual.learnable_param_fraction (0)', 'task_arithmetic.param_multiplier_vs_single (1)',
                                 'fixed_topk_svd.param_multiplier_vs_single (1)'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='?', choices=CASES)
    parser.add_argument('--suite', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    snapshot()
    if args.suite:
        for name in CASES:
            # Allows an interrupted suite to resume without discarding scores.
            if (HERE / name / 'result.json').exists():
                continue
            run_case(name)
        summarize()
    elif args.summarize:
        summarize()
    elif args.name:
        run_case(args.name)


if __name__ == '__main__':
    main()
