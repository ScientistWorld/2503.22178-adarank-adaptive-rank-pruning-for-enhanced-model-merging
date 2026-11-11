"""Summarize saved audit evidence using only the Python standard library.

Successful empty results skip losses, as the specified rank does. A nonzero
hidden-test exit with no fresh score file is separately unrankable; the gym
has not supplied a worst-value result or an external failure policy.
"""
from pathlib import Path
import json
import math

HERE = Path(__file__).resolve().parent
EXP = 'synthetic_4task_merge'


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def summarize(case_names=None):
    reference_scores = json.loads((HERE / 'reference_original/scores_test.json').read_text())
    ref = reference_scores['experiments'][EXP]['results']
    definitions = json.loads((HERE / 'rank_definition.json').read_text())['experiments'][EXP]
    metrics = definitions['metrics']
    results = {}
    if case_names is None:
        case_names = [p.parent.name for p in sorted(HERE.glob('*/result.json'))]
    for name in case_names:
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
        if record['test_run']['exit_code'] != 0 or not record['test_score_written']:
            rankings[name] = dict(audit_population_rank=None,
                                  rank_status='rejected_without_fresh_scores',
                                  contributions={}, test_results=record.get('test_results', {}),
                                  test_exit_code=record['test_run']['exit_code'])
            continue
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
        rankings[name] = dict(audit_population_rank=total, rank_status='successful_test', contributions=contributions,
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



if __name__ == '__main__':
    summarize()
