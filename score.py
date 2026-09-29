"""Score held-out predictions and write a filmable local scoreboard."""
import html
import itertools
from collections import Counter
import json
import random

import config


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def metrics(rows, labels, emails, model):
    if not rows:
        raise ValueError('No test predictions to score')
    names = list(config.LABELS)
    matrix = [[0] * 4 for _ in names]
    correct, wrong = [], []
    for row in rows:
        truth = labels[row['id']]
        pred = row['pred']
        if truth not in names or pred not in names:
            raise ValueError('Unknown label in scoring data')
        matrix[names.index(truth)][names.index(pred)] += 1
        correct.append(int(truth == pred))
        if truth != pred and row['confidence'] >= 0.9:
            wrong.append({'id': row['id'], 'subject': emails[row['id']]['subject'],
                          'pred': pred, 'true': truth, 'confidence': row['confidence']})
    per_class = {}
    for i, name in enumerate(names):
        tp = matrix[i][i]
        precision = tp / max(1, sum(r[i] for r in matrix))
        recall = tp / max(1, sum(matrix[i]))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class[name] = {'precision': precision, 'recall': recall, 'f1': f1, 'support': sum(matrix[i])}
    rng = random.Random(42)
    bootstrap = [sum(rng.choices(correct, k=len(correct))) / len(correct) for _ in range(1000)]
    usage_known = all(r.get('input_tokens') is not None for r in rows)
    tokens = sum(r.get('input_tokens') or 0 for r in rows)
    cost = tokens * config.JEV_PRICE_PER_M_INPUT / 1_000_000 if usage_known else None
    if model != 'jev':
        cost = 0
    # Pre-registered secondary score (config.NEEDS_ME): needs me vs does not need me.
    need_truth = [labels[r['id']] in config.NEEDS_ME for r in rows]
    need_pred = [r['pred'] in config.NEEDS_ME for r in rows]
    need_hits = sum(t and p for t, p in zip(need_truth, need_pred))
    needs_me = {'accuracy': sum(t == p for t, p in zip(need_truth, need_pred)) / len(rows),
                'caught': need_hits, 'of': sum(need_truth),
                'recall': need_hits / max(1, sum(need_truth)),
                'false_alarms': sum(p and not t for t, p in zip(need_truth, need_pred))}
    return {'n': len(rows), 'accuracy': sum(correct) / len(rows), 'needs_me': needs_me,
            'accuracy_ci95': [percentile(bootstrap, .025), percentile(bootstrap, .975)],
            'per_class': per_class, 'macro_f1': sum(v['f1'] for v in per_class.values()) / 4,
            'confusion_labels': names, 'confusion_matrix': matrix,
            'confident_wrong_count': len(wrong), 'confident_wrong': wrong,
            'latency_p50_ms': percentile([r['latency_ms'] for r in rows], .5),
            'latency_p95_ms': percentile([r['latency_ms'] for r in rows], .95),
            'input_tokens': tokens if usage_known else None, 'known_input_tokens': tokens,
            'cost_usd': cost, 'cost_per_1000_usd': cost * 1000 / len(rows) if cost is not None else None,
            'cost_note': '$0 (local)' if model != 'jev' else ('usage available' if usage_known else 'Unknown: missing input token usage')}


def paired_comparisons(predictions, labels):
    pairs = []
    for a, b in itertools.combinations(predictions, 2):
        common = sorted(predictions[a].keys() & predictions[b].keys())
        a_only = [i for i in common if predictions[a][i]['pred'] == labels[i] and predictions[b][i]['pred'] != labels[i]]
        b_only = [i for i in common if predictions[b][i]['pred'] == labels[i] and predictions[a][i]['pred'] != labels[i]]
        pairs.append({'a': a, 'b': b, 'n_paired': len(common), 'a_right_b_wrong': len(a_only),
                      'b_right_a_wrong': len(b_only), 'a_only_ids': a_only, 'b_only_ids': b_only})
    return pairs


def render_html(board, blur=False):
    esc = html.escape
    columns = []
    for name, m in board['models'].items():
        cost = f'${m["cost_per_1000_usd"]:.5f}' if m['cost_per_1000_usd'] is not None else 'Unknown'
        if name != 'jev':
            cost = '$0 (local)'
        matrix = '<tr><th>True / predicted</th>' + ''.join(f'<th>{k}</th>' for k in config.LABELS) + '</tr>'
        for label, row in zip(config.LABELS, m['confusion_matrix']):
            matrix += f'<tr><th>{label}</th>' + ''.join(f'<td>{n}</td>' for n in row) + '</tr>'
        errors = ''.join(f'<li><span class="subject">{esc(r["subject"])}</span><br><small>{esc(r["id"])}<br>{r["true"]} &rarr; {r["pred"]} ({r["confidence"]:.0%})</small></li>' for r in m['confident_wrong']) or '<li>None</li>'
        per_class = ''.join(f'<tr><th>{label}</th><td>{v["precision"]:.3f}</td><td>{v["recall"]:.3f}</td><td>{v["f1"]:.3f}</td></tr>' for label, v in m['per_class'].items())
        columns.append(f'''<section><h2>{esc(name)}</h2><div class="accuracy">{m['accuracy']:.1%}</div>
<p>95% CI {m['accuracy_ci95'][0]:.1%} to {m['accuracy_ci95'][1]:.1%}<br>{m['n']} / {board['test_size']} test emails</p>
<dl><dt>Caught what needs me</dt><dd>{m['needs_me']['caught']} / {m['needs_me']['of']} ({m['needs_me']['recall']:.0%}), {m['needs_me']['false_alarms']} false alarms</dd><dt>Macro F1</dt><dd>{m['macro_f1']:.3f}</dd><dt>Confident wrong</dt><dd>{m['confident_wrong_count']}</dd>
<dt>Latency p50 / p95</dt><dd>{m['latency_p50_ms']:.0f} / {m['latency_p95_ms']:.0f} ms</dd>
<dt>Cost per 1,000</dt><dd>{cost}</dd></dl><h3>Confusion matrix</h3><table>{matrix}</table>
<h3>Per-class metrics</h3><table><tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{per_class}</table>
<h3>Confident errors</h3><ul>{errors}</ul></section>''')
    pairs = ''.join(f'<p>{esc(p["a"])} alone right: {p["a_right_b_wrong"]} | {esc(p["b"])} alone right: {p["b_right_a_wrong"]} | paired n={p["n_paired"]}</p>' for p in board['paired'])
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Inbox classifier scoreboard</title>
<style>body{background:#0b1019;color:#edf4ff;font:24px/1.4 system-ui;padding:40px;margin:0}h1{font-size:48px}
main{display:grid;grid-template-columns:repeat(''' + str(max(1, len(columns))) + ''',minmax(0,1fr));gap:24px}
section{background:#161f2e;border:1px solid #344359;border-radius:18px;padding:24px;overflow-wrap:anywhere}
h2{color:#8fe5c0;text-transform:uppercase}.accuracy{font-size:76px;font-weight:800}p,small,dt{color:#acbdd3}
dd{margin:0 0 20px;font-size:32px;font-weight:700}table{width:100%;font-size:15px;border-collapse:collapse}
td,th{padding:8px;border:1px solid #344359;text-align:center}li{margin-bottom:20px}small{font-size:14px}
@media(max-width:1000px){main{grid-template-columns:1fr}}''' + ('.subject{filter:blur(8px)}' if blur else '') + '''</style>
<h1>Same inbox. Three classifiers.</h1><p>Held-out test set. Confidence threshold: 90%. Bootstrap: 1,000 resamples.</p>''' + (f'<p>Lazy guess (always "{esc(board["majority_baseline"]["label"])}"): <b>{board["majority_baseline"]["accuracy"]:.1%}</b></p>' if board.get('majority_baseline') else '') + ('<p>PARTIAL RESULTS: model coverage differs. Compare paired counts on shared ids.</p>' if board['partial'] else '') + '<main>' + ''.join(columns) + '</main><h2>Paired comparisons (McNemar counts)</h2>' + pairs + '</html>'


def score(directory, blur=False):
    ids = set(json.loads((directory / 'test_ids.json').read_text()))
    labels = {r['id']: r['label'] for r in config.read_jsonl(directory / 'labels.jsonl')}
    emails = {r['id']: r for r in config.read_jsonl(directory / 'emails.jsonl')}
    if not ids <= labels.keys() or not ids <= emails.keys():
        raise ValueError('Test ids missing human labels or emails')
    predictions, scores = {}, {}
    for name in ('jev', 'laya', 'laya-ft'):
        path = directory / 'results' / f'{name}.jsonl'
        if not path.exists():
            continue
        selected = {r['id']: r for r in config.read_jsonl(path) if r['id'] in ids}
        if not selected:
            raise ValueError(f'{name} has no predictions on the test split')
        predictions[name] = selected
        scores[name] = metrics(list(selected.values()), labels, emails, name)
    if not scores:
        raise ValueError('No model results found')
    # The "lazy guess": always answer the test set's most common label. Title C and the
    # hook's "worse than guessing" line are only honest if a model scores below this.
    counts = Counter(labels[i] for i in ids)
    top_label, top_count = counts.most_common(1)[0]
    baseline = {'label': top_label, 'accuracy': top_count / len(ids),
                'class_counts': dict(counts)}
    board = {'test_size': len(ids), 'majority_baseline': baseline, 'models': scores,
             'partial': any(len(p) != len(ids) for p in predictions.values()),
             'paired': paired_comparisons(predictions, labels)}
    config.write_json(directory / 'results' / 'scoreboard.json', board)
    (directory / 'results' / 'scoreboard.html').write_text(render_html(board, blur), encoding='utf-8')
    print(json.dumps(board, indent=2))
    return board


if __name__ == '__main__':
    p = config.parser(__doc__)
    p.add_argument('--blur-subjects', action='store_true')
    args = p.parse_args()
    score(config.data_dir(args), args.blur_subjects)
