"""Run Jev or a local Laya server with resumable predictions."""
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from export_mbox import scrub_pii


def model_state(email):
    return {'subject': scrub_pii(email['subject']), 'from_domain': email['from_domain'],
            'text': scrub_pii(email['text'])[:500]}


def parse_response(payload):
    """Accept the shared protocol and obvious singular/wrapped variants."""
    root = payload
    if isinstance(root, dict):
        root = root.get('data', root.get('result', root))
    answer = root.get('answers', root.get('answer', root)) if isinstance(root, dict) else None
    if isinstance(answer, dict):
        answer = answer.get('triage', answer)
    keys = list(payload) if isinstance(payload, dict) else [type(payload).__name__]
    nested = list(answer) if isinstance(answer, dict) else []
    error = f'Unexpected response keys: root={keys}, answer={nested}'
    if not isinstance(answer, dict):
        raise ValueError(error)
    pred = answer.get('choice', answer.get('label'))
    probs = answer.get('probabilities', answer.get('probs'))
    confidence = answer.get('confidence')
    if pred not in config.LABELS or not isinstance(probs, dict) or set(probs) != set(config.LABELS):
        raise ValueError(error)
    try:
        probs = {k: float(v) for k, v in probs.items()}
        confidence = float(probs[pred] if confidence is None else confidence)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in [confidence, *probs.values()]):
            raise ValueError(error)
        if not math.isclose(sum(probs.values()), 1, abs_tol=0.02):
            raise ValueError(error)
        usage = root.get('usage', payload.get('usage', {})) or {}
        tokens = usage.get('input_tokens', usage.get('prompt_tokens'))
        if tokens is not None and (isinstance(tokens, bool) or int(tokens) != tokens or tokens < 0):
            raise ValueError(error)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(error) from None
    return {'pred': pred, 'confidence': confidence, 'probabilities': probs, 'input_tokens': tokens}


def request_one(email, model, key=None, post=None, sleep=time.sleep):
    if post is None:
        import requests
        post = requests.post
    payload = {'state': model_state(email), 'questions': config.QUESTIONS}
    headers = {}
    if model == 'jev':
        payload['model'] = config.JEV_MODEL
        headers['Authorization'] = f'Bearer {key}'
    elif os.environ.get('LAYA_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['LAYA_API_KEY']
    endpoint = config.JEV_ENDPOINT if model == 'jev' else config.LAYA_ENDPOINT
    start = time.perf_counter()
    for attempt in range(5):
        try:
            response = post(endpoint, json=payload, headers=headers, timeout=(10, 120), allow_redirects=False)
        except Exception:
            # Do not print exception bodies or prepared requests containing credentials.
            raise RuntimeError(f'Connection failed for id={email["id"]}') from None
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt < 4:
                sleep(2 ** attempt)
                continue
        if response.status_code != 200:
            raise RuntimeError(f'HTTP {response.status_code} for id={email["id"]}')
        try:
            raw = response.json()
        except ValueError:
            raise ValueError('Server returned invalid JSON') from None
        return {'id': email['id'], **parse_response(raw), 'latency_ms': (time.perf_counter() - start) * 1000}
    raise RuntimeError('Retry limit reached')


def run(directory, model, split='test', workers=None):
    ids = json.loads((directory / f'{split}_ids.json').read_text())
    emails = {r['id']: r for r in config.read_jsonl(directory / 'emails.jsonl')}
    if not set(ids) <= emails.keys():
        raise ValueError('Split contains missing email ids')
    path = directory / 'results' / f'{model}.jsonl'
    done = {r['id'] for r in config.read_jsonl(path)}
    pending = [emails[i] for i in ids if i not in done]
    if not pending:
        print('All requested ids already completed')
        return
    key = config.api_key(directory) if model == 'jev' else None
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=workers or (4 if model == 'jev' else 1)) as pool, path.open('a', encoding='utf-8') as out:
        jobs = {pool.submit(request_one, row, model, key): row['id'] for row in pending}
        for job in as_completed(jobs):
            try:
                row = job.result()
            except (RuntimeError, ValueError) as exc:
                failures += 1
                print(f'Failed id={jobs[job]}: {exc}')
                continue
            out.write(json.dumps(row) + '\n')
            out.flush()
            os.fsync(out.fileno())
            print(f'Completed id={row["id"]}')
    if failures:
        raise RuntimeError(f'{failures} requests failed; rerun to resume')


if __name__ == '__main__':
    p = config.parser(__doc__)
    p.add_argument('--model', choices=['jev', 'laya', 'laya-ft'], required=True)
    p.add_argument('--split', choices=['test', 'train'], default='test')
    p.add_argument('--workers', type=int)
    args = p.parse_args()
    if args.workers is not None and args.workers < 1:
        p.error('--workers must be positive')
    run(config.data_dir(args), args.model, args.split, args.workers)
