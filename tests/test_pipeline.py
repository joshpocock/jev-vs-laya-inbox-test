import json
import mailbox
from email import policy
from email.parser import BytesParser
from pathlib import Path

import pytest

import config
from export_mbox import export, parse_message, scrub_pii
from finetune_laya import build_cases, calibration_split
from label_app import LabelStore
from run_model import parse_response, request_one, run
from sample.make_synthetic import generate
from score import metrics, paired_comparisons, render_html, score
from split import make_split


@pytest.fixture
def dataset(tmp_path):
    emails, labels = generate()
    config.write_jsonl(tmp_path / 'emails.jsonl', emails)
    config.write_jsonl(tmp_path / 'labels.jsonl', labels)
    return tmp_path


def test_pii():
    raw = 'Alice Example: alice@example.com +1 (416) 555-0199 https://example.com/a?b=2 www.test.org 12345678'
    clean = scrub_pii(raw, 'Alice Example')
    assert 'Alice' not in clean and '@' not in clean and 'example.com' not in clean
    assert '416' not in clean and '12345678' not in clean
    assert all(token in clean for token in ('[email]', '[phone]', '[link]', '[number]'))


def message(identity='a', sender='Alice Example <alice@example.com>', body='Hello Alice Example. Call 416-555-0199.', date='Mon, 21 Sep 2026 10:00:00 +0000', extra=''):
    return BytesParser(policy=policy.default).parsebytes(
        f'From: {sender}\nMessage-ID: <{identity}>\nDate: {date}\nSubject: Alice Example update\n{extra}Content-Type: text/plain; charset=utf-8\n\n{body}'.encode())


def test_mbox_fixture(tmp_path):
    source = tmp_path / 'inbox.mbox'
    box = mailbox.mbox(source)
    for msg in [message(), message(), message('owner', 'Me <owner@example.com>'),
                message('sent', extra='X-Gmail-Labels: Inbox,Sent\n'), message('empty', body=''),
                message('newest', date='Tue, 22 Sep 2026 10:00:00 +0000')]:
        box.add(msg)
    box.close()
    rows = export(source, tmp_path / 'out', ['owner@example.com'], 1)
    assert len(rows) == 1
    assert rows[0]['id'] == parse_message(message('newest'))['id']
    assert rows[0]['from_domain'] == 'example.com'
    assert 'Alice Example' not in rows[0]['text'] + rows[0]['subject']
    assert len(export(source, tmp_path / 'out', ['owner@example.com'], 20)) == 2


def test_html_and_plain_preference():
    from email.message import EmailMessage
    msg = EmailMessage()
    msg['Message-ID'] = '<html>'
    msg.set_content('<p>Hello</p><script>secret()</script><p>World</p>', subtype='html')
    assert 'Hello' in parse_message(msg)['text'] and 'secret' not in parse_message(msg)['text']
    msg = EmailMessage()
    msg.set_content('Plain version')
    msg.add_alternative('<p>Other version</p>', subtype='html')
    assert parse_message(msg)['text'] == 'Plain version'


def test_fallback_cleaning(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'laya.email', None)
    row = parse_message(message(body='Please reply.\nOn Monday someone wrote:\nOld secret'))
    assert row['text'] == 'Please reply.'


def test_split_frozen_stratified(dataset):
    test, train = make_split(dataset, 20)
    labels = {r['id']: r['label'] for r in config.read_jsonl(dataset / 'labels.jsonl')}
    assert len(test) == 20 and len(train) == 60 and not set(test) & set(train)
    assert {k: sum(labels[i] == k for i in test) for k in config.LABELS} == dict.fromkeys(config.LABELS, 5)
    frozen = (dataset / 'test_ids.json').read_bytes()
    with pytest.raises(FileExistsError):
        make_split(dataset, 24)
    assert (dataset / 'test_ids.json').read_bytes() == frozen
    assert make_split(dataset, 20, True) == (test, train)


def payload():
    return {'answers': {'triage': {'choice': 'urgent', 'confidence': .91,
                                 'probabilities': {k: .91 if k == 'urgent' else .03 for k in config.LABELS}}}}


def test_laya_jev_and_fallback_payloads():
    data = payload()
    assert parse_response(data)['input_tokens'] is None
    data['usage'] = {'input_tokens': 234}
    assert parse_response(data)['input_tokens'] == 234
    answer = data['answers']['triage'].copy()
    del answer['confidence']
    assert parse_response({'answer': {'triage': answer}})['confidence'] == .91
    assert parse_response({'data': {'answer': answer}, 'usage': {'prompt_tokens': 12}})['input_tokens'] == 12
    with pytest.raises(ValueError, match='keys'):
        parse_response({'unexpected': 'body must not appear in error'})
    data['answers']['triage']['confidence'] = float('nan')
    with pytest.raises(ValueError):
        parse_response(data)


def test_retries_and_request_shape():
    calls, delays = [], []
    class Response:
        def __init__(self, code):
            self.status_code = code
        def json(self):
            return payload()
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response(429 if len(calls) < 3 else 200)
    email = {'id': '1', 'subject': 'Hi a@example.com', 'text': 'x' * 800, 'from_domain': 'example.com'}
    row = request_one(email, 'jev', 'secret', post=post, sleep=delays.append)
    assert delays == [1, 2] and row['pred'] == 'urgent'
    request = calls[-1][1]['json']
    assert request['questions'] == config.QUESTIONS and request['model'] == config.JEV_MODEL
    assert len(request['state']['text']) == 500 and '@' not in request['state']['subject']
    assert not calls[-1][1]['allow_redirects']
    calls.clear()
    request_one(email, 'laya', post=post, sleep=lambda _: None)
    assert 'model' not in calls[-1][1]['json']


def test_resumable_run(dataset, monkeypatch):
    import run_model
    make_split(dataset, 20)
    calls = []
    def fake(email, model, key):
        calls.append(email['id'])
        return {'id': email['id'], **parse_response(payload()), 'latency_ms': 10}
    monkeypatch.setattr(run_model, 'request_one', fake)
    run(dataset, 'laya')
    run(dataset, 'laya')
    assert len(calls) == 20


def test_known_scores_and_pairing():
    labels = {'1': 'urgent', '2': 'fyi', '3': 'junk', '4': 'needs_reply'}
    emails = {i: {'subject': '<unsafe>'} for i in labels}
    rows = [{'id': i, 'pred': 'urgent' if i == '2' else label, 'confidence': .95,
             'latency_ms': n * 10, 'input_tokens': 100} for n, (i, label) in enumerate(labels.items(), 1)]
    m = metrics(rows, labels, emails, 'jev')
    assert m['accuracy'] == .75 and m['confident_wrong_count'] == 1
    assert m['confident_wrong'][0]['id'] == '2'
    assert m['per_class']['urgent']['precision'] == .5
    assert m['macro_f1'] == pytest.approx(2 / 3)
    assert m['latency_p50_ms'] == 25 and m['latency_p95_ms'] == pytest.approx(38.5)
    assert m['cost_usd'] == pytest.approx(400 * .042 / 1_000_000)
    board = {'models': {'jev': m}, 'test_size': 4, 'partial': False, 'paired': []}
    html = render_html(board, True)
    assert '&lt;unsafe&gt;' in html and '<unsafe>' not in html and 'blur(8px)' in html
    a = {r['id']: r for r in rows}
    b = {i: {'pred': label} for i, label in labels.items()}
    pair = paired_comparisons({'a': a, 'b': b}, labels)[0]
    assert pair['a_right_b_wrong'] == 0 and pair['b_right_a_wrong'] == 1
    rows[0]['input_tokens'] = None
    assert metrics(rows, labels, emails, 'jev')['cost_usd'] is None


def test_score_ignores_train(dataset):
    test, train = make_split(dataset, 20)
    rows = [{'id': i, 'pred': 'urgent', 'confidence': .95, 'latency_ms': 10, 'input_tokens': None} for i in test + train]
    config.write_jsonl(dataset / 'results' / 'laya.jsonl', rows)
    board = score(dataset)
    assert board['models']['laya']['n'] == 20
    assert (dataset / 'results' / 'scoreboard.html').exists()


def test_default_data_dir_and_repository_guard(monkeypatch, tmp_path):
    monkeypatch.delenv('INBOX_TEST_DATA', raising=False)
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    path = config.resolve_data_dir()
    assert path == tmp_path / 'inbox-test-data' and config.REPO not in path.parents
    with pytest.raises(ValueError):
        config.resolve_data_dir(config.ROOT / 'data')
    with pytest.raises(ValueError):
        config.resolve_data_dir(config.SYNTHETIC_DIR, allow_synthetic=False)


def test_labels_resume_skip_and_undo(dataset):
    (dataset / 'labels.jsonl').unlink()
    app = LabelStore(dataset)
    first = app.state()['email']['id']
    app.action('urgent', first)
    assert LabelStore(dataset).state()['count'] == 1
    second = app.state()['email']['id']
    app.action('skip', second)
    app.action('undo', None)
    assert app.state()['email']['id'] == second
    app.action('undo', None)
    assert app.state()['email']['id'] == first
    with pytest.raises(ValueError):
        app.action('junk', 'stale-id')


def test_training_cases_no_leakage(dataset):
    test, train = make_split(dataset, 20)
    cases = build_cases(dataset)
    training, calibration = calibration_split(cases)
    assert len(training) == 54 and len(calibration) == 6
    assert not {c['id'] for c in training} & {c['id'] for c in calibration}
    assert not {c['id'] for c in cases} & set(test)
    assert sorted(cases[0]['gold']['triage']['probabilities'].values()) == pytest.approx([.03, .03, .03, .91])
    config.write_json(dataset / 'train_ids.json', train + [test[0]])
    with pytest.raises(ValueError, match='disjoint'):
        build_cases(dataset)
