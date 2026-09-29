"""Generate 80 invented emails and labels; optionally initialize a fake-data run."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

TOPICS = ['website launch', 'studio booking', 'invoice review', 'design handoff', 'team workshop',
          'product demo', 'delivery schedule', 'support rota', 'research plan', 'office move',
          'quarterly report', 'training session', 'supplier review', 'event registration', 'client proposal',
          'release checklist', 'budget meeting', 'equipment order', 'project brief', 'account migration']


def generate():
    emails, labels = [], []
    templates = {
        'urgent': ('Action today: {topic}', 'We are blocked on the {topic}. Please approve the pending change before 3 pm today so the team can proceed. The deadline cannot move.'),
        'needs_reply': ('Your thoughts on the {topic}', 'Could you send your preference for the {topic} when you have a chance? I need your reply to finish the plan. Next week is fine.'),
        'fyi': ('Update: {topic}', 'Here is the status update you subscribed to for the {topic}. Everything is on schedule. This message is for your records and no response is needed.'),
        'junk': ('Exclusive offer for your {topic}', 'We have never spoken, but our premium service could transform your {topic}. Buy our promotional package this week for a special discount.'),
    }
    for label, (subject, body) in templates.items():
        for n, topic in enumerate(TOPICS):
            identity = f'synthetic-{label}-{n:02d}'
            emails.append({'id': identity, 'date': (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=n)).isoformat(),
                           'from_domain': f'{label.replace("_", "-")}.example',
                           'subject': subject.format(topic=topic), 'text': body.format(topic=topic)})
            labels.append({'id': identity, 'label': label, 'labelled_at': '2026-01-21T00:00:00+00:00'})
    return emails, labels


def main():
    p = config.parser(__doc__)
    p.add_argument('--mock-results', action='store_true', help='Write deterministic fake model results for an offline plumbing demo')
    args = p.parse_args()
    emails, labels = generate()
    folder = Path(__file__).resolve().parent
    config.write_jsonl(folder / 'synthetic_emails.jsonl', emails)
    config.write_jsonl(folder / 'synthetic_labels.jsonl', labels)
    if args.data_dir:
        directory = config.data_dir(args)
        if directory != config.SYNTHETIC_DIR.resolve():
            p.error('Synthetic initialization is restricted to sample/.synthetic-data to protect real data')
        config.write_jsonl(directory / 'emails.jsonl', emails)
        config.write_jsonl(directory / 'labels.jsonl', labels)
        if args.mock_results:
            for model, every in [('jev', 9), ('laya', 5), ('laya-ft', 12)]:
                rows = []
                for i, row in enumerate(labels):
                    names = list(config.LABELS)
                    pred = names[(names.index(row['label']) + 1) % 4] if i % every == 0 else row['label']
                    rows.append({'id': row['id'], 'pred': pred, 'confidence': .94,
                                 'probabilities': {k: .94 if k == pred else .02 for k in names},
                                 'latency_ms': 50 + i, 'input_tokens': 200 if model == 'jev' else None})
                config.write_jsonl(directory / 'results' / f'{model}.jsonl', rows)
    elif args.mock_results:
        p.error('--mock-results requires --data-dir sample/.synthetic-data')
    print('Generated 80 fake emails and labels. Synthetic metrics are not benchmark evidence.')


if __name__ == '__main__':
    main()
