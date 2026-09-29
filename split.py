"""Create a deterministic, frozen stratified test split."""
import json
import random
from collections import Counter

import config


def stratify(labels, size, seed=42):
    if not 0 < size < len(labels):
        raise ValueError('Test size must be positive and smaller than the labelled dataset')
    groups = {label: sorted(i for i, value in labels.items() if value == label) for label in config.LABELS}
    quotas = {label: len(ids) * size / len(labels) for label, ids in groups.items()}
    counts = {label: int(n) for label, n in quotas.items()}
    for label in sorted(groups, key=lambda k: (-(quotas[k] - counts[k]), k))[:size - sum(counts.values())]:
        counts[label] += 1
    rng = random.Random(seed)
    test, train = [], []
    for label, ids in groups.items():
        rng.shuffle(ids)
        test.extend(ids[:counts[label]])
        train.extend(ids[counts[label]:])
    return sorted(test), sorted(train)


def make_split(directory, size=300, force=False):
    if (directory / 'test_ids.json').exists() and not force:
        raise FileExistsError('Frozen test split exists; use --force only to intentionally replace it')
    emails = {r['id'] for r in config.read_jsonl(directory / 'emails.jsonl')}
    labels = {r['id']: r['label'] for r in config.read_jsonl(directory / 'labels.jsonl')}
    if any(label not in config.LABELS for label in labels.values()) or not labels.keys() <= emails:
        raise ValueError('Invalid labels or labels with missing emails')
    test, train = stratify(labels, size)
    config.write_json(directory / 'train_ids.json', train)
    config.write_json(directory / 'test_ids.json', test)
    for name, ids in [('test', test), ('train', train)]:
        print(name, dict(Counter(labels[i] for i in ids)))
    return test, train


if __name__ == '__main__':
    p = config.parser(__doc__)
    p.add_argument('--test-size', type=int, default=300)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    make_split(config.data_dir(args), args.test_size, args.force)
