"""Shared configuration and privacy boundary for every command."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = next((p for p in (ROOT, *ROOT.parents) if (p / '.git').exists()), ROOT)
SYNTHETIC_DIR = ROOT / 'sample' / '.synthetic-data'
LABELS = {
    'urgent': 'A real person or critical system needs action from me today; time-sensitive.',
    'needs_reply': 'A real person expects a reply from me, but not urgently.',
    'fyi': 'Informational and useful to me, no reply needed: receipts, account and security notices, notifications from tools I use, updates I rely on.',
    'junk': 'Anything I would not care about losing: promotions, marketing, cold outreach, spam, and newsletters or creator emails I do not read.',
}
# Criteria revised 2026-09-28 during labelling, BEFORE any model run: the first wording called
# every subscribed newsletter "fyi", but the owner labels unread newsletters as junk. Graded
# against the owner's real rule, so every model gets that rule too.

# Secondary score, pre-registered 2026-09-28 before any model run: does the model catch what
# needs the owner? The fyi/junk line is a personal-taste call; this score is immune to it.
NEEDS_ME = {'urgent', 'needs_reply'}
INSTRUCTIONS = 'Triage this email for the owner of the inbox.'
QUESTIONS = {'triage': {'type': 'choice', 'instructions': INSTRUCTIONS, 'criteria': LABELS}}
JEV_ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
LAYA_ENDPOINT = 'http://127.0.0.1:8000/v1/systemone'
JEV_MODEL = 'jev-1.13.0'
JEV_PRICE_PER_M_INPUT = 0.042


def resolve_data_dir(value=None, allow_synthetic=True):
    path = Path(value or os.environ.get('INBOX_TEST_DATA') or
                Path(os.environ.get('USERPROFILE', Path.home())) / 'inbox-test-data').expanduser().resolve()
    if path == REPO or REPO in path.parents:
        if not (allow_synthetic and path == SYNTHETIC_DIR.resolve()):
            raise ValueError('DATA_DIR must be outside the repository; only sample/.synthetic-data is reserved for generated fixtures.')
    return path


DATA_DIR = resolve_data_dir()


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--data-dir', help='External data directory, or sample/.synthetic-data for fake fixtures only')
    return p


def data_dir(args):
    path = resolve_data_dir(args.data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path):
    if not path.exists():
        return []
    # Split on '\n' only: str.splitlines() also breaks on U+2028/U+0085, which real email
    # bodies contain inside JSON strings (found on the first Gmail snapshot, 2026-09-28).
    return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    temp.replace(path)


def api_key(directory):
    key = os.environ.get('TYPESAFE_API_KEY')
    if not key and (directory / '.env').exists():
        for line in (directory / '.env').read_text(encoding='utf-8').splitlines():
            name, sep, value = line.strip().removeprefix('export ').partition('=')
            if sep and name.strip() == 'TYPESAFE_API_KEY':
                key = value.strip().strip('\"\'')
    if not key:
        raise ValueError('Set TYPESAFE_API_KEY in the environment or DATA_DIR/.env')
    return key


if __name__ == '__main__':
    args = parser(__doc__).parse_args()
    print(resolve_data_dir(args.data_dir))
