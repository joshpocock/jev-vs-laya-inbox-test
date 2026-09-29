"""Snapshot the newest Inbox emails straight from the Gmail API into DATA_DIR/emails.jsonl.

A faster alternative to Google Takeout + export_mbox.py. Same output, same cleaning:
each raw message goes through export_mbox.parse_message, so the PII scrub, sender-name
removal and 500-character cap are identical.

Read-only: it lists and reads messages and never modifies, labels or sends anything.
Auth: reuses the Google Workspace CLI login already on this machine (`gws auth login`),
which must include a Gmail scope. The token is read at runtime and never printed.
Emails go from Gmail to DATA_DIR on this machine only.

The result is a frozen snapshot. Re-running replaces emails.jsonl, so do it BEFORE
labelling and never after split.py.
"""
import base64
import concurrent.futures
import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from email import policy
from email.parser import BytesParser

import config
from export_mbox import parse_message

API = 'https://gmail.googleapis.com/gmail/v1/users/me'


def access_token():
    p = subprocess.run(['npx', '@googleworkspace/cli', 'auth', 'export', '--unmasked'],
                       capture_output=True, text=True, encoding='utf-8', shell=True)
    if '{' not in p.stdout:
        raise SystemExit('Google login not found. Run `gws auth login` with a Gmail scope first.')
    c = json.loads(p.stdout[p.stdout.index('{'):])
    data = urllib.parse.urlencode({'client_id': c['client_id'], 'client_secret': c['client_secret'],
                                   'refresh_token': c['refresh_token'],
                                   'grant_type': 'refresh_token'}).encode()
    with urllib.request.urlopen(urllib.request.Request('https://oauth2.googleapis.com/token', data=data)) as r:
        return json.loads(r.read())['access_token']


def get(url, token):
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < 4:
                import time
                time.sleep(2 ** attempt)
                continue
            raise SystemExit(f'Gmail API error {e.code} on {url.split("?")[0]}')


def list_ids(token, limit):
    ids, page = [], None
    while len(ids) < limit:
        q = {'labelIds': 'INBOX', 'maxResults': min(500, limit - len(ids))}
        if page:
            q['pageToken'] = page
        r = get(f'{API}/messages?{urllib.parse.urlencode(q)}', token)
        ids += [m['id'] for m in r.get('messages', [])]
        page = r.get('nextPageToken')
        if not page:
            break
    return ids[:limit]


def fetch(mid, token):
    r = get(f'{API}/messages/{mid}?format=raw', token)
    raw = base64.urlsafe_b64decode(r['raw'] + '=' * (-len(r['raw']) % 4))
    return BytesParser(policy=policy.default).parsebytes(raw), 'SENT' in r.get('labelIds', [])


def export(directory, owners=(), limit=1500, fetch_extra=1.3):
    config.resolve_data_dir(directory, allow_synthetic=False)
    owners = {o.lower() for o in owners}
    token = access_token()
    # Fetch a little extra: owner-sent mail and empty bodies are dropped by the cleaner.
    ids = list_ids(token, int(limit * fetch_extra))
    rows = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for msg, sent in ex.map(lambda i: fetch(i, token), ids):
            row = parse_message(msg, owners, sent)
            if row and (row['id'] not in rows or row['date'] > rows[row['id']]['date']):
                rows[row['id']] = row
    result = sorted(rows.values(), key=lambda r: (r['date'], r['id']), reverse=True)[:limit]
    config.write_jsonl(directory / 'emails.jsonl', result)
    return len(ids), result


def main():
    p = config.parser(__doc__)
    p.add_argument('--owner-email', action='append', default=[])
    p.add_argument('--limit', type=int, default=1500)
    args = p.parse_args()
    if args.limit < 1:
        p.error('--limit must be positive')
    fetched, result = export(config.data_dir(args), args.owner_email, args.limit)
    # Counts only: never print subjects or bodies.
    print(f'Fetched {fetched} inbox messages, kept {len(result)} cleaned emails in DATA_DIR/emails.jsonl')


if __name__ == '__main__':
    main()
