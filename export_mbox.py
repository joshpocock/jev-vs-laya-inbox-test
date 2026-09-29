"""Export and scrub a Takeout mbox or folder of eml files."""
import hashlib
import mailbox
import re
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

import config


def scrub_pii(text, sender_name=''):
    text = re.sub(r'(?i)\b[\w.+%-]+@[\w.-]+\.[a-z]{2,}\b', '[email]', text)
    text = re.sub(r'(?i)\b(?:https?://|www\.)[^\s<>]+', '[link]', text)
    # Any other URI scheme (app deep links like zoommtg:// or slack:// can carry meeting ids).
    text = re.sub(r'(?i)\b[a-z][a-z0-9+.-]{1,30}://[^\s<>]*', '[link]', text)
    # Formatted telephone numbers; bare digit runs are handled separately.
    text = re.sub(r'(?<!\w)\+?\d[\d ()\.\-]{5,}\d(?!\w)',
                  lambda m: '[phone]' if 7 <= len(re.sub(r'\D', '', m[0])) <= 15
                  and re.search(r'[ ()\.\-+]', m[0]) else m[0], text)
    text = re.sub(r'\d{6,}', '[number]', text)
    if sender_name.strip():
        text = re.sub(r'(?<!\w)' + re.escape(sender_name.strip()) + r'(?!\w)', '', text, flags=re.I)
    return text.strip()


class PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        if tag in ('br', 'p', 'div', 'li'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def body_text(message):
    plain, html = [], []
    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() == 'attachment':
            continue
        kind = part.get_content_type()
        if kind not in ('text/plain', 'text/html'):
            continue
        raw = part.get_payload(decode=True)
        if raw is None:
            continue
        try:
            value = raw.decode(part.get_content_charset() or 'utf-8', errors='replace')
        except LookupError:
            value = raw.decode('utf-8', errors='replace')
        (plain if kind == 'text/plain' else html).append(value)
    if any(p.strip() for p in plain):
        return '\n'.join(plain)
    parser = PlainHTML()
    parser.feed('\n'.join(html))
    return ''.join(parser.parts)


def clean_body(text):
    try:
        from laya.email import clean_email_body
    except ImportError:
        text = re.split(r'(?im)^\s*(?:On .{0,300}wrote:|-----Original Message-----)', text)[0]
        text = re.split(r'(?im)^\s*(?:--|Best regards,?|Kind regards,?|Regards,?|Thanks,?)\s*$', text)[0]
        return text.strip()
    return clean_email_body(text, max_chars=max(3000, len(text)))


def parse_message(message, owners=(), sent_folder=False):
    senders = getaddresses(message.get_all('From', []))
    if sent_folder or any(address.lower() in owners for _, address in senders):
        return None
    folders = ','.join(str(message.get(k, '')) for k in ('X-Gmail-Labels', 'X-Folder', 'X-Mozilla-Keys'))
    if re.search(r'(?i)(?:^|[,/\\])\s*(?:sent|sent mail|sent items|\\sent)\s*(?:$|[,/\\])', folders):
        return None
    name, address = senders[0] if senders else ('', '')
    text = scrub_pii(clean_body(body_text(message)), name)
    if not text:
        return None
    mid = str(message.get('Message-ID', '')).strip()
    # Missing Message-ID: stable hash of source bytes, instead of merging all such mail.
    identity = mid.encode() if mid else message.as_bytes()
    try:
        date = parsedate_to_datetime(str(message.get('Date', '')))
        date = date.replace(tzinfo=date.tzinfo or timezone.utc).astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        date = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return {'id': hashlib.sha256(identity).hexdigest(), 'date': date.isoformat(),
            'from_domain': address.rpartition('@')[2].lower(),
            'subject': scrub_pii(str(message.get('Subject', '')), name), 'text': text[:500]}


def export(source, directory, owners=(), limit=1500):
    if config.REPO == source.resolve() or config.REPO in source.resolve().parents:
        raise ValueError('Raw exports must be outside the repository')
    config.resolve_data_dir(directory, allow_synthetic=False)
    owners = {o.lower() for o in owners}
    rows = {}
    def add(msg, sent=False):
        row = parse_message(msg, owners, sent)
        if row and (row['id'] not in rows or row['date'] > rows[row['id']]['date']):
            rows[row['id']] = row
    if source.is_dir():
        for path in sorted(source.rglob('*')):
            if path.suffix.lower() == '.eml':
                with path.open('rb') as stream:
                    add(BytesParser(policy=policy.default).parse(stream),
                        any(p.lower() in ('sent', 'sent mail', 'sent items') for p in path.parts[:-1]))
    else:
        box = mailbox.mbox(source, factory=lambda f: BytesParser(policy=policy.default).parse(f), create=False)
        try:
            for msg in box:
                add(msg, source.stem.lower() in ('sent', 'sent mail', 'sent items'))
        finally:
            box.close()
    result = sorted(rows.values(), key=lambda r: (r['date'], r['id']), reverse=True)[:limit]
    config.write_jsonl(directory / 'emails.jsonl', result)
    return result


def main():
    p = config.parser(__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--owner-email', action='append', default=[])
    p.add_argument('--limit', type=int, default=1500)
    args = p.parse_args()
    if args.limit < 1:
        p.error('--limit must be positive')
    print(f'Exported {len(export(args.source, config.data_dir(args), args.owner_email, args.limit))} emails')


if __name__ == '__main__':
    main()
