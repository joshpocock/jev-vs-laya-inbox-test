"""Local, resumable human labelling with keyboard shortcuts."""
import json
import secrets
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import config

PAGE = '''<!doctype html><meta charset="utf-8"><title>Inbox labelling</title>
<style>body{background:#10141c;color:#edf2fa;font:24px system-ui;margin:5vh auto;max-width:1100px;padding:24px}
button{font:22px system-ui;padding:20px;margin:8px;border:2px solid #59657a;border-radius:12px;cursor:pointer}
.draft{background:#a6efbd;border-color:#1f9b57}pre{white-space:pre-wrap;font:26px/1.5 system-ui;min-height:250px}
.blur{filter:blur(9px)}small{color:#aebbd0}</style>
<div id="progress"></div><h1 id="subject"></h1><small id="domain"></small><pre id="body"></pre>
<div id="buttons"></div><button onclick="act('skip')">S: Skip</button><button onclick="act('undo')">U: Undo</button>
<p><small>1-4: label | Enter: confirm draft | S: skip | U: undo</small></p>
<script>const token=TOKEN; let current=null,busy=false;
const labels=['urgent','needs_reply','fyi','junk'];
async function refresh(){const r=await fetch('/state');const s=await r.json();current=s.email;
document.getElementById('progress').textContent=s.count+' labelled / '+s.target+' | '+JSON.stringify(s.counts);
document.getElementById('subject').textContent=current?current.subject:'All done for this session';
document.getElementById('domain').textContent=current?current.from_domain:'';
document.getElementById('body').textContent=current?current.text:'';
for(const id of ['subject','body'])document.getElementById(id).className=s.blur?'blur':'';
const buttons=document.getElementById('buttons');buttons.replaceChildren();
labels.forEach((label,i)=>{const b=document.createElement('button');b.textContent=(i+1)+': '+label;
b.className=current&&current.draft===label?'draft':'';b.disabled=!current;b.onclick=()=>act(label);buttons.append(b)});}
async function act(action){if(busy)return;busy=true;try{const r=await fetch('/action',{method:'POST',
headers:{'Content-Type':'application/json','X-Token':token},body:JSON.stringify({action,id:current?.id})});
if(!r.ok)throw Error('Action rejected');await refresh()}catch(e){alert(e.message)}finally{busy=false}}
document.addEventListener('keydown',e=>{if(e.repeat)return;const k=e.key.toLowerCase();
if(['1','2','3','4','s','u','enter'].includes(k)){e.preventDefault();
if(labels[Number(k)-1])act(labels[Number(k)-1]);else if(k==='s')act('skip');else if(k==='u')act('undo');
else if(k==='enter'&&current?.draft)act(current.draft)}});refresh();</script>'''


class LabelStore:
    def __init__(self, directory, target=None, blur=False):
        self.path = directory / 'labels.jsonl'
        self.emails = config.read_jsonl(directory / 'emails.jsonl')
        self.labels = {r['id']: r for r in config.read_jsonl(self.path)}
        self.drafts = {r['id']: r['label'] for r in config.read_jsonl(directory / 'draft_labels.jsonl')}
        self.target = min(target or len(self.emails), len(self.emails))
        self.blur, self.skipped, self.history = blur, set(), []

    def state(self):
        count = sum(r['id'] in self.labels for r in self.emails)
        email = next((r for r in self.emails if r['id'] not in self.labels and r['id'] not in self.skipped), None) if count < self.target else None
        return {'email': {**email, 'draft': self.drafts.get(email['id'])} if email else None,
                'count': count, 'target': self.target, 'counts': dict(Counter(r['label'] for r in self.labels.values())), 'blur': self.blur}

    def action(self, action, identity):
        if action == 'undo':
            if self.history:
                identity, previous, was_skip = self.history.pop()
                if previous is None:
                    self.labels.pop(identity, None)
                else:
                    self.labels[identity] = previous
                if was_skip:
                    self.skipped.discard(identity)
            elif self.labels:
                # Across restarts, undo the most recently labelled email.
                identity = max(self.labels, key=lambda i: self.labels[i]['labelled_at'])
                del self.labels[identity]
        else:
            current = self.state()['email']
            if current is None or identity != current['id'] or action not in (*config.LABELS, 'skip'):
                raise ValueError('Stale or invalid action')
            self.history.append((identity, self.labels.get(identity), action == 'skip'))
            if action == 'skip':
                self.skipped.add(identity)
            else:
                self.labels[identity] = {'id': identity, 'label': action, 'labelled_at': datetime.now(timezone.utc).isoformat()}
        config.write_jsonl(self.path, self.labels.values())


def main():
    p = config.parser(__doc__)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--target', type=int)
    p.add_argument('--blur', action='store_true')
    args = p.parse_args()
    store = LabelStore(config.data_dir(args), args.target, args.blur)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, kind='application/json'):
            raw = body.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Frame-Options', 'DENY')
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.headers.get('Host') != f'127.0.0.1:{args.port}':
                return self.reply(403, '{}')
            if self.path == '/':
                self.reply(200, PAGE.replace('TOKEN', json.dumps(token)), 'text/html')
            elif self.path == '/state':
                self.reply(200, json.dumps(store.state()))
            else:
                self.reply(404, '{}')

        def do_POST(self):
            if self.path != '/action' or self.headers.get('X-Token') != token:
                return self.reply(403, '{}')
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 4096:
                    raise ValueError()
                body = json.loads(self.rfile.read(size))
                store.action(body['action'], body.get('id'))
            except (ValueError, KeyError, TypeError):
                return self.reply(400, '{}')
            self.reply(200, '{}')

    print(f'Open http://127.0.0.1:{args.port}')
    with HTTPServer(('127.0.0.1', args.port), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
