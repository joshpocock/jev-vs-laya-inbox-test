# Jev vs Laya: test both on your own inbox

This is the tool from my video. I ran TypeSafe's **Jev** (paid, hosted) against the free, open-source **Laya** on 300 of my real emails, then trained Laya on my own inbox to see if it could catch up. This repo lets you run the same test on yours.

**Easiest way:** open this folder in **Claude Code** or **Codex** and say:

> Run the Jev vs Laya test on my inbox. Follow AGENTS.md.

The agent walks you through every step and stops where you need to act, like labelling. [AGENTS.md](AGENTS.md) has the full flow and the privacy rules it follows.

## What it measures

Every email goes into one of four buckets: **urgent**, **needs a reply**, **FYI** or **junk**. You label a few hundred by hand; that's the answer key. Each model then guesses the same 300 test emails, and `score.py` builds a scoreboard:

- **Accuracy**, with a range, so you can tell a real win from a tie.
- **The lazy guess:** what you'd score by always picking your most common bucket. A model has to beat this.
- **Confidently wrong:** errors made at 90%+ confidence. These are the ones that bury a real email.
- **Caught what needs me:** urgent plus needs a reply, versus everything else.
- **Speed and cost** per 1,000 emails.

## What you need

- Python 3.11+.
- An NVIDIA GPU to train Laya. Running Laya works on CPU, just slower.
- A TypeSafe API key for the Jev round (typesafe.ai).
- About 15-40 minutes to label emails.

## Keys and access: what you actually need

Nothing here gives this repo, or me, access to anything. Everything runs on your machine.

| Step | What you need | Where your data goes |
|---|---|---|
| Get your emails (default) | Nothing. Download your own inbox from Google Takeout (Mail, Inbox only, mbox) | Stays on your computer |
| Get your emails (optional) | Your own Google Workspace CLI login (`gws auth login`, Gmail read scope) for `export_gmail.py` | Stays on your computer |
| Laya | Nothing. The model downloads free from Hugging Face | Runs locally |
| Jev | **Your own TypeSafe API key** (typesafe.ai), in `~/inbox-test-data/.env` as `TYPESAFE_API_KEY=...` | Trimmed emails go to TypeSafe |
| Pre-filled label suggestions (optional) | Your own Claude Code login | Trimmed emails go to Anthropic, only with `--yes` |

A 300-email Jev run is roughly 150K-200K input tokens: under a cent at $0.042 per million. New TypeSafe accounts may not include free credit.

## Privacy

- All email data lives in `~/inbox-test-data` (Windows: `%USERPROFILE%\inbox-test-data`), **outside this repo**. The tools refuse to write inside it.
- Before anything reaches a model, the filter removes email addresses, phone numbers, links (including app links), long number strings and the sender's name. It keeps only the subject, the sender's domain and the first 500 characters.
- Laya runs entirely on your machine. Jev is an API, so the trimmed emails go to TypeSafe.
- The optional pre-filled label suggestions send the trimmed emails to Anthropic through the Claude CLI, and only if you pass `--yes`.
- The regex scrub is not perfect anonymisation. Spot-check `emails.jsonl` yourself before running Jev.

## Manual steps

The full commands, including Windows and Mac variants, are in [AGENTS.md](AGENTS.md). In short:

```bash
pip install torch "laya[serve]" requests pytest   # RTX 50-series: torch from the cu128 index
python fetch_model.py                              # Laya checkpoint, ~850 MB
python export_mbox.py ~/inbox-test-data/raw/Inbox.mbox --owner-email you@example.com   # from Google Takeout
python label_app.py --target 450                   # label at http://127.0.0.1:8765
python split.py --test-size 300                    # freeze the test set
python run_model.py --model jev                    # needs TYPESAFE_API_KEY
python serve_laya.py --checkpoint ~/inbox-test-data/models/laya-base   # second terminal
python run_model.py --model laya
python finetune_laya.py --base-checkpoint ~/inbox-test-data/models/laya-base --epochs 4
python serve_laya.py --checkpoint ~/inbox-test-data/checkpoints/laya-inbox-ft          # second terminal
python run_model.py --model laya-ft
python score.py                                    # open ~/inbox-test-data/results/scoreboard.html
```

Test the plumbing with no email, network or GPU: `python -m pytest tests -q`, and the synthetic sample in `sample/`.

## Credits and licence

- The code is MIT licensed (see LICENSE).
- `finetune_laya.py` adapts the training recipe from [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations (Apache-2.0) to a single GPU. See NOTICE.
- Jev is a product of TypeSafe AI. This repo is not affiliated with TypeSafe or Convai.
