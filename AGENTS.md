# AGENTS.md: run the Jev vs Laya inbox test

You are helping a user run this test on their own inbox. Walk them through it one step at a time. Explain each step in one plain sentence before running it, run it, report the result, then move to the next. Stop and hand over to the user wherever this file says **HUMAN STEP**.

## What the test does

Three classifiers get the same emails and the same question: "Is this urgent, needs a reply, FYI, or junk?"

1. **Jev**: TypeSafe's paid hosted decision model (needs an API key).
2. **Laya**: the free open-source model, running locally.
3. **Laya fine-tuned**: Laya trained on the user's own labelled emails.

The user's hand labels are the answer key. `score.py` builds a scoreboard with accuracy, confidently-wrong answers, "caught what needs me", speed and cost.

## Hard rules (never break these)

1. **Never read, print, summarise or quote email content.** Do not `cat`, `head`, `grep` or open anything in the data folder (`emails.jsonl`, `labels.jsonl`, `results/`, raw exports). The scripts print counts only; report those. If the user wants to look at emails, they open the files themselves.
2. **Never print, echo or log the TypeSafe API key.** Tell the user to put it in `<DATA_DIR>/.env` as `TYPESAFE_API_KEY=...` themselves, or set it as an environment variable. Never ask them to paste it into chat.
3. **All data stays in DATA_DIR, outside this repo.** The default is `~/inbox-test-data` (Windows: `%USERPROFILE%\inbox-test-data`); it can be overridden with `INBOX_TEST_DATA`. The tools refuse to write inside the repo. Never commit data.
4. **Never run `split.py --force` after any model has run.** The test set is frozen. Re-splitting after seeing results is cheating.
5. **Never re-run a model because the user dislikes its result.** Re-run only after a real error, and archive the old `results/<model>.jsonl` first.
6. **Ask before anything that sends email text off the machine:**
   - Jev (to TypeSafe);
   - `draft_labels.py --method claude` (to Anthropic).
   Say what leaves the machine: subject, sender domain and the first 500 scrubbed characters.
7. **Labels are a HUMAN STEP.** Never label emails yourself, and never accept all drafts for the user.

## The flow

Run commands from the repo root. On Windows, use PowerShell; if venv activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first (it affects only that window).

### 0. Setup (once)
```bash
python -m venv ~/inbox-test-data/.venv
# Windows:    & "$env:USERPROFILE\inbox-test-data\.venv\Scripts\Activate.ps1"
# Mac/Linux:  source ~/inbox-test-data/.venv/bin/activate
pip install torch            # NVIDIA RTX 50-series: pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "laya[serve]" requests pytest
python -m pytest tests -q    # should pass, no network or GPU needed
python fetch_model.py        # downloads the Laya checkpoint (~850 MB) into DATA_DIR/models/laya-base
```
Check the GPU with `python -c "import torch; print(torch.cuda.is_available())"`. No GPU is fine: Laya runs on CPU, just slower, and training will be slow.

### 1. Get the emails
- **Default (any Gmail):** HUMAN STEP.
  1. Go to takeout.google.com.
  2. Deselect all, then choose Mail, then only the Inbox label, in mbox format.
  3. Download the export, unzip it, and put `Inbox.mbox` at `<DATA_DIR>/raw/Inbox.mbox`.
  4. Then run:
     ```bash
     python export_mbox.py <DATA_DIR>/raw/Inbox.mbox --owner-email THEIR_ADDRESS --limit 1500
     ```
- **Optional fast path:** `python export_gmail.py --owner-email THEIR_ADDRESS --limit 1500` pulls from the Gmail API read-only. It needs the Google Workspace CLI (`gws auth login` with a Gmail scope); if that isn't set up, use Takeout.

Report the count. Have the user spot-check `emails.jsonl` **themselves** to confirm no names or addresses slipped through.

### 2. Labels (HUMAN STEP, ~15-40 min)
- **Optional:** pre-filled suggestions. Ask first (see rule 6), then run `python draft_labels.py --method claude --yes --limit 500`.
- Start the app: `python label_app.py --target 450`
- Tell the user to open http://127.0.0.1:8765. The keys:
  - **1** urgent, **2** needs a reply, **3** FYI, **4** junk;
  - **Enter** accepts a suggestion; **S** skips; **U** undoes.
- The bucket definitions are in `config.py` (`LABELS`). The user must label by those, for themselves, consistently. Tip: "If this vanished, would I care?" Yes means FYI, no means junk.
- Wait for the user to finish. Don't poll the data.

### 3. Freeze the test set
```bash
python split.py --test-size 300
```
Report the per-bucket counts.
- Flag any bucket with fewer than about 20 test emails: too few to judge alone.
- Point out the "lazy guess" baseline: the share of the biggest bucket.

### 4. Round 1: Jev (ask first, rule 6)
The user must have set `TYPESAFE_API_KEY` themselves (rule 2). Then:
```bash
python run_model.py --model jev --workers 4
```

### 5. Round 2: Laya (untrained)
In a **second terminal** (activate the venv there too), start the server and leave it running:
```bash
LAYA_DEVICE=cuda python serve_laya.py --checkpoint <DATA_DIR>/models/laya-base
# Windows: $env:LAYA_DEVICE='cuda'; python serve_laya.py --checkpoint "$env:USERPROFILE\inbox-test-data\models\laya-base"
```
Wait for "Application startup complete", then in the first terminal:
```bash
python run_model.py --model laya
```
Use `LAYA_DEVICE=cpu` if there's no GPU or you get an out-of-memory error.

### 6. Scoreboard
```bash
python score.py --blur-subjects
```
Tell the user to open `<DATA_DIR>/results/scoreboard.html`. Report the headline numbers from the command's JSON output only:
- accuracy with its range;
- the lazy guess;
- confidently wrong;
- caught what needs me;
- median latency and cost.

Do not print subjects.

### 7. Round 3: train Laya on the user's own labels
Stop the Laya server (Ctrl+C) to free the GPU, then:
```bash
python finetune_laya.py --base-checkpoint <DATA_DIR>/models/laya-base --epochs 4
```
It trains only on the training split and never touches the test set. Report the minutes and peak GPU memory it prints.
- **Out of memory:** tell the user. Do not tweak settings to chase a better score.
- **Try it first:** `--dry-run` runs a tiny version to check it works.

### 8. Run the trained model and final score
In the second terminal:
```bash
LAYA_DEVICE=cuda python serve_laya.py --checkpoint <DATA_DIR>/checkpoints/laya-inbox-ft
```
Then:
```bash
python run_model.py --model laya-ft
python score.py --blur-subjects
```
Summarise the result honestly:
- If two models' accuracy ranges overlap, call it a tie.
- A model only "works" if it clearly beats the lazy guess.
- Speed compares a local GPU with a network API, so it is not a like-for-like race.

## If something breaks
- **Jev 401:** the key is wrong or not set. The user fixes it themselves.
- **Jev 429:** it retries on its own.
- **Port 8000 busy:** stop any old Laya server.
- **Test failures after an edit:** run `python -m pytest tests -q`. Never delete a test to make it pass.
