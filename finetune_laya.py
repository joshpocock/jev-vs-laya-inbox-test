"""Single-GPU RLCD fine-tuning using only the training split, with calibration."""
import json
import math
import os
import random
import time
from pathlib import Path

import config
from run_model import model_state

# Reference: github.com/NandhaKishorM/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb,
# code cell 6 lines 19-50 (items), cell 8 lines 12-54 (collate and temperature),
# cell 8 lines 56-287 (RLCD, checkpoint, calibration). The notebook writes a DDP
# executable, not an importable trainer. Adapt only the orchestration here; reuse
# laya/common.py:384-406 (model), 410-438 (proper_reward), build_sequence and QTYPES.
# docs/finetune_browser_agent.md describes the same single-GPU RLCD adaptation.


def build_cases(directory, smoothing=0.09):
    if not 0 <= smoothing < 0.75:
        raise ValueError('Label smoothing is total off-label mass, in [0, 0.75)')
    ids = json.loads((directory / 'train_ids.json').read_text())
    test = set(json.loads((directory / 'test_ids.json').read_text()))
    if set(ids) & test or len(ids) != len(set(ids)):
        raise ValueError('Training ids must be unique and disjoint from test ids')
    emails = {r['id']: r for r in config.read_jsonl(directory / 'emails.jsonl')}
    labels = {r['id']: r['label'] for r in config.read_jsonl(directory / 'labels.jsonl')}
    cases = []
    for identity in ids:
        if identity not in emails or labels.get(identity) not in config.LABELS:
            raise ValueError('Missing training email or valid human label')
        gold = {k: 1 - smoothing if k == labels[identity] else smoothing / 3 for k in config.LABELS}
        cases.append({'id': identity, 'state': model_state(emails[identity]),
                      'questions': config.QUESTIONS, 'gold': {'triage': {'probabilities': gold}}})
    return cases


def calibration_split(cases):
    order = list(cases)
    random.Random(42).shuffle(order)
    n = min(400, len(order) // 10)
    if n < 1:
        raise ValueError('At least 10 training cases are required to hold out calibration')
    return order[n:], order[:n]


def train(directory, base, epochs=4, smoothing=0.09, dry_run=False):
    # No implicit Hub downloads. Operator provisions the local base first.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    base = Path(base).resolve()
    for name in ('rl_agent_config.json', 'model.safetensors', 'encoder/config.json', 'tokenizer'):
        if not (base / name).exists():
            raise ValueError(f'Local base checkpoint is missing {name}')
    cases = build_cases(directory, smoothing)
    training, calibration = calibration_split(cases)
    if dry_run:
        if len(training) < 16:
            raise ValueError('Dry run requires at least 16 training examples after calibration holdout')
        training = training[:16]
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from laya.common import QTYPES, build_model, build_sequence, proper_reward
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    cfg = json.loads((base / 'rl_agent_config.json').read_text())
    cfg.update(gradient_checkpointing=True, max_len=512, head_max_len=256, max_tokens_per_batch=2048)
    tok = AutoTokenizer.from_pretrained(str(base / 'tokenizer'), local_files_only=True)
    model = build_model(cfg, encoder_dir=str(base / 'encoder'))
    model.load_state_dict(load_file(str(base / 'model.safetensors')), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.head_checkpointing = True
    model.to('cuda').train()

    def item(case):
        q = case['questions']['triage']
        ids, markers = build_sequence(tok, case['state'], {'t': 'choice', 'ins': q['instructions'],
                                      'crit': q['criteria']}, cfg['max_len'], cfg['head_max_len'])
        if len(markers) != 4:
            raise ValueError('Token budget truncated choice markers')
        return {'ids': ids, 'markers': markers,
                'target': [case['gold']['triage']['probabilities'][k] for k in config.LABELS]}

    training, calibration = [item(c) for c in training], [item(c) for c in calibration]

    def collate(items):
        n, length = len(items), max(len(it['ids']) for it in items)
        ids = torch.full((n, length), tok.pad_token_id, dtype=torch.long, device='cuda')
        attention = torch.zeros_like(ids)
        for i, it in enumerate(items):
            ids[i, :len(it['ids'])] = torch.tensor(it['ids'], device='cuda')
            attention[i, :len(it['ids'])] = 1
        positions = torch.tensor([it['markers'] for it in items], device='cuda')
        mask = torch.ones((n, 4), dtype=torch.bool, device='cuda')
        qtypes = torch.full((n,), QTYPES['choice'], dtype=torch.long, device='cuda')
        target = torch.tensor([it['target'] for it in items], device='cuda')
        return (ids, attention, positions, mask, qtypes), target

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'encoder.' in n], 'lr': 2.5e-5},
        {'params': [p for n, p in model.named_parameters() if 'encoder.' not in n], 'lr': 1e-4},
    ], weight_decay=0.01)
    # A dry run makes exactly two optimizer updates using the same 16 examples.
    effective = 16 if dry_run else 32
    n_epochs = 2 if dry_run else epochs
    updates = n_epochs * math.ceil(len(training) / effective)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    output = directory / 'checkpoints' / ('laya-inbox-ft-dry-run' if dry_run else 'laya-inbox-ft')

    def save(path):
        path.mkdir(parents=True, exist_ok=True)
        save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}, str(path / 'model.safetensors'))
        model.encoder.config.save_pretrained(str(path / 'encoder'))
        tok.save_pretrained(str(path / 'tokenizer'))
        config.write_json(path / 'rl_agent_config.json', cfg)

    for epoch in range(n_epochs):
        random.Random(42 + epoch).shuffle(training)
        sigma = 0.4 - 0.3 * epoch / max(1, n_epochs - 1)
        for offset in range(0, len(training), effective):
            group = training[offset:offset + effective]
            optimizer.zero_grad(set_to_none=True)
            for b in range(0, len(group), 4):
                chunk = group[b:b + 4]
                batch, target = collate(chunk)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits, act = model(*batch)
                logits = logits.float()
                mask, qtypes = batch[3], batch[4]
                noise = torch.randn((4,) + logits.shape, device='cuda') * sigma
                noise = noise - noise.mean(-1, keepdim=True)
                z = logits.detach().unsqueeze(0) + noise
                q = torch.softmax(z, -1)
                with torch.no_grad():
                    reward = proper_reward(q, target.unsqueeze(0), qtypes, mask, w_sph=0.75, w_rps=1.0)
                    advantage = reward - reward.mean(0, keepdim=True)
                    advantage = advantage / (advantage.std() + 1e-6)
                logp = -((z - logits.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
                rl = -(advantage * logp).mean()
                ce = -(target * torch.log_softmax(logits, -1)).sum(-1).mean()
                loss = (rl + ce) * len(chunk) / len(group) + 0.0 * act.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        save(output / 'checkpoint_latest')
        print(f'Epoch {epoch + 1}/{n_epochs} completed')
    del optimizer, scaler, scheduler
    torch.cuda.empty_cache()
    model.eval()
    logits_list, targets = [], []
    with torch.no_grad():
        for offset in range(0, len(calibration), 4):
            batch, target = collate(calibration[offset:offset + 4])
            with torch.autocast('cuda', dtype=torch.float16):
                logits, _ = model(*batch)
            logits_list.append(logits.float().cpu())
            targets.append(target.cpu())
    temperature = 1.0
    if len(calibration) >= 10:
        z, target = torch.cat(logits_list), torch.cat(targets)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
        def closure():
            opt.zero_grad()
            loss = -(target * torch.log_softmax(z / log_t.exp(), -1)).sum(-1).mean()
            loss.backward()
            return loss
        try:
            opt.step(closure)
            temperature = float(log_t.exp().clamp(0.1, 10).item())
            if not math.isfinite(temperature):
                raise ValueError('Non-finite temperature')
        except (RuntimeError, ValueError):
            temperature = 1.2
            print('Temperature fitting failed; using recipe fallback 1.2')
    else:
        print('Fewer than 10 calibration cases; recipe uses temperature 1.0')
    temperatures = [1.2, 1.2, 1.2]
    temperatures[QTYPES['choice']] = temperature
    cfg.update(temperature=temperatures, fine_tuned=True, model_name='laya-inbox-ft')
    cfg.pop('temperature_by_options', None)
    save(output)
    config.write_json(output / 'training_summary.json', {
        'train_cases': len(training), 'calibration_cases': len(calibration), 'dry_run': dry_run,
        'epochs': n_epochs, 'temperature': temperature, 'label_smoothing': smoothing,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
    })
    print(f'Elapsed: {time.perf_counter() - started:.1f}s; peak GPU: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB; temperature: {temperature:.4f}')


if __name__ == '__main__':
    p = config.parser(__doc__)
    p.add_argument('--base-checkpoint', type=Path, required=True, help='Already downloaded local Laya checkpoint')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--label-smoothing', type=float, default=0.09, help='Total off-label mass; 0.09 gives 0.91/0.03/0.03/0.03')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.epochs < 1:
        p.error('--epochs must be positive')
    train(config.data_dir(args), args.base_checkpoint, args.epochs, args.label_smoothing, args.dry_run)
