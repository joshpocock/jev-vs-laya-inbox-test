"""Serve an already downloaded local checkpoint using Laya's HTTP app."""
import os
from pathlib import Path

import config


def main():
    p = config.parser(__doc__)
    p.add_argument('--checkpoint', type=Path, help='Defaults to DATA_DIR/checkpoints/laya-inbox-ft')
    args = p.parse_args()
    path = (args.checkpoint or config.data_dir(args) / 'checkpoints' / 'laya-inbox-ft').resolve()
    if not (path / 'rl_agent_config.json').exists():
        p.error('Local checkpoint is missing rl_agent_config.json')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['LAYA_HOST'] = '127.0.0.1'
    # github.com/NandhaKishorM/laya/laya/serve.py:97-113,302-322: LAYA_MODELS only selects aliases.
    # github.com/NandhaKishorM/laya/laya/router.py:362-392: Router(models=...) accepts local paths.
    # Map all routes so language detection cannot accidentally load another model.
    from laya.router import Router
    import laya.serve
    router = Router(models={k: str(path) for k in ('english', 'multilingual', 'typed-decisions')},
                    device=os.environ.get('LAYA_DEVICE', 'cuda'), max_loaded=1)
    router.preload(['english'])
    laya.serve.build_router = lambda: router
    laya.serve.main()


if __name__ == '__main__':
    main()
