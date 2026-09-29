"""Download the published English Laya checkpoint into DATA_DIR/models/laya-base.

One-time setup step (~850 MB). Fetches only the four pieces the tools need:
model.safetensors, rl_agent_config.json, encoder/ and tokenizer/.
"""
from huggingface_hub import snapshot_download

import config


def main():
    p = config.parser(__doc__)
    args = p.parse_args()
    target = config.data_dir(args) / 'models' / 'laya-base'
    snapshot_download('convaiinnovations/laya', local_dir=str(target),
                      allow_patterns=['model.safetensors', 'rl_agent_config.json', 'encoder/*', 'tokenizer/*'])
    print(f'Laya base checkpoint ready in {target}')


if __name__ == '__main__':
    main()
