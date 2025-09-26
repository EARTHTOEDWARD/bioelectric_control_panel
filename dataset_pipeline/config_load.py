import json
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        cfg = json.load(f)
    # basic normalization/guards
    if cfg['spectral']['welch']['overlap'] < 0 or cfg['spectral']['welch']['overlap'] >= 1:
        cfg['spectral']['welch']['overlap'] = 0.5
    return cfg
