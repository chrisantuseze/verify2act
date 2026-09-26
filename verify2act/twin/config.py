"""Twin configuration: the YAML in ``configs/twin/`` as a plain dict, with optional deep-merged overrides."""

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "twin" / "dofbot_twin.yaml"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _merge(cfg, overrides)
    return cfg
