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
    """``base: <file>`` in a config (relative to it) loads that config first and deep-merges this one on top."""
    path = Path(path or DEFAULT_CONFIG)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(load_config(str(path.parent / base)), cfg)
    if overrides:
        _merge(cfg, overrides)
    return cfg
