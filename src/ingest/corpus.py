"""Shared corpus definition: config.yaml decides what is in, nothing else does."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config():
    return yaml.safe_load(open(ROOT / "config.yaml"))


def in_corpus(wanted, competition_id, season_name):
    if wanted["include"] == "all":
        return competition_id not in wanted.get("exclude", [])
    sel = wanted["include"].get(competition_id)
    return sel == "all" or (sel is not None and season_name in sel)
