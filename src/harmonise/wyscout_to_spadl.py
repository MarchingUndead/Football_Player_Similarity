"""SPADL layer for the secondary provider (Wyscout / Pappalardo, CC BY 4.0).

Same normalisation as the StatsBomb path (to_spadl.py): socceraction's
maintained converter -> 23-type SPADL vocabulary, 105x68 m pitch, acting team
attacks left -> right, one Parquet per competition-season, provider column set.
Output under data/spadl/wyscout/ — same schema and dtypes as spadl/statsbomb/
so downstream code can read either or both.

Provider precedence (§2.2/§2.7): Wyscout contributes only what StatsBomb lacks
— the big-five 2017/18 leagues and Euro 2016. WC2018 exists in both providers;
StatsBomb's copy is already in the corpus, so it is EXCLUDED here via
config.yaml. Never both.

Known fidelity gaps vs StatsBomb (phasewise_notes.txt has the full list):
no native carries (SPADL infers synthetic straight-line dribbles — §4.3),
no under_pressure, no xG, no freeze frames, shot placement 9-zone only.

The Pappalardo layout on disk keeps events/ and matches/ in subdirectories;
PublicWyscoutLoader expects one flat directory, so a temp dir of symlinks
bridges the two — the dvc-tracked raw tree is never modified.

Usage: python src/harmonise/wyscout_to_spadl.py [competition_id]
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from corpus import ROOT, in_corpus, load_config  # noqa: E402

import socceraction.spadl as spadl  # noqa: E402
from socceraction.data.wyscout import PublicWyscoutLoader  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

CFG = load_config()
RAW = ROOT / CFG["paths"]["wyscout"]
OUT = ROOT / "data" / "spadl" / "wyscout"

# dtype parity with spadl/statsbomb parquets, so a joint read never conflicts
SCHEMA_CASTS = {"original_event_id": str, "player_id": "float64"}


def flat_view(raw: Path) -> str:
    """Symlink the nested Pappalardo layout into one flat temp directory."""
    flat = Path(tempfile.mkdtemp(prefix="wyscout_flat_"))
    for f in list(raw.glob("*.json")) + list(raw.glob("events/*.json")) \
            + list(raw.glob("matches/*.json")):
        (flat / f.name).symlink_to(f.resolve())
    return str(flat)


def orientation_check(df: pd.DataFrame, label: str) -> None:
    """§4.2: the assertion written before it is needed, run on every provider."""
    shots = df[df.type_id.isin([11, 12, 13])]          # shot, shot_penalty, shot_freekick
    assert shots.start_x.mean() > 70, \
        f"{label}: shots do not cluster near the attacking goal — orientation broken"
    passes = df[df.type_id == 0]
    inb = (passes.end_x.between(0, 105) & passes.end_y.between(0, 68)).mean()
    assert inb > 0.99, f"{label}: {1 - inb:.1%} of pass ends out of bounds"
    print(f"  orientation ok: mean shot x {shots.start_x.mean():.1f}, "
          f"pass ends in bounds {inb:.1%}")


def convert_comp_season(loader, c):
    cid, sid = c["competition_id"], c["season_id"]
    out_file = OUT / f"{cid}_{sid}.parquet"
    if out_file.exists():
        return 0

    games = loader.games(cid, sid)
    parts = []
    for _, g in tqdm(games.iterrows(), total=len(games), leave=False,
                     desc=f"{c['competition_name']} {c['season_name']}"):
        events = loader.events(g.game_id)
        actions = spadl.wyscout.convert_to_actions(events, g.home_team_id)
        actions = spadl.play_left_to_right(actions, g.home_team_id)
        parts.append(actions)

    df = pd.concat(parts, ignore_index=True)
    df["competition_id"] = cid
    df["season_id"] = sid
    df["provider"] = "wyscout"
    df = df.astype(SCHEMA_CASTS)

    if df.duplicated(["game_id", "action_id"]).any():
        raise ValueError(f"{out_file.name}: (game_id, action_id) not unique")
    orientation_check(df, out_file.name)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_file,
                   compression="zstd")
    return len(df)


if __name__ == "__main__":
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    wanted = CFG["corpus"]["wyscout"]
    loader = PublicWyscoutLoader(root=flat_view(RAW), download=False)

    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for c in loader.competitions().to_dict("records"):
        if only is not None and c["competition_id"] != only:
            continue
        if not in_corpus(wanted, c["competition_id"], c["season_name"]):
            continue
        n = convert_comp_season(loader, c)
        if n:
            print(f"{c['competition_name']} {c['season_name']}: {n:,} actions")
        total += n

    staged = sum(f.stat().st_size for f in OUT.glob("*.parquet"))
    print(f"\n{total:,} new action rows | wyscout spadl layer {staged / 1e6:.0f} MB")

    sys.stdout.flush()
    os._exit(0)          # same pyarrow shutdown-deadlock workaround as to_spadl.py
