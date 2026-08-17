"""Wyscout support dims: what the feature stage needs beyond the action stream.

The StatsBomb side gets competitions/minutes/positions from the DuckDB views
(built off staging). Wyscout skipped staging (raw -> spadl directly), so this
stage supplies the same three facts from the loader + players.json:

  competitions.parquet  (competition_id, season_id, season, is_country, name)
  minutes.parquet       one row per (player, team, season): minutes + names
  positions.parquet     player_id -> position / pos_group (players.json role;
                        GK/DF/MF/FW comes straight from Wyscout's role.code2)

All ids are written WITH the +10M provider offset (spadl_union.py), so these
tables join the offset action stream directly. WC2018 stays excluded via
config.yaml, same as the spadl stage.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from corpus import ROOT, in_corpus, load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spadl_union import WYSCOUT_ID_OFFSET as OFF  # noqa: E402
from wyscout_to_spadl import flat_view  # noqa: E402

from socceraction.data.wyscout import PublicWyscoutLoader  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

CFG = load_config()
RAW = ROOT / CFG["paths"]["wyscout"]
OUT = ROOT / "data" / "spadl" / "wyscout_support"

if __name__ == "__main__":
    loader = PublicWyscoutLoader(root=flat_view(RAW), download=False)
    wanted = CFG["corpus"]["wyscout"]
    comps = [c for c in loader.competitions().to_dict("records")
             if in_corpus(wanted, c["competition_id"], c["season_name"])]

    OUT.mkdir(parents=True, exist_ok=True)

    cdf = pd.DataFrame(comps)
    cdf = pd.DataFrame({
        "competition_id": cdf.competition_id + OFF,
        "season_id": cdf.season_id + OFF,
        "season": cdf.season_name,
        "is_country": cdf.country_name.eq("International"),
        "competition_name": cdf.competition_name,
    })
    cdf.to_parquet(OUT / "competitions.parquet", index=False)

    pj = json.load(open(RAW / "players.json"))
    pos = pd.DataFrame({
        "player_id": [p["wyId"] + OFF for p in pj],
        "position": [p["role"]["name"] for p in pj],
        # Wyscout's code2 says MD where StatsBomb-derived groups say MF —
        # one vocabulary, or position filters silently split by provider
        "pos_group": [p["role"]["code2"].replace("MD", "MF") for p in pj],
    })
    pos.to_parquet(OUT / "positions.parquet", index=False)

    rows = []
    for c in comps:
        games = loader.games(c["competition_id"], c["season_id"])
        for gid in tqdm(games.game_id, leave=False,
                        desc=f"{c['competition_name']} {c['season_name']}"):
            p = loader.players(gid)
            t = loader.teams(gid)[["team_id", "team_name"]]
            p = p.merge(t, on="team_id")
            rows.append(pd.DataFrame({
                "player_id": p.player_id + OFF, "team_id": p.team_id + OFF,
                "season": c["season_name"], "minutes": p.minutes_played,
                "player": p.player_name, "team": p.team_name,
            }))
    minutes = (pd.concat(rows, ignore_index=True)
               .groupby(["player_id", "team_id", "season"], as_index=False)
               .agg(minutes=("minutes", "sum"), player=("player", "first"),
                    team=("team", "first")))
    minutes.to_parquet(OUT / "minutes.parquet", index=False)

    print(f"{len(cdf)} comps | {len(pos):,} player roles | "
          f"{len(minutes):,} (player, team, season) minute rows -> "
          f"{OUT.relative_to(ROOT)}")

    sys.stdout.flush()
    os._exit(0)
