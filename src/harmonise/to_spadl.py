"""SPADL layer (M2-T1): staging-adjacent semantic normalisation via socceraction.

Converts every corpus match to SPADL (Decroos et al., KDD 2019) — one row per
on-ball action, 23-type vocabulary, 105x68 m pitch, every action re-oriented so
the acting team attacks left -> right. One Parquet per competition-season under
data/spadl/statsbomb/, plus the three vocabulary tables (actiontypes, results,
bodyparts) written once.

This stage reads the pinned RAW clone through socceraction's maintained
StatsBombLoader rather than our staging Parquet: reconstructing the loader's
exact input format from staging would re-implement maintained conversion code
and own its edge cases. Raw is pinned by manifest.yaml, so provenance holds.

Primary key of the output: (game_id, action_id). original_event_id traces every
action back to the raw/staging event uuid.

Usage: python src/harmonise/to_spadl.py [competition_id]
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from corpus import ROOT, in_corpus, load_config  # noqa: E402

import socceraction.spadl as spadl  # noqa: E402
from socceraction.data.statsbomb import StatsBombLoader  # noqa: E402

# pandas-3.0 deprecation chatter from socceraction 1.5.3 internals, and the
# per-match "inferred xy_fidelity_version" notice — informational, not ours to fix
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inferred")   # xy/shot fidelity notices

CFG = load_config()
OUT = ROOT / "data" / "spadl" / "statsbomb"


def convert_comp_season(loader, c):
    cid, sid = c["competition_id"], c["season_id"]
    out_file = OUT / f"{cid}_{sid}.parquet"
    if out_file.exists():
        return 0

    games = loader.games(cid, sid)
    parts = []
    for _, g in tqdm(games.iterrows(), total=len(games), leave=False,
                     desc=f"{c['competition_name']} {c['season_name']}"):
        try:
            events = loader.events(g.game_id)
        except FileNotFoundError:
            print(f"skip {g.game_id}: listed in matches but no events file (see coverage)")
            continue
        actions = spadl.statsbomb.convert_to_actions(events, g.home_team_id)
        actions = spadl.play_left_to_right(actions, g.home_team_id)
        parts.append(actions)

    df = pd.concat(parts, ignore_index=True)
    df["competition_id"] = cid
    df["season_id"] = sid
    df["provider"] = "statsbomb"   # constant today; schema survives Wyscout's return (§2.7)

    if df.duplicated(["game_id", "action_id"]).any():
        raise ValueError(f"{out_file.name}: (game_id, action_id) not unique")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_file,
                   compression="zstd")
    return len(df)


if __name__ == "__main__":
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    wanted = CFG["corpus"]["statsbomb"]
    loader = StatsBombLoader(getter="local", root=str(ROOT / CFG["paths"]["statsbomb"]))

    OUT.mkdir(parents=True, exist_ok=True)
    # the vocabulary (dimension) tables the fact table's ids point into
    for name, dim in (("actiontypes", spadl.actiontypes_df()),
                      ("results", spadl.results_df()),
                      ("bodyparts", spadl.bodyparts_df())):
        pq.write_table(pa.Table.from_pandas(dim, preserve_index=False),
                       OUT / f"{name}.parquet", compression="zstd")

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
    print(f"\n{total:,} new action rows | spadl layer now {staged / 1e6:.0f} MB")

    # pyarrow/jemalloc threads can deadlock interpreter shutdown (observed: 6h futex
    # wait after all work finished, which stalls `dvc repro`). Work is flushed to
    # disk at this point — skip teardown entirely.
    sys.stdout.flush()
    import os
    os._exit(0)
