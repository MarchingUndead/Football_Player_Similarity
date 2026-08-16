"""M3-T6: the quality axis (PLAN §5.2), event-derived per D13.

Style says who plays alike; quality says who is better. The two are kept in
SEPARATE tables so no similarity model can accidentally train on quality —
consumers report "closest in style, then ranked by quality".

v1 scalar: non-penalty xG + xA per 90, from staging (StatsBomb's own shot xG).
xA = the xG of the next shot in the same possession after a pass flagged
shot_assist/goal_assist — the possession + event-index columns make this exact,
no name matching, no heuristic window. Penalties are excluded from xG (winning
a penalty is not shooting skill); shootouts are excluded by period <= 4.

Per-action VAEP (socceraction) is the planned v2 — required before M5, whose
gradient-reversal head predicts per-action quality. This table's schema is
already what downstream consumes; VAEP adds a column, not a redesign.

Output: data/features/player_season_quality.parquet
"""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from corpus import ROOT, load_config  # noqa: E402

CFG = load_config()
DB = ROOT / "data" / "football.duckdb"
OUT = ROOT / "data" / "features" / "player_season_quality.parquet"

SQL = """
WITH ev AS (
    SELECT e.*, c.season
    FROM events e
    JOIN competitions c USING (competition_id, season_id)
    WHERE e.period <= 4
),

shots AS (
    SELECT match_id, possession, "index", player_id, team_id, season,
           shot_xg, shot_type
    FROM ev WHERE type = 'Shot'
),
xg AS (                      -- non-penalty xG by the shooter
    SELECT player_id, team_id, season,
           sum(shot_xg) FILTER (shot_type <> 'Penalty') AS npxg
    FROM shots GROUP BY ALL
),

assist_passes AS (           -- passes StatsBomb flags as feeding a shot
    SELECT match_id, possession, "index", player_id, team_id, season
    FROM ev
    WHERE type = 'Pass' AND (pass_shot_assist OR pass_goal_assist)
),
pass_shot AS (               -- link each flagged pass to the NEXT shot in its possession
    SELECT p.player_id, p.team_id, p.season, s.shot_xg,
           row_number() OVER (PARTITION BY p.match_id, p.possession, p."index"
                              ORDER BY s."index") AS rn
    FROM assist_passes p
    JOIN shots s ON s.match_id = p.match_id AND s.possession = p.possession
                AND s."index" > p."index"
),
xa AS (                      -- xA = xG of the shot the pass created
    SELECT player_id, team_id, season, sum(shot_xg) AS xa
    FROM pass_shot WHERE rn = 1 GROUP BY ALL
),

mins AS (                    -- entity spine: everyone with recorded minutes
    SELECT m.player_id, m.team_id, c.season,
           sum(m.minutes) AS minutes, any_value(m.player) AS player
    FROM player_match_minutes m
    JOIN matches ma USING (match_id)
    JOIN competitions c USING (competition_id, season_id)
    GROUP BY ALL
)

SELECT m.player_id, m.team_id, m.season, m.player, m.minutes,
       coalesce(xg.npxg, 0)                        AS nonpenalty_xg_total,
       coalesce(xa.xa, 0)                          AS xa_total,
       90 * coalesce(xg.npxg, 0) / m.minutes       AS nonpenalty_xg_per_90,
       90 * coalesce(xa.xa, 0) / m.minutes         AS xa_per_90,
       90 * (coalesce(xg.npxg, 0) + coalesce(xa.xa, 0)) / m.minutes
                                                   AS nonpenalty_xg_plus_xa_per_90
FROM mins m
LEFT JOIN xg USING (player_id, team_id, season)
LEFT JOIN xa USING (player_id, team_id, season)
WHERE m.minutes > 0
"""

if __name__ == "__main__":
    con = duckdb.connect(str(DB), read_only=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con.sql(f"COPY ({SQL}) TO '{OUT}' (FORMAT parquet, COMPRESSION zstd)")

    df = con.sql(f"SELECT * FROM '{OUT}'").df()
    print(f"{len(df):,} entities -> {OUT.relative_to(ROOT)}")
    top = (df[df.minutes >= 900]
           .nlargest(5, "nonpenalty_xg_plus_xa_per_90")
           [["player", "season", "minutes", "nonpenalty_xg_plus_xa_per_90"]])
    print("top npxG+xA/90 (900+ min):")
    print(top.to_string(index=False))
