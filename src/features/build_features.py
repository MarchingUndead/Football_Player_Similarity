"""Feature stage (M3): SPADL actions -> one row per (player_id, team_id, season).

The entity is one player at one team in one season (D14): national teams are
distinct team_ids, so club vs country separates by construction; a mid-season
transfer splits into two entities; league + CL at the same club pool into one.
`context` (club/country) survives as a derived attribute.

Every style feature is a SHARE or RATE, never a raw count — composition over the
simplex separates what a player chooses to do from how much they play (PLAN §5.2).
Volume lives only in the support columns, used downstream for explicit filtering
(no shrinkage — D15).

Two adjustments ride along (PLAN §5.3, minus shrinkage per D15):

1. Possession-opportunity rates, in the SQL. Composition shares hide the
   environment: a midfielder in a team that never has the ball tackles a lot
   because there is lots of tackling to do. So defensive activity is measured
   per 100 OPPONENT actions and offensive involvement per 100 TEAM actions,
   both counted only in the matches the player appeared in for that team.

2. Team fixed-effect residualization, in pandas. For every style feature,
   subtract the action-weighted (team_id, season) mean; what remains is how the
   player deviates from their own team's way of playing. The per-feature share
   of variance the team explains is written to reports/team_deconfounding.md —
   the "which features are system, which are player" finding.

Outputs:
  data/features/player_season_context.parquet        (raw shares/rates)
  data/features/player_season_team_adjusted.parquet  (same columns, residualized)
  reports/team_deconfounding.md
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from corpus import ROOT, load_config  # noqa: E402

CFG = load_config()
SPADL = ROOT / "data" / "spadl" / "statsbomb"
DB = ROOT / "data" / "football.duckdb"
OUT_RAW = ROOT / "data" / "features" / "player_season_context.parquet"
OUT_ADJ = ROOT / "data" / "features" / "player_season_team_adjusted.parquet"
PASS_RISK = ROOT / "data" / "features" / "pass_difficulty_entity.parquet"

NOT_FEATURES = {"player_id", "team_id", "season", "context", "matches", "actions",
                "minutes", "player", "team", "position", "pos_group",
                "team_group_size"}
MIN_ACTIONS_FOR_R2 = 300   # R2 measured on solid entities so noise doesn't drown signal

# zone grid: 6 columns own-goal -> attacking, 5 rows across the pitch (PLAN §5.1)
ZX, ZY = 105 / 6, 68 / 5

SQL = f"""
WITH actions AS (            -- one clean pass over spadl: names on, shootout off
    SELECT a.*, t.type_name, r.result_name, b.bodypart_name,
           c.season, CASE WHEN c.is_country THEN 'country' ELSE 'club' END AS context
    FROM '{SPADL}/[0-9]*.parquet' a
    JOIN '{SPADL}/actiontypes.parquet' t USING (type_id)
    JOIN '{SPADL}/results.parquet' r USING (result_id)
    JOIN '{SPADL}/bodyparts.parquet' b USING (bodypart_id)
    JOIN competitions c USING (competition_id, season_id)
    WHERE a.period_id <= 4 AND t.type_name <> 'non_action'
),

spine AS (                   -- the entity and its support columns
    SELECT player_id, team_id, season, any_value(context) AS context,
           count(DISTINCT game_id) AS matches, count(*) AS actions
    FROM actions GROUP BY player_id, team_id, season
),

minutes AS (                 -- support: playing time, from the lineup-stint view
    SELECT m.player_id, m.team_id, c.season,
           sum(m.minutes) AS minutes,
           any_value(m.player) AS player, any_value(m.team) AS team
    FROM player_match_minutes m
    JOIN matches ma USING (match_id)
    JOIN competitions c USING (competition_id, season_id)
    GROUP BY ALL
),

pos_counts AS (              -- how often each entity was fielded in each position
    SELECT l.player_id, l.team_id, c.season, l.position,
           CASE WHEN l.position ILIKE '%goalkeeper%' THEN 'GK'
                WHEN l.position ILIKE '%back%' THEN 'DF'
                WHEN l.position ILIKE '%midfield%' THEN 'MF'
                ELSE 'FW' END AS pos_group,
           count(*) AS n
    FROM lineups l
    JOIN competitions c USING (competition_id, season_id)
    WHERE l.position IS NOT NULL
    GROUP BY ALL
),
position AS (                -- metadata for probes/filters, not a style feature
    SELECT player_id, team_id, season, position, pos_group
    FROM (SELECT *, row_number() OVER (PARTITION BY player_id, team_id, season
                                       ORDER BY n DESC) AS rn
          FROM pos_counts)
    WHERE rn = 1
),

-- 1. action mix: of everything this player does on the ball, what fraction is X?
mix AS (
    SELECT player_id, team_id, season,
           count(*) FILTER (type_name = 'pass')          / count(*)::DOUBLE AS pass_share_of_actions,
           count(*) FILTER (type_name = 'dribble')       / count(*)::DOUBLE AS carry_share_of_actions,
           count(*) FILTER (type_name = 'take_on')       / count(*)::DOUBLE AS take_on_share_of_actions,
           count(*) FILTER (type_name = 'cross')         / count(*)::DOUBLE AS cross_share_of_actions,
           count(*) FILTER (type_name IN ('shot', 'shot_penalty', 'shot_freekick'))
                                                         / count(*)::DOUBLE AS shot_share_of_actions,
           count(*) FILTER (type_name = 'tackle')        / count(*)::DOUBLE AS tackle_share_of_actions,
           count(*) FILTER (type_name = 'interception')  / count(*)::DOUBLE AS interception_share_of_actions,
           count(*) FILTER (type_name = 'clearance')     / count(*)::DOUBLE AS clearance_share_of_actions,
           count(*) FILTER (type_name = 'foul')          / count(*)::DOUBLE AS foul_share_of_actions,
           count(*) FILTER (type_name IN ('freekick_short', 'freekick_crossed',
                                          'corner_short', 'corner_crossed'))
                                                         / count(*)::DOUBLE AS setpiece_share_of_actions,
           count(*) FILTER (type_name = 'throw_in')      / count(*)::DOUBLE AS throw_in_share_of_actions,
           count(*) FILTER (type_name = 'bad_touch')     / count(*)::DOUBLE AS bad_touch_share_of_actions,
           count(*) FILTER (type_name LIKE 'keeper%')    / count(*)::DOUBLE AS keeper_action_share_of_actions
    FROM actions GROUP BY ALL
),

-- 2. spatial signature: share of actions started in each 6x5 zone.
--    x0 = own-goal end .. x5 = attacking end; y0 .. y4 across the pitch width.
zone_counts AS (
    SELECT player_id, team_id, season,
           'zone_x' || least(floor(start_x / {ZX}), 5)::INT
               || '_y' || least(floor(start_y / {ZY}), 4)::INT
               || '_share_of_actions' AS zone,
           count(*) AS n
    FROM actions GROUP BY ALL
),
zones AS (
    PIVOT (
        SELECT player_id, team_id, season, zone,
               n / sum(n) OVER (PARTITION BY player_id, team_id, season) AS share
        FROM zone_counts)
    ON zone USING sum(share)
),

-- 3. pass geometry: on the player's passes, where do they go and how?
pass AS (
    SELECT player_id, team_id, season,
           avg(sqrt((end_x - start_x) ^ 2 + (end_y - start_y) ^ 2)) AS pass_length_mean_m,
           avg((end_x - start_x >= abs(end_y - start_y))::INT)      AS forward_share_of_passes,
           avg((start_x - end_x >= abs(end_y - start_y))::INT)      AS backward_share_of_passes,
           avg((sqrt((end_x-start_x)^2 + (end_y-start_y)^2) >= 30)::INT) AS long_share_of_passes,
           avg((end_x - start_x >= 10)::INT)                        AS progressive_share_of_passes,
           avg((abs(end_y - start_y) >= 25)::INT)                   AS switch_share_of_passes,
           avg((result_name = 'success')::INT)                      AS pass_success_rate
    FROM actions WHERE type_name = 'pass' GROUP BY ALL
),

-- 4. carry profile: when they move with the ball, how far, how direct, into danger?
carry AS (
    SELECT player_id, team_id, season,
           avg(sqrt((end_x - start_x) ^ 2 + (end_y - start_y) ^ 2)) AS carry_length_mean_m,
           avg((end_x - start_x >= 10)::INT)                        AS progressive_share_of_carries,
           avg((start_x < 70 AND end_x >= 70)::INT)                 AS final_third_entry_share_of_carries,
           avg((end_x >= 88.5 AND end_y BETWEEN 13.84 AND 54.16)::INT) AS box_entry_share_of_carries
    FROM actions WHERE type_name = 'dribble' GROUP BY ALL
),

-- 5. duel/shot/footedness profiles
takeon AS (
    SELECT player_id, team_id, season,
           avg((result_name = 'success')::INT) AS take_on_success_rate
    FROM actions WHERE type_name = 'take_on' GROUP BY ALL
),
shot AS (                    -- open-play shots only; placement z lives in staging (M6)
    SELECT player_id, team_id, season,
           avg(sqrt((105 - start_x) ^ 2 + (34 - start_y) ^ 2)) AS shot_distance_to_goal_mean_m,
           avg((start_x < 88.5)::INT)                          AS outside_box_share_of_shots,
           avg((bodypart_name = 'head')::INT)                  AS header_share_of_shots
    FROM actions WHERE type_name = 'shot' GROUP BY ALL
),
foot AS (                    -- footedness: among foot actions with a known side
    SELECT player_id, team_id, season,
           avg((bodypart_name = 'foot_left')::INT) AS left_foot_share_of_foot_actions
    FROM actions WHERE bodypart_name IN ('foot_left', 'foot_right') GROUP BY ALL
),

-- 6. defensive height + overall directness
defense AS (
    SELECT player_id, team_id, season,
           avg(start_x)                AS defensive_action_height_mean_m,
           avg((start_x >= 52.5)::INT) AS opponent_half_share_of_defensive_actions
    FROM actions WHERE type_name IN ('tackle', 'interception') GROUP BY ALL
),
tempo AS (                   -- net field progress per on-ball involvement
    SELECT player_id, team_id, season,
           avg(end_x - start_x) AS upfield_progress_per_pass_or_carry_mean_m
    FROM actions WHERE type_name IN ('pass', 'dribble') GROUP BY ALL
),

-- 7. possession-opportunity rates: activity per 100 team/opponent actions,
--    counted only over the games the player appeared in for that team.
game_team AS (               -- how much each team played the ball in each game
    SELECT game_id, team_id, count(*) AS n FROM actions GROUP BY ALL
),
game_opp AS (                -- and how much their opponents did
    SELECT gt.game_id, gt.team_id, sum(o.n) AS opp_n
    FROM game_team gt
    JOIN game_team o ON o.game_id = gt.game_id AND o.team_id <> gt.team_id
    GROUP BY gt.game_id, gt.team_id
),
entity_games AS (
    SELECT DISTINCT player_id, team_id, season, game_id FROM actions
),
env AS (                     -- the opportunity environment of each entity
    SELECT e.player_id, e.team_id, e.season,
           sum(gt.n) AS team_actions, sum(go.opp_n) AS opp_actions
    FROM entity_games e
    JOIN game_team gt USING (game_id, team_id)
    JOIN game_opp go USING (game_id, team_id)
    GROUP BY ALL
),
type_counts AS (
    SELECT player_id, team_id, season,
           count(*) FILTER (type_name IN ('pass', 'cross', 'dribble', 'take_on',
                                          'shot', 'shot_penalty', 'shot_freekick'))
               AS off_n,
           count(*) FILTER (type_name IN ('tackle', 'interception', 'clearance'))
               AS def_n
    FROM actions GROUP BY ALL
),
rates AS (
    SELECT c.player_id, c.team_id, c.season,
           100.0 * c.off_n / e.team_actions AS offensive_actions_per_100_team_actions,
           100.0 * c.def_n / e.opp_actions  AS defensive_actions_per_100_opponent_actions
    FROM type_counts c JOIN env e USING (player_id, team_id, season)
),

-- 8. risk profile, precomputed by src/features/pass_difficulty.py (M3-T2):
--    how hard are the passes they attempt, and do they beat the model's expectation?
risk AS (
    SELECT * FROM '{PASS_RISK}'
)

SELECT s.*, m2.minutes, m2.player, m2.team, p.position, p.pos_group,
       mix.* EXCLUDE (player_id, team_id, season),
       zones.* EXCLUDE (player_id, team_id, season),
       pass.* EXCLUDE (player_id, team_id, season),
       carry.* EXCLUDE (player_id, team_id, season),
       takeon.* EXCLUDE (player_id, team_id, season),
       shot.* EXCLUDE (player_id, team_id, season),
       foot.* EXCLUDE (player_id, team_id, season),
       defense.* EXCLUDE (player_id, team_id, season),
       tempo.* EXCLUDE (player_id, team_id, season),
       rates.* EXCLUDE (player_id, team_id, season),
       risk.* EXCLUDE (player_id, team_id, season)
FROM spine s
LEFT JOIN minutes m2 USING (player_id, team_id, season)
LEFT JOIN position p USING (player_id, team_id, season)
LEFT JOIN mix USING (player_id, team_id, season)
LEFT JOIN zones USING (player_id, team_id, season)
LEFT JOIN pass USING (player_id, team_id, season)
LEFT JOIN carry USING (player_id, team_id, season)
LEFT JOIN takeon USING (player_id, team_id, season)
LEFT JOIN shot USING (player_id, team_id, season)
LEFT JOIN foot USING (player_id, team_id, season)
LEFT JOIN defense USING (player_id, team_id, season)
LEFT JOIN tempo USING (player_id, team_id, season)
LEFT JOIN rates USING (player_id, team_id, season)
LEFT JOIN risk USING (player_id, team_id, season)
"""


def residualize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subtract action-weighted (team_id, season) means from every style feature.

    Returns the residualized copy and the per-feature team-R2 table.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns
                    if c not in NOT_FEATURES and df[c].dtype.kind == "f"]

    grp = df.groupby(["team_id", "season"])
    df["team_group_size"] = grp["player_id"].transform("size")
    r2_rows = []
    solid = df.actions >= MIN_ACTIONS_FOR_R2
    for col in feature_cols:
        w = df.actions.where(df[col].notna())
        wmean = (df[col] * w).groupby([df.team_id, df.season]).transform("sum") \
            / w.groupby([df.team_id, df.season]).transform("sum")
        resid = df[col] - wmean
        var_raw = df.loc[solid, col].var()
        var_res = resid[solid].var()
        r2_rows.append({"feature": col,
                        "team_r2": 1 - var_res / var_raw if var_raw else float("nan")})
        df[col] = resid

    r2 = pd.DataFrame(r2_rows).sort_values("team_r2", ascending=False)
    return df, r2


if __name__ == "__main__":
    con = duckdb.connect(str(DB), read_only=True)
    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    con.sql(f"COPY ({SQL}) TO '{OUT_RAW}' (FORMAT parquet, COMPRESSION zstd)")

    df = con.sql(f"SELECT * FROM '{OUT_RAW}'").df()
    n_feat = sum(1 for c in df.columns
                 if c not in NOT_FEATURES and df[c].dtype.kind == "f")
    print(f"{len(df):,} entities x {n_feat} features -> {OUT_RAW.relative_to(ROOT)}")

    dup = df.duplicated(["player_id", "team_id", "season"]).sum()
    print("duplicate keys:", dup)

    messi = df[(df.player.str.contains('Messi Cuccittini', na=False))
               & (df.season == '2014/2015')]
    if len(messi):
        r = messi.iloc[0]
        print(f"Messi 14/15 {r.team} ({r.context}): carry {r.carry_share_of_actions:.0%}, "
              f"take-on succ {r.take_on_success_rate:.0%}, "
              f"left foot {r.left_foot_share_of_foot_actions:.0%}")

    adj, r2 = residualize(df)
    adj.to_parquet(OUT_ADJ, compression="zstd", index=False)

    report = ROOT / CFG["paths"]["reports"] / "team_deconfounding.md"
    lines = [
        "# Team deconfounding (M3)",
        "",
        "Regenerated by `src/features/build_features.py` on every `dvc repro`. Do not edit.",
        "",
        "Every style feature is residualized against its action-weighted (team, season)",
        f"mean. `team_r2` = share of between-entity variance (entities with ≥"
        f"{MIN_ACTIONS_FOR_R2} actions) explained by which team you play for — high means",
        "system-driven, low means player-intrinsic. The residual table is what similarity",
        "models consume; this ranking is a finding in its own right.",
        "",
        "| feature | team_r2 |",
        "|---|--:|",
    ] + [f"| {r.feature} | {r.team_r2:.3f} |" for r in r2.itertuples()]
    report.write_text("\n".join(lines) + "\n")

    print(f"{len(adj):,} entities residualized -> {OUT_ADJ.relative_to(ROOT)}")
    print("most team-driven:  ",
          ", ".join(f"{r.feature} {r.team_r2:.2f}" for r in r2.head(5).itertuples()))
    print("most player-driven:",
          ", ".join(f"{r.feature} {r.team_r2:.2f}" for r in r2.tail(5).itertuples()))
