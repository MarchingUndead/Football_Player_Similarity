"""Query layer (M1-T9): a DuckDB database of views over the staging Parquet.

Creates data/football.duckdb — views only, no copied data, rebuild any time:
  events / lineups / matches   straight over the Parquet globs
  competitions                 from competitions.json incl. the club/country flag
  player_match_minutes         from lineup position stints (HH:MM clock)
  player_season_context        the (player, season, context) entity roster (D6/D11)

Ends with the M1-T9 acceptance benchmark: top 20 by minutes in Euro 2024, < 1 s.
"""

import json
import time

import duckdb

from corpus import ROOT, load_config

CFG = load_config()
STAGING = ROOT / CFG["paths"]["staging"] / "statsbomb"
DB = ROOT / "data" / "football.duckdb"

if __name__ == "__main__":
    DB.unlink(missing_ok=True)
    con = duckdb.connect(str(DB))

    for t in ("events", "lineups", "matches"):
        con.sql(f"CREATE VIEW {t} AS SELECT * FROM '{STAGING}/{t}/*.parquet'")

    comps = json.load(open(ROOT / CFG["paths"]["statsbomb"] / "competitions.json"))
    con.sql("""
        CREATE TABLE competitions (
            competition_id INT, season_id INT, competition TEXT, season TEXT,
            gender TEXT, is_country BOOL)""")
    for c in comps:
        con.execute("INSERT INTO competitions VALUES (?, ?, ?, ?, ?, ?)",
                    [c["competition_id"], c["season_id"], c["competition_name"],
                     c["season_name"], c["competition_gender"],
                     bool(c.get("competition_international"))])

    # lineup stint times are a cumulative MM:SS match clock (first part reaches 124
    # in extra time, second part <= 59); NULL to_time = played to the end (match
    # length from the last non-shootout event). Exact to within a minute or two —
    # fine for rosters and filters.
    con.sql("""
        CREATE VIEW player_match_minutes AS
        WITH match_end AS (
            SELECT match_id, max(minute) + 1 AS end_min
            FROM events WHERE period <= 4 GROUP BY match_id),
        stints AS (
            SELECT l.match_id, l.player_id, l.player, l.team_id, l.team,
                   cast(split_part(l.from_time, ':', 1) AS INT)
                     + cast(split_part(l.from_time, ':', 2) AS INT) / 60.0 AS from_min,
                   coalesce(cast(split_part(l.to_time, ':', 1) AS INT)
                     + cast(split_part(l.to_time, ':', 2) AS INT) / 60.0, m.end_min) AS to_min
            FROM lineups l JOIN match_end m USING (match_id)
            WHERE l.from_time IS NOT NULL)
        SELECT match_id, player_id, player, team_id, team,
               sum(greatest(to_min - from_min, 0)) AS minutes
        FROM stints GROUP BY ALL""")

    # the project's unit of analysis: one row = one comparable entity (D6 + D11)
    con.sql("""
        CREATE VIEW player_season_context AS
        SELECT e.player_id, any_value(e.player) AS player,
               c.season,
               CASE WHEN c.is_country THEN 'country' ELSE 'club' END AS context,
               string_agg(DISTINCT c.competition, ' + ') AS competitions,
               count(DISTINCT e.match_id) AS matches,
               count(*) AS actions
        FROM events e
        JOIN competitions c USING (competition_id, season_id)
        WHERE e.player_id IS NOT NULL AND e.period <= 4
        GROUP BY e.player_id, c.season, context""")

    # M1-T9 acceptance benchmark
    t0 = time.perf_counter()
    top = con.sql("""
        SELECT m.player, sum(m.minutes) AS minutes
        FROM player_match_minutes m
        JOIN matches ma USING (match_id)
        JOIN competitions c USING (competition_id, season_id)
        WHERE c.competition = 'UEFA Euro' AND c.season = '2024'
        GROUP BY m.player ORDER BY minutes DESC LIMIT 20""").df()
    dt = time.perf_counter() - t0

    n_entities = con.sql("SELECT count(*) FROM player_season_context").fetchone()[0]
    print(top.to_string(index=False))
    print(f"\nbenchmark: top-20-by-minutes Euro 2024 in {dt * 1000:.0f} ms "
          f"({'PASS' if dt < 1 else 'FAIL'} vs 1 s gate)")
    print(f"entity roster: {n_entities:,} (player, season, context) rows")
    print(f"wrote {DB.relative_to(ROOT)}")
