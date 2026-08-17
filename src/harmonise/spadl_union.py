"""Cross-provider SPADL access: one action stream, collision-free ids.

StatsBomb and Wyscout both use small-integer ids for players, teams, games,
competitions and seasons — the namespaces overlap, so a naive union could
silently merge two different humans into one entity. Every Wyscout id is
therefore offset by +10,000,000 (all native ids on both sides are < 10M) at
READ time, uniformly, in every consumer. Reversible: id >= 10M is Wyscout;
subtract the offset to recover the wyId. The spadl parquets keep native ids
for lineage.

The wyscout_support/ dim tables (competitions, minutes, positions — written by
wyscout_dims.py) already carry offset ids, so they join the offset action
stream directly.
"""

WYSCOUT_ID_OFFSET = 10_000_000

_ID_COLS = ["player_id", "team_id", "game_id", "competition_id", "season_id"]


def actions_union_sql(root) -> str:
    """SELECT over both providers' spadl parquets with Wyscout ids offset."""
    replace = ",\n           ".join(
        f"CASE provider WHEN 'wyscout' THEN {c} + {WYSCOUT_ID_OFFSET} "
        f"ELSE {c} END AS {c}" for c in _ID_COLS)
    return f"""
    SELECT * REPLACE (
           {replace})
    FROM read_parquet(['{root}/data/spadl/statsbomb/[0-9]*.parquet',
                       '{root}/data/spadl/wyscout/[0-9]*.parquet'])
    """


def comps_union_sql(root) -> str:
    """competitions dim over both providers: StatsBomb's DuckDB view UNION the
    offset Wyscout support table. Requires a connection to football.duckdb."""
    return f"""
    SELECT competition_id, season_id, season, is_country FROM competitions
    UNION ALL
    SELECT competition_id, season_id, season, is_country
    FROM '{root}/data/spadl/wyscout_support/competitions.parquet'
    """
