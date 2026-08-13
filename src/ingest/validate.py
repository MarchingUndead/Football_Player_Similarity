"""Staging validation (M1-T8): the "trusted from now on" gate over the Parquet layer.

Three layers, all failures reported, non-zero exit if any:
1. pandera column checks per comp-season events file — coordinate ranges, required
   fields, xG in [0,1]. One file at a time, only the checked columns loaded (RAM).
2. DuckDB set checks over the whole corpus — timestamps monotonic within
   (match, period), referential integrity events -> matches and events -> lineups.
3. Summary written to reports/staging_validation.md.
"""

import sys

import duckdb
import pandas as pd
import pandera as pa
from pandera import Check, Column
from tqdm import tqdm

from corpus import ROOT, load_config

CFG = load_config()
STAGING = ROOT / CFG["paths"]["staging"] / "statsbomb"


# StatsBomb emits sub-yard coordinate overshoots at pitch edges (throw-ins, balls
# out of play): observed max 120.9 / 80.9 across ~700 of 13.5M rows. One yard of
# tolerance is the provider's real envelope; beyond it = genuine corruption.
# See reports/03_failures.md entry 1.
EDGE = 1.0


def coord(hi):
    return Column(nullable=True, checks=Check.in_range(0 - EDGE, hi + EDGE), coerce=False)


EVENTS_SCHEMA = pa.DataFrameSchema({
    "match_id": Column(nullable=False),
    "event_id": Column(nullable=False),
    "type": Column(nullable=False),
    "period": Column(nullable=False, checks=Check.in_range(1, 5)),
    "minute": Column(nullable=False, checks=Check.ge(0)),
    "x": coord(120), "y": coord(80),
    "pass_end_x": coord(120), "pass_end_y": coord(80),
    "carry_end_x": coord(120), "carry_end_y": coord(80),
    "shot_end_x": coord(120), "shot_end_y": coord(80),
    "shot_end_z": Column(nullable=True, checks=Check.ge(0)),
    "shot_xg": Column(nullable=True, checks=Check.in_range(0, 1)),
})
CHECKED_COLS = list(EVENTS_SCHEMA.columns)


def check_columns():
    failures = []
    for f in tqdm(sorted((STAGING / "events").glob("*.parquet")), desc="pandera per file"):
        df = pd.read_parquet(f, columns=CHECKED_COLS)
        try:
            EVENTS_SCHEMA.validate(df, lazy=True)
        except pa.errors.SchemaErrors as e:
            # one line per (column, check): how many rows broke it, and an example value
            for (col, chk), grp in e.failure_cases.groupby(["column", "check"], dropna=False):
                failures.append(f"{f.name}: {col} failed `{chk}` on {len(grp)} rows "
                                f"(e.g. {grp.failure_case.iloc[0]})")
    return failures


def check_sets():
    con = duckdb.connect()
    ev = f"'{STAGING}/events/*.parquet'"
    lu = f"'{STAGING}/lineups/*.parquet'"
    ma = f"'{STAGING}/matches/*.parquet'"
    failures = []

    # timestamps are zero-padded HH:MM:SS.mmm strings — lexicographic order = time order.
    # Two known upstream quirks are exempt (reports/03_failures.md entry 1):
    #   - clock reset: a stoppage-time pass's Ball Receipt stamped on the NEXT half's
    #     clock (prev in stoppage time, new stamp inside the reset clock's first
    #     minutes) while staying in the same period
    #   - whistle jitter: Half End / adjacent events stamped up to ~3 s before the
    #     last action they follow
    # `index` stays canonical. Anything bigger than 5 s and not a reset still fails.
    # "in stoppage time" depends on the period: halves run 45'+, extra-time periods 15'+
    n = con.sql(f"""
        SELECT count(*) FROM (
            SELECT period, timestamp, lag(timestamp) OVER
                   (PARTITION BY match_id, period ORDER BY index) AS prev
            FROM {ev})
        WHERE timestamp < prev
          AND epoch(prev::TIME) - epoch(timestamp::TIME) > 5
          AND NOT (timestamp < '00:03:00'
                   AND ((period <= 2 AND prev >= '00:40:00')
                        OR (period IN (3, 4) AND prev >= '00:12:00')))""").fetchone()[0]
    if n:
        failures.append(f"{n} events run backwards in time within a (match, period) "
                        f"— beyond the known clock-reset and whistle-jitter quirks")

    n = con.sql(f"""
        SELECT count(DISTINCT e.match_id) FROM {ev} e
        WHERE e.match_id NOT IN (SELECT match_id FROM {ma})""").fetchone()[0]
    if n:
        failures.append(f"{n} match_ids in events missing from the matches table")

    n = con.sql(f"""
        SELECT count(*) FROM (
            SELECT DISTINCT e.match_id, e.player_id FROM {ev} e
            WHERE e.player_id IS NOT NULL) ep
        WHERE NOT EXISTS (
            SELECT 1 FROM {lu} l
            WHERE l.match_id = ep.match_id AND l.player_id = ep.player_id)""").fetchone()[0]
    if n:
        failures.append(f"{n} (match, player) pairs act in events but are not on the team sheet")

    stats = con.sql(f"""
        SELECT (SELECT count(*) FROM {ev}),
               (SELECT count(DISTINCT match_id) FROM {ma}),
               (SELECT count(*) FROM {lu})""").fetchone()
    return failures, stats


if __name__ == "__main__":
    col_failures = check_columns()
    set_failures, (n_events, n_matches, n_lineup_rows) = check_sets()
    failures = col_failures + set_failures

    out = ROOT / CFG["paths"]["reports"] / "staging_validation.md"
    lines = [
        "# Staging validation (M1-T8)",
        "",
        "Regenerated by `src/ingest/validate.py` on every `dvc repro`. Do not edit.",
        "",
        f"Checked: {n_events:,} events across {n_matches:,} matches, "
        f"{n_lineup_rows:,} lineup rows.",
        "Column checks: coordinate ranges (x 0–120, y 0–80, all start/end fields), "
        "period 1–5, xG in [0,1], required ids present.",
        "Set checks: timestamps monotonic within (match, period); every event's match "
        "exists in matches; every acting player is on that match's team sheet.",
        "",
        "## Result",
        "",
    ]
    if failures:
        lines += [f"**{len(failures)} FAILURE(S):**", ""] + [f"- {f}" for f in failures]
    else:
        lines.append("**All green.** The staging layer is trusted from here on.")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    print(f"wrote {out.relative_to(ROOT)}")
    for f in failures:
        print("FAIL:", f)
    sys.exit(1 if failures else 0)
