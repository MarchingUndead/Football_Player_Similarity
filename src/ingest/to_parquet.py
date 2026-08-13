"""Staging layer (M1-T7): StatsBomb match JSON → Parquet, one file per competition-season.

Three tables per comp-season under data/staging/statsbomb/:
  events/   one row per event, nested dicts flattened to the columns the project
            uses (PLAN §5); deep blobs (shot freeze frames, tactics) kept as JSON
            strings. The raw clone stays the lossless source of record.
  lineups/  one row per player position stint — this is where minutes come from.
  matches/  one row per match.

Every row carries (match_id, competition_id, season_id) so the project's unit of
analysis, (player, competition, season), is a plain GROUP BY away.

Usage: python src/ingest/to_parquet.py [competition_id]
Converts the config.yaml corpus; the optional argument restricts to one
competition (targeted rebuild / quick test). Existing output files are skipped.
"""

import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from corpus import ROOT, in_corpus, load_config

CFG = load_config()
SB = ROOT / CFG["paths"]["statsbomb"]
OUT = ROOT / CFG["paths"]["staging"] / "statsbomb"

BATCH_ROWS = 200_000  # flush to parquet at this size; keeps RAM flat on 1.3M-event seasons

f32, i32, i64 = pa.float32(), pa.int32(), pa.int64()

EVENTS_SCHEMA = pa.schema([
    # identity + ordering
    ("match_id", i64), ("competition_id", i32), ("season_id", i32),
    ("event_id", pa.string()), ("index", i32), ("period", i32),
    ("timestamp", pa.string()), ("minute", i32), ("second", i32),
    # what, who, where
    ("type", pa.string()), ("team_id", i32), ("team", pa.string()),
    ("player_id", i64), ("player", pa.string()), ("position", pa.string()),
    ("x", f32), ("y", f32), ("duration", f32),
    # possession context
    ("possession", i32), ("possession_team_id", i32), ("play_pattern", pa.string()),
    # flags present on many event types
    ("under_pressure", pa.bool_()), ("counterpress", pa.bool_()),
    ("off_camera", pa.bool_()), ("out", pa.bool_()),
    # outcome/body_part/technique are unified across pass/shot/dribble/duel/...
    ("outcome", pa.string()), ("body_part", pa.string()), ("technique", pa.string()),
    ("related_events", pa.list_(pa.string())),
    # pass
    ("pass_end_x", f32), ("pass_end_y", f32), ("pass_length", f32), ("pass_angle", f32),
    ("pass_height", pa.string()), ("pass_recipient_id", i64), ("pass_type", pa.string()),
    ("pass_cross", pa.bool_()), ("pass_switch", pa.bool_()), ("pass_through_ball", pa.bool_()),
    ("pass_cut_back", pa.bool_()), ("pass_shot_assist", pa.bool_()), ("pass_goal_assist", pa.bool_()),
    # carry
    ("carry_end_x", f32), ("carry_end_y", f32),
    # dribble
    ("dribble_nutmeg", pa.bool_()), ("dribble_overrun", pa.bool_()), ("dribble_no_touch", pa.bool_()),
    # shot — end_z is the goal-mouth placement only StatsBomb has
    ("shot_end_x", f32), ("shot_end_y", f32), ("shot_end_z", f32),
    ("shot_xg", f32), ("shot_type", pa.string()), ("shot_first_time", pa.bool_()),
    ("shot_freeze_frame", pa.string()),
    # starting XI / tactical shift payload
    ("tactics", pa.string()),
])

LINEUPS_SCHEMA = pa.schema([
    ("match_id", i64), ("competition_id", i32), ("season_id", i32),
    ("team_id", i32), ("team", pa.string()),
    ("player_id", i64), ("player", pa.string()), ("nickname", pa.string()),
    ("jersey_number", i32), ("country", pa.string()),
    # one row per position stint; unused subs get one row with nulls
    ("position", pa.string()), ("from_time", pa.string()), ("to_time", pa.string()),
    ("from_period", i32), ("to_period", i32),
    ("start_reason", pa.string()), ("end_reason", pa.string()),
])

MATCHES_SCHEMA = pa.schema([
    ("match_id", i64), ("competition_id", i32), ("season_id", i32),
    ("competition", pa.string()), ("season", pa.string()),
    ("match_date", pa.string()), ("kick_off", pa.string()), ("match_week", i32),
    ("competition_stage", pa.string()), ("stadium", pa.string()),
    ("home_team_id", i32), ("home_team", pa.string()), ("home_score", i32),
    ("away_team_id", i32), ("away_team", pa.string()), ("away_score", i32),
    ("has_360", pa.bool_()),
])


def sub(d, *keys):
    # walk nested dicts, None as soon as anything is missing
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# event types whose JSON sub-dict key is not the lowercased type name
TYPE_KEY = {"Goal Keeper": "goalkeeper", "50/50": "50_50", "Ball Receipt*": "ball_receipt"}


def event_row(e, match_id, cid, sid):
    x, y = (e.get("location") or [None, None])[:2]
    shot_end = sub(e, "shot", "end_location") or [None, None, None]
    type_name = e["type"]["name"]
    # outcome/body_part/technique live inside the type-specific sub-dict (pass, shot, duel...)
    detail = e.get(TYPE_KEY.get(type_name, type_name.lower().replace(" ", "_")))
    detail = detail if isinstance(detail, dict) else {}

    row = {
        "match_id": match_id, "competition_id": cid, "season_id": sid,
        "event_id": e["id"], "index": e.get("index"), "period": e.get("period"),
        "timestamp": e.get("timestamp"), "minute": e.get("minute"), "second": e.get("second"),
        "type": type_name,
        "team_id": sub(e, "team", "id"), "team": sub(e, "team", "name"),
        "player_id": sub(e, "player", "id"), "player": sub(e, "player", "name"),
        "position": sub(e, "position", "name"),
        "x": x, "y": y, "duration": e.get("duration"),
        "possession": e.get("possession"),
        "possession_team_id": sub(e, "possession_team", "id"),
        "play_pattern": sub(e, "play_pattern", "name"),
        "under_pressure": bool(e.get("under_pressure")),
        "counterpress": bool(e.get("counterpress")),
        "off_camera": bool(e.get("off_camera")),
        "out": bool(e.get("out")),
        "outcome": sub(detail, "outcome", "name"),
        "body_part": sub(detail, "body_part", "name"),
        "technique": sub(detail, "technique", "name"),
        "related_events": e.get("related_events"),
        "pass_end_x": (sub(e, "pass", "end_location") or [None])[0],
        "pass_end_y": (sub(e, "pass", "end_location") or [None, None])[1],
        "pass_length": sub(e, "pass", "length"), "pass_angle": sub(e, "pass", "angle"),
        "pass_height": sub(e, "pass", "height", "name"),
        "pass_recipient_id": sub(e, "pass", "recipient", "id"),
        "pass_type": sub(e, "pass", "type", "name"),
        "pass_cross": bool(sub(e, "pass", "cross")),
        "pass_switch": bool(sub(e, "pass", "switch")),
        "pass_through_ball": bool(sub(e, "pass", "through_ball")),
        "pass_cut_back": bool(sub(e, "pass", "cut_back")),
        "pass_shot_assist": bool(sub(e, "pass", "shot_assist")),
        "pass_goal_assist": bool(sub(e, "pass", "goal_assist")),
        "carry_end_x": (sub(e, "carry", "end_location") or [None])[0],
        "carry_end_y": (sub(e, "carry", "end_location") or [None, None])[1],
        "dribble_nutmeg": bool(sub(e, "dribble", "nutmeg")),
        "dribble_overrun": bool(sub(e, "dribble", "overrun")),
        "dribble_no_touch": bool(sub(e, "dribble", "no_touch")),
        "shot_end_x": shot_end[0], "shot_end_y": shot_end[1],
        "shot_end_z": shot_end[2] if len(shot_end) > 2 else None,
        "shot_xg": sub(e, "shot", "statsbomb_xg"),
        "shot_type": sub(e, "shot", "type", "name"),
        "shot_first_time": bool(sub(e, "shot", "first_time")),
    }
    ff = sub(e, "shot", "freeze_frame")
    row["shot_freeze_frame"] = json.dumps(ff) if ff else None
    row["tactics"] = json.dumps(e["tactics"]) if "tactics" in e else None
    return row


def lineup_rows(match_id, cid, sid):
    rows = []
    lineups_file = SB / "lineups" / f"{match_id}.json"
    if not lineups_file.exists():
        return rows
    for team in json.load(open(lineups_file)):
        for p in team["lineup"]:
            base = {
                "match_id": match_id, "competition_id": cid, "season_id": sid,
                "team_id": team["team_id"], "team": team["team_name"],
                "player_id": p["player_id"], "player": p["player_name"],
                "nickname": p.get("player_nickname"),
                "jersey_number": p.get("jersey_number"),
                "country": sub(p, "country", "name"),
            }
            if not p["positions"]:
                rows.append(base)  # unused sub: on the sheet, never on the pitch
            for pos in p["positions"]:
                rows.append(base | {
                    "position": pos["position"],
                    "from_time": pos["from"], "to_time": pos["to"],
                    "from_period": pos["from_period"], "to_period": pos["to_period"],
                    "start_reason": pos["start_reason"], "end_reason": pos["end_reason"],
                })
    return rows


def match_row(m, cid, sid):
    return {
        "match_id": m["match_id"], "competition_id": cid, "season_id": sid,
        "competition": sub(m, "competition", "competition_name"),
        "season": sub(m, "season", "season_name"),
        "match_date": m.get("match_date"), "kick_off": m.get("kick_off"),
        "match_week": m.get("match_week"),
        "competition_stage": sub(m, "competition_stage", "name"),
        "stadium": sub(m, "stadium", "name"),
        "home_team_id": sub(m, "home_team", "home_team_id"),
        "home_team": sub(m, "home_team", "home_team_name"),
        "home_score": m.get("home_score"),
        "away_team_id": sub(m, "away_team", "away_team_id"),
        "away_team": sub(m, "away_team", "away_team_name"),
        "away_score": m.get("away_score"),
        "has_360": m.get("match_status_360") == "available",
    }


def convert(c):
    cid, sid = c["competition_id"], c["season_id"]
    stem = f"{cid}_{sid}.parquet"
    events_out = OUT / "events" / stem
    if events_out.exists():
        return 0, 0

    matches = json.load(open(SB / "matches" / str(cid) / f"{sid}.json"))
    matches = [m for m in matches if m.get("match_status") == "available"]
    if not matches:
        return 0, 0
    for d in ("events", "lineups", "matches"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    n_rows, raw_bytes = 0, 0
    batch, lineups = [], []
    writer = pq.ParquetWriter(events_out, EVENTS_SCHEMA, compression="zstd")
    for m in tqdm(matches, desc=f"{c['competition_name']} {c['season_name']}", leave=False):
        events_file = SB / "events" / f"{m['match_id']}.json"
        if not events_file.exists():
            continue  # coverage.py already reports these
        raw_bytes += events_file.stat().st_size
        for e in json.load(open(events_file)):
            batch.append(event_row(e, m["match_id"], cid, sid))
        lineups += lineup_rows(m["match_id"], cid, sid)
        if len(batch) >= BATCH_ROWS:
            writer.write_table(pa.Table.from_pylist(batch, schema=EVENTS_SCHEMA))
            n_rows += len(batch)
            batch = []
    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=EVENTS_SCHEMA))
        n_rows += len(batch)
    writer.close()

    pq.write_table(pa.Table.from_pylist(lineups, schema=LINEUPS_SCHEMA),
                   OUT / "lineups" / stem, compression="zstd")
    pq.write_table(pa.Table.from_pylist([match_row(m, cid, sid) for m in matches],
                                        schema=MATCHES_SCHEMA),
                   OUT / "matches" / stem, compression="zstd")
    return n_rows, raw_bytes


if __name__ == "__main__":
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    wanted = CFG["corpus"]["statsbomb"]
    comps = json.load(open(SB / "competitions.json"))

    total_rows, total_raw = 0, 0
    for c in comps:
        if only is not None and c["competition_id"] != only:
            continue
        if not in_corpus(wanted, c["competition_id"], c["season_name"]):
            continue
        n_rows, raw_bytes = convert(c)
        if n_rows:
            print(f"{c['competition_name']} {c['season_name']}: {n_rows:,} events")
        total_rows += n_rows
        total_raw += raw_bytes

    staged = sum(f.stat().st_size for f in OUT.rglob("*.parquet"))
    print(f"\n{total_rows:,} new event rows | staging now {staged / 1e9:.2f} GB"
          + (f" | this run read {total_raw / 1e9:.2f} GB raw JSON "
             f"(~{total_raw / max(staged, 1):.0f}x if all new)" if total_raw else ""))
