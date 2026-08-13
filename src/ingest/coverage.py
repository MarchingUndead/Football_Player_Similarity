"""Coverage report: what is actually on disk, per competition-season.

Counts matches / events / players / minutes / 360 availability from the local
files and writes reports/00_coverage.md. Nothing is hardcoded: competitions
come from competitions.json, the include-list comes from config.yaml, and the
report is regenerated from scratch on every run — so upstream drift shows up
here instead of as a mystery bug in week 6.
"""

import json
from datetime import datetime, timezone

import yaml
from tqdm import tqdm

from corpus import ROOT, in_corpus, load_config

CFG = load_config()
SB = ROOT / CFG["paths"]["statsbomb"]
WY = ROOT / CFG["paths"]["wyscout"]


def statsbomb_coverage():
    comps = json.load(open(SB / "competitions.json"))
    wanted = CFG["corpus"]["statsbomb"]
    rows, problems = [], []

    for c in tqdm(comps, desc="statsbomb comp-seasons"):
        cid, sid = c["competition_id"], c["season_id"]

        row = {
            "competition": c["competition_name"],
            "season": c["season_name"],
            "gender": c["competition_gender"],
            "in_corpus": in_corpus(wanted, cid, c["season_name"]),
            "matches": 0, "matches_360": 0, "events": 0, "minutes": 0,
            "players": set(),
            # B5 guard: upstream has renamed columns before; None must not crash us
            "has_360_flag": bool(c.get("match_available_360")),
        }

        matches_file = SB / "matches" / str(cid) / f"{sid}.json"
        if not matches_file.exists():
            problems.append(f"{c['competition_name']} {c['season_name']}: "
                            f"listed in competitions.json but no matches file on disk")
            rows.append(row)
            continue

        for m in json.load(open(matches_file)):
            if m.get("match_status") != "available":
                continue
            events_file = SB / "events" / f"{m['match_id']}.json"
            if not events_file.exists():
                problems.append(f"{c['competition_name']} {c['season_name']}: "
                                f"match {m['match_id']} marked available but events file missing")
                continue
            events = json.load(open(events_file))

            row["matches"] += 1
            row["events"] += len(events)
            if m.get("match_status_360") == "available":
                row["matches_360"] += 1
            last_minute = 0
            for e in events:
                if e["minute"] > last_minute:
                    last_minute = e["minute"]
                if "player" in e:
                    row["players"].add(e["player"]["id"])
            row["minutes"] += last_minute + 1

        rows.append(row)

    # config targets that point at nothing on disk — report, never silently drop
    if wanted["include"] != "all":
        on_disk = {(c["competition_id"], c["season_name"]) for c in comps}
        disk_cids = {c["competition_id"] for c in comps}
        for cid, sel in wanted["include"].items():
            if cid not in disk_cids:
                problems.append(f"config wants competition_id {cid} — not in competitions.json at all")
            elif sel != "all":
                for season in sel:
                    if (cid, season) not in on_disk:
                        problems.append(f"config wants competition_id {cid} season {season} — not on disk")

    return rows, problems


def wyscout_coverage():
    wanted = CFG["corpus"]["wyscout"]
    rows = []

    event_files = sorted((WY / "events").glob("events_*.json"))
    for f in tqdm(event_files, desc="wyscout competitions"):
        name = f.stem.removeprefix("events_")
        row = {
            "competition": name.replace("_", " "),
            "in_corpus": name in wanted,
            "matches": 0, "events": 0, "minutes": 0,
            "players": set(),
        }

        matches_file = WY / "matches" / f"matches_{name}.json"
        if matches_file.exists():
            row["matches"] = len(json.load(open(matches_file)))

        events = json.load(open(f))
        row["events"] = len(events)
        # eventSec resets each half: per-match minutes = sum over periods of the last eventSec
        period_end = {}
        for e in events:
            if e["playerId"] != 0:
                row["players"].add(e["playerId"])
            if e["matchPeriod"] != "P":  # shootouts are not playing time
                key = (e["matchId"], e["matchPeriod"])
                if e["eventSec"] > period_end.get(key, 0):
                    period_end[key] = e["eventSec"]
        row["minutes"] = round(sum(period_end.values()) / 60)

        rows.append(row)

    return rows


def totals(rows):
    t = {"matches": 0, "matches_360": 0, "events": 0, "minutes": 0, "players": set()}
    for r in rows:
        for k in ("matches", "matches_360", "events", "minutes"):
            t[k] += r.get(k, 0)
        t["players"] |= r["players"]
    return t


def write_report(sb_rows, sb_problems, wy_rows):
    manifest = yaml.safe_load(open(ROOT / "manifest.yaml"))
    lines = [
        "# 00 — Coverage report",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `src/ingest/coverage.py`. "
        "Do not edit by hand — every number is computed from the local data.",
        "",
        f"- StatsBomb clone pinned at `{manifest['statsbomb']['commit']}`",
        f"- Wyscout collection `{manifest['wyscout']['collection_doi']}`",
        "- ✓ = in the corpus include-list in `config.yaml`; other rows are on disk but not ingested",
        "- players = distinct players appearing in events; minutes = summed match durations",
        "- 360 = matches with 360 frames *available upstream* (the three-sixty/ folder is not fetched until M6)",
        "",
        "## StatsBomb",
        "",
        "| corpus | competition | season | gender | matches | 360 | events | players | minutes |",
        "|---|---|---|---|--:|--:|--:|--:|--:|",
    ]

    order = sorted(sb_rows, key=lambda r: (not r["in_corpus"], r["competition"], r["season"]))
    for r in order:
        mark = "✓" if r["in_corpus"] else ""
        lines.append(f"| {mark} | {r['competition']} | {r['season']} | {r['gender']} "
                     f"| {r['matches']} | {r['matches_360']} | {r['events']:,} "
                     f"| {len(r['players']):,} | {r['minutes']:,} |")

    t = totals([r for r in sb_rows if r["in_corpus"]])
    lines += [
        "",
        f"**Corpus subset (✓ rows): {t['matches']:,} matches, {t['matches_360']:,} with 360, "
        f"{t['events']:,} events, {len(t['players']):,} distinct players, {t['minutes']:,} minutes.**",
        "",
        "## Wyscout",
        "",
        "| corpus | competition | matches | events | players | minutes |",
        "|---|---|--:|--:|--:|--:|",
    ]

    for r in sorted(wy_rows, key=lambda r: (not r["in_corpus"], r["competition"])):
        mark = "✓" if r["in_corpus"] else ""
        lines.append(f"| {mark} | {r['competition']} | {r['matches']} | {r['events']:,} "
                     f"| {len(r['players']):,} | {r['minutes']:,} |")

    t = totals([r for r in wy_rows if r["in_corpus"]])
    lines += [
        "",
        f"**Corpus subset (✓ rows): {t['matches']:,} matches, {t['events']:,} events, "
        f"{len(t['players']):,} distinct players, {t['minutes']:,} minutes.**",
        "",
        "## Problems",
        "",
    ]
    if sb_problems:
        for p in sb_problems:
            lines.append(f"- {p}")
    else:
        lines.append("None: every config target exists on disk, every available match has an events file.")

    out = ROOT / CFG["paths"]["reports"] / "00_coverage.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    if sb_problems:
        print(f"{len(sb_problems)} problem(s) — see the report")


if __name__ == "__main__":
    sb_rows, sb_problems = statsbomb_coverage()
    wy_rows = wyscout_coverage()
    write_report(sb_rows, sb_problems, wy_rows)
