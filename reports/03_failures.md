# Failures log

Running record of data bugs and upstream quirks found, with evidence and the decision
taken. Maintained by hand; newest last. (PLAN §4.2 predicted entry 1 would exist.)

## 1 — 2026-08-12: staging validation caught two upstream quirks (M1-T8, first run)

The first full run of `src/ingest/validate.py` failed with two findings. Both were
investigated and turned out to be StatsBomb data quirks, not pipeline bugs.

**a) Sub-yard coordinate overshoots.** ~700 of 13,567,319 rows carry coordinates
slightly outside the nominal 120×80 pitch (observed max x = 120.9, y = 80.9; start
and end coordinates alike; concentrated in the older Messi-era La Liga files but
present in modern comps too). These are edge-of-pitch situations — throw-ins, balls
going out — recorded a step beyond the line.
**Decision:** staging keeps the values untouched (D7 — no semantic edits at staging);
the validator's bounds widened to ±1 yard as the provider's real envelope. Anything
beyond ±1 still fails. Downstream zone-binning must clip to [0,120]×[0,80] at the
feature stage.

**b) Half-boundary timestamp regressions: 582 "backwards" events, two shapes.**
*Clock resets* (546 + 9): a pass in stoppage time whose `Ball Receipt*` is stamped on
the *next* period's reset clock (`00:00:00.x`, one as late as `00:01:27`) while still
belonging to the previous period. Occurs at normal half ends (prev ~`00:46`) **and** at
extra-time period ends (prev ~`00:16`, periods 3/4 — the WC 2022 final has one). *Whistle jitter* (34 + 2):
`Half End` / `Pressure` rows stamped 0.1–2.4 s *before* the last action they follow —
annotation-order noise at the whistle. StatsBomb's `index` column orders all of them
correctly.
**Decision:** `index` is the canonical event order (already how staging and everything
downstream sorts). The monotonicity check exempts exactly two patterns — jitter ≤ 5 s,
and resets where the previous event sits in the period's stoppage time (≥ 40' in
halves, ≥ 12' in extra-time periods) and the new stamp is inside the reset clock's
first 3 minutes; anything else still fails. Duration-like features must never be
computed from raw timestamp differences across such boundaries.

## 2 — 2026-08-12: SPADL rebuild differs by exactly one row (unexplained, watched)

The SPADL layer was built twice (the second build after a DVC bookkeeping incident
forced a re-run). First build: 7,775,283 actions; rebuild: 7,775,282 — same 3,846
games, zero duplicate (game_id, action_id), and the one available per-competition
reference (Champions League finals, 38,705) matches exactly. The differing row could
not be localized because per-competition counts from the first build were not kept.
**Decision:** accepted — one row in 7.8M with all structural checks passing. Guard
added for the future: per-competition action counts now live in this repo's records
via `dvc.lock` hashes, and any future full rebuild should be diffed per comp-season
before the old layer is discarded. Also from this incident: `to_spadl.py` ends with
`os._exit(0)` because pyarrow/jemalloc threads deadlocked interpreter shutdown
(6h futex wait after all work completed), which stalled `dvc repro`; and data
outputs must never be `git add`-ed — DVC refuses outputs git already tracks.
