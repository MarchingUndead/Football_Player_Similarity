# Architectural & system-level decisions

Running log, newest last. Each entry: what was decided, why, and what it forces
downstream. Update this file in the same change that implements a decision.

## D1 — config.yaml is the single source of truth (2026-08-10)
All paths and the corpus definition live in `config.yaml` and nowhere else.
Scripts read it; none hardcode paths or competition lists.
**Why:** one place to change, no drift between scripts.
**Forces:** any new script imports `src/ingest/corpus.py` helpers instead of
re-parsing config or re-implementing include logic.

## D2 — manifest.yaml is evidence, not configuration (2026-08-10)
`manifest.yaml` records what the outside world gave us: upstream commit SHAs,
DOI versions, file hashes, fetch dates. Updated only when a download happens.
**Why:** distinguishes "I have a bug" from "the world moved"; makes reported
numbers reproducible against pinned data.
**Forces:** re-fetching data without updating the manifest is a process bug.

## D3 — DVC tracks what we produce; git-pinned upstream stays out of DVC (2026-08-10)
`dvc add` for raw archives we control (wyscout, transfermarkt, reep);
`dvc.yaml` stages for produced artifacts (staging Parquet onward). The
StatsBomb clone is never dvc-added — its own git SHA pins it, and
`manifest.yaml` stands in for it as a stage dependency.
**Why:** content-hashing a 14 GB clone duplicates what git already guarantees.
**Forces:** `dvc repro` re-runs a stage only when code, config, or the
manifest pin changes — not when untracked raw files are touched.

## D4 — StatsBomb-only corpus; Wyscout deferred (2026-08-11)
Wyscout stays on disk, pinned, but nothing is ingested from it. The plan's
provider-precedence and carry-inference-benchmark machinery is dormant, not
deleted.
**Why:** one provider = no format harmonisation debt while the pipeline is
built; StatsBomb alone now supplies 13.5M events.
**Forces:** if Wyscout returns, it enters through the same staging pattern and
the SPADL stage must then resolve coordinate conventions (percentages vs
120×80) and the missing-carries problem.

## D5 — everything StatsBomb released enters, including partial slices (2026-08-11)
Single-team / star-player releases (Messi seasons, Arsenal 03/04, Leverkusen
23/24, old WC selections) are in the corpus. Indian Super League is the only
exclusion. There is no complete version of the sliced seasons to prefer —
StatsBomb never released one.
**Why:** the slices are the only source for the longitudinal (Messi) and
era-control (Arsenal) studies.
**Forces:** per-entity sample sizes vary by ~100× (a slice opponent may have
1 match; a 15/16 league player has 38). Shrinkage + confidence intervals
(PLAN Part 5) are load-bearing, not optional.

## D6 — unit of representation is (player_id, competition_id, season_id) (2026-08-11)
Never pool a career into one vector. Internationals are separate entities from
club seasons by construction of the key. Contrastive training positives may
only come from within one player-competition-season; the plan's
"positives across club changes" idea is dropped (it would train career pooling
into the embeddings).
**Why:** playing styles drift across a career — KDB 17/18 vs Özil 15/16 is a
meaningful comparison; KDB-the-career is not a style.
**Forces:** every feature aggregate, embedding, index entry and API result is
keyed by the triple. Cross-season similarity of the same player becomes a
*finding* (drift trajectories) and an *evaluation* (E1/E2), never an input
assumption.

## D7 — staging = format change only, no semantic decisions (2026-08-11)
`data/staging/statsbomb/` re-houses raw JSON as flat, explicitly-typed Parquet:
no coordinate conversion, no vocabulary mapping, no filtering beyond the corpus
include-list. Deep blobs (shot freeze frames, tactics) ride as JSON strings.
The raw clone remains the lossless source of record.
**Why:** semantic normalisation (SPADL, 105×68, action vocabulary) belongs in
one place downstream where it can be validated once; staging stays rebuildable
and argument-free.
**Forces:** SPADL stage (M2) reads staging Parquet, never raw JSON. Any field
staging didn't flatten is still recoverable from raw without re-downloading.

## D8 — one Parquet file per competition-season; three tables (2026-08-11)
`events/`, `lineups/` (position stints → minutes), `matches/` — each keyed
`{competition_id}_{season_id}.parquet`, every row carrying
(match_id, competition_id, season_id).
**Why:** comp-season is the natural rebuild unit, matches DuckDB glob-query
patterns, and keeps any single conversion inside the RAM ceiling.
**Forces:** targeted rebuilds are per competition; queries address the corpus
as `events/*.parquet`, never individual files.

## D9 — streamed conversion under a fixed RAM ceiling (2026-08-11)
The corpus is never loaded whole: one comp-season at a time, rows flushed to
the ParquetWriter every 200k events, explicit pyarrow schema so a type
mismatch fails loudly at write time.
**Why:** 14 GB JSON through a 10 GB WSL RAM budget; inferred schemas can
silently vary between batches.
**Forces:** schema changes are deliberate edits to `EVENTS_SCHEMA`, and adding
columns means a staging rebuild (`dvc repro`).

## D10 — every count is computed, never asserted (2026-08-10, ongoing)
Coverage numbers come from disk at run time; anything skippable is logged to a
Problems section instead of silently dropped; independent counts are used to
cross-check each other (staging row total must equal coverage event total).
**Why:** the upstream repo drifts weekly; silent skips are the classic ingest
bug.
**Forces:** hardcoding a match/event count anywhere is a review-blocking bug.

## D11 — within a season, club and country pool separately; shootouts never count (2026-08-12)
Refines D6: the represented entity is **(player_id, season, context)** with context ∈
{club, country}. All club competitions of one season pool into one entity (league +
cups + European competition); national-team competitions pool into their own. The
split comes from StatsBomb's `competition_international` flag (Champions League
correctly = club), never from name matching. Penalty-shootout events (period 5)
are excluded from every aggregate.
**Why:** one season of one player is one behavioural sample — splitting by
competition fragments already-small samples, but club and country are genuinely
different contexts (system, teammates, stakes). Shootout kicks are not open-play
choices and silently inflate goal counts.
**Forces:** feature aggregation (M3) groups by (player, season, context);
`period <= 4` is a standard filter in every aggregate; the entity roster view
carries a `context` column.

## D12 — SPADL via socceraction's loader over pinned raw, not over staging (2026-08-12)
The SPADL stage (`src/harmonise/to_spadl.py`) reads the raw StatsBomb clone through
socceraction's maintained `StatsBombLoader` + `convert_to_actions`, then
`play_left_to_right`. This is a deliberate exception to "downstream reads staging"
(D7): reconstructing the loader's exact input format from our staging Parquet would
re-implement maintained conversion code and adopt its edge cases as our bugs.
**Why:** converter correctness is worth more than layer purity; the raw clone is
pinned by manifest.yaml, so reproducibility is unaffected.
**Forces:** the SPADL output keeps `original_event_id`, so actions still join back
to staging rows (which remain the source for StatsBomb-only riches: pressures,
freeze frames, shot z). PK of the action table: (game_id, action_id).

## D13 — market value dropped entirely; quality axis is event-derived (2026-08-12)
Transfermarkt market value is removed from the project — not used as the quality
axis, not used anywhere. Quality is measured from the event data itself
(VAEP / xG+xA per 90, PLAN §5.2), which every corpus entity has by construction.
**Why:** market value is heavily confounded by country/league economics, so it is
not a clean quality signal for similarity work — and the identity join showed it
would only exist for men anyway (73.2% of men resolved, 0% of women; D-report
`reports/identity_resolution.md`).
**Forces:** Transfermarkt's remaining jobs are bios (age/height/foot/position),
transfer histories (E5 replacement pairs), and appearance counts (shrinkage
support) — all men-only via the identity map. Nothing downstream may join
`player_valuations.csv`.

## D14 — entity key is (player_id, team_id, season); context becomes derived (2026-08-15)
Supersedes the D6/D11 formulation. The comparable entity is one player at one
team in one season. National teams are distinct team_ids, so club/country
separation holds **by construction** instead of by a context flag; the flag
survives as a derived attribute (is the team's competition international) for
filtering and eval. Mid-season transfers now split into separate entities —
correct, since a different system is a different behavioural context. Same-team
competitions still pool (Barcelona league + CL final = one entity).
**Why:** the previous key silently pooled two clubs when a player transferred
within a season; team is part of what identifies a behavioural sample.
**Forces:** features, adjustments, embeddings and the index all key on
(player_id, team_id, season). Identification is by player_id, never name —
names collide.

## D15 — empirical-Bayes shrinkage dropped; transparent support filtering instead (2026-08-15)
No feature is pulled toward a positional prior. Sparse entities keep their raw
(possession-adjusted, team-residualized) values, and every table carries support
columns (matches, minutes, actions, team group size) so consumers filter
explicitly — e.g. "entities with 450+ minutes" — instead of trusting an
invisible prior.
**Why:** shrinkage blends an entity's data with an assumed population, and how
much blending happened is invisible at query time; a filter is auditable.
**Forces:** low-support entities are noisy and *visibly* so; eval must always
state its support threshold; bootstrap confidence intervals (M5) remain planned
— they quantify noise without altering point estimates.
