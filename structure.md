# Code structure — who reads what, who produces what

Kept current with every change that adds/moves a file or dependency, so diffs
between working sessions stay legible. Arrows = "reads from / writes to".

## Data flow (current state, end of M1-T7)

```
config.yaml ───────────────┐  (paths + corpus include-list — single source of truth)
manifest.yaml ─────────────┤  (upstream pins: SHAs, DOIs, hashes — evidence)
                           │
        ┌──────────────────┴───────────────────┐
        │                                      │
src/ingest/corpus.py                           │
  ROOT, load_config(), in_corpus()             │
        │ imported by                          │
        ├──────────────────┐                   │
        ▼                  ▼                   │
src/ingest/coverage.py   src/ingest/to_parquet.py   ← both walk data/raw/statsbomb/data/
        │                  │                        (competitions.json → matches/ → events/, lineups/)
        ▼                  ▼
reports/00_coverage.md   data/staging/statsbomb/
  (committed, regenerated       events/{cid}_{sid}.parquet   ~55 typed cols, 13.57M rows
   on corpus change)            lineups/{cid}_{sid}.parquet  position stints → minutes
                                matches/{cid}_{sid}.parquet  metadata, scores, has_360
```

## Pipeline orchestration

```
dvc.yaml
  staging_statsbomb:   to_parquet.py            -> data/staging/statsbomb
  validate_staging:    validate.py    (dep: staging) -> reports/staging_validation.md
  orientation_check:   orientation_check.py (dep: staging) -> reports/orientation_check.md
  build_views:         build_views.py (dep: staging) -> data/football.duckdb
  spadl_statsbomb:     to_spadl.py    (dep: manifest pin, reads raw via socceraction) -> data/spadl/statsbomb
```
`dvc repro` rebuilds only what changed; the three downstream stages re-run whenever
staging does. The StatsBomb clone is deliberately NOT a dep — manifest.yaml's
pinned SHA stands in for it (D3).

## File inventory

| File | Role | Depends on | Consumed by |
|---|---|---|---|
| `config.yaml` | paths + corpus definition | — | corpus.py (and via it, everything) |
| `manifest.yaml` | upstream provenance pins | updated on download only | dvc.yaml dep; report headers |
| `src/ingest/corpus.py` | shared config/include logic | config.yaml | coverage.py, to_parquet.py |
| `src/ingest/coverage.py` | counts what's on disk | corpus.py, raw statsbomb+wyscout | writes reports/00_coverage.md |
| `src/ingest/to_parquet.py` | raw JSON → staging Parquet | corpus.py, raw statsbomb | writes data/staging/statsbomb/ |
| `src/ingest/validate.py` | T8: pandera + set checks over staging | corpus.py, staging | writes reports/staging_validation.md |
| `src/ingest/orientation_check.py` | T10: mirrored-coordinate tripwire | corpus.py, staging | writes reports/orientation_check.md |
| `src/ingest/build_views.py` | T9: DuckDB views + minutes + entity roster | corpus.py, staging, competitions.json | writes data/football.duckdb |
| `src/harmonise/to_spadl.py` | M2-T1: SPADL action table (D12) | corpus.py, raw statsbomb via socceraction | writes data/spadl/statsbomb/ |
| `src/harmonise/identity_join.py` | M2-T2: StatsBomb→reep→Transfermarkt id map | staging, raw reep + transfermarkt, overrides csv | writes data/identity/player_map.parquet + report |
| `src/harmonise/identity_overrides.csv` | manual id fixes, win over the bridge | hand-edited | identity_join.py |
| `dvc.yaml` / `dvc.lock` | pipeline stage + recorded hashes | see above | `dvc repro` |
| `notebooks/00_data_tour.ipynb` | exploration only, not pipeline | raw data | human |
| `notebooks/example_match_events.ipynb` | event-map visualization of one match half | staging Parquet, config.yaml | human |
| `data/raw/*.dvc` | DVC tracking of raw archives | — | `dvc pull/checkout` |

## Invariants to preserve in future changes

- Every staged row carries (match_id, competition_id, season_id); player rows
  additionally player_id — the (player, comp, season) key (D6) must survive
  every downstream table.
- Corpus membership is asked via `in_corpus()`, never re-implemented.
- Scripts resolve paths from `ROOT` (repo root via `corpus.py`), so they run
  from anywhere; entry points assume the venv, not system python.
- Staging row totals must reconcile with coverage event totals after any
  corpus or schema change (D10).

## Next additions (planned, not yet built)

- M2 Writeup #1 → reports/01_corpus_normalisation.md
- M3 feature stage → reads data/spadl/, grouped by (player, season, context) per D6/D11
