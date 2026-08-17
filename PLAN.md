# Player Style Similarity — Working Plan

*Short redraft (2026-08-13) of the original `~/football/PLAN.md` + `EXECUTION.md`, which stay as
the detailed archive. This file reflects every decision to date; rationale lives in
`decisions.md`, wiring in `structure.md`, phase math in `~/football/phasewise_notes.txt`.*

*Position (2026-08-16): data + measurement layers done (M0–M3, VAEP pending). Next: M4 —
similarity itself is now just a distance over the feature table; the phase's real work is the
eval harness, because nothing distinguishes a good neighbour list from a bad one without it.
Every later model (M5+, GPU) plugs into the same harness and must beat the M4 baselines.*

## The idea

Query a player, get players who make the **same choices in the same situations** — style, not
quality. Style is modeled as a conditional behaviour policy **π(choice | state, player)**;
quality (VAEP / xG+xA per 90, event-derived) is measured separately and removed from the style
representation. Unit of analysis: the **action** (sample-efficient on tournament data). Unit of
comparison: the entity **(player_id, team_id, season)** — a national team is a different team,
so club and country are separate by construction; never a whole career; shootouts always
excluded. No shrinkage anywhere (D15): sparse entities stay raw and visibly noisy, consumers
filter on the support columns.

## Data (all open, all pinned in manifest.yaml)

| Source | Role | Status |
|---|---|---|
| StatsBomb Open Data | the corpus: everything released except ISL — 3,846 matches, 13.57M events, 79 comp-seasons, 426 matches with 360 upstream | ✅ ingested |
| Wyscout (Pappalardo) | secondary provider: big-five 2017/18 + Euro 2016 (WC18 excluded — StatsBomb copy wins, §2.2) | ◐ SPADL-converted 2026-08-17; features/identity integration pending |
| Transfermarkt | bios, transfers (E5), appearances; **market value dropped (D13)** | ✅ on disk; men-only via id map |
| reep | id bridge StatsBomb↔Transfermarkt: men 73.2%, women 0% (structural) | ✅ joined |
| DFL + SkillCorner tracking | validation of spatial proxies only | on disk / to fetch |

Corpus realities: all-teams league seasons exist only for 2015/16 (top-4 leagues) + women's
leagues; everything else is a single-team/star slice, accepted knowingly (only form available).

## Pipeline (single-tier: Parquet + embedded DuckDB + DVC; no servers)

```
raw (pinned) → staging (flat typed parquet, format change only, validated)
             → spadl (7.78M actions, 105x68, 23 types, PK game_id+action_id)
             → features (17,826 entities x 66 share/rate features, self-describing names;
                         possession rates + team residualization in the same stage)
             + pass-difficulty model (OOF, AUC .861) + quality table (npxG+xA/90, separate)
             → models → eval → serve
dvc repro rebuilds only what changed; manifest pins upstream; config.yaml defines the corpus
```

## Milestones

| M | Content | Status |
|---|---|---|
| M0 | env, WSL/CUDA, venv, pins | ✅ |
| M1 | ingest, coverage, staging, validation, views, orientation check | ✅ |
| M2 | SPADL, identity join, Writeup #1 | ✅ (Wyscout tasks dormant) |
| M3 | features + possession rates + team deconfounding (r2 report) + pass-difficulty risk model + quality axis v1 (npxG+xA/90). **Leftover: per-action VAEP** — must land before M5's quality head | ✅ (VAEP pending) |
| M4 | **similarity v1 + the harness that judges it.** Eval E1–E8 runnable against any vector table; baselines z-score+cosine, Ledoit-Wolf Mahalanobis, Player Vectors (NMF role components ≈ soft clustering); first ranked-neighbour lists + scoreboard; E6 triplets sent to annotators NOW (latency spans M5–M7) | ← next |
| M5 | GPU begins. conditional behaviour encoder: π(c\|s,e_p), InfoNCE within entity, gradient-reversal quality head, shrunk embeddings + bootstrap CIs | — |
| M6 | player2vec + action transformer; fetch 360 → spatial head: opponents-in-radius, line-breaking, Voronoi, **attacker gravity/distortion**, pass-options counterfactual; profiling appendix | — |
| M7 | serve: DuckDB feature store, brute-force + hnswlib comparison, FastAPI `/similar` + explanation endpoint, Streamlit | — |
| M8 | human triplet eval, failure analysis, Writeup #2 | — |

Slip protection: M0–M4 alone is a complete publishable project. GPU is idle until M5 by design;
training uses VRAM-resident batching (whole feature tensor fits in 6 GB).

**M4 trajectory study** (findings deliverable, enabled by per-season keying): careers = ordered
season sequences of baseline vectors, never centroids. (1) adjacent drift ✅
(`reports/style_drift.md`: self 0.716 vs random 0.326; Messi's min is the PSG move); (2) DTW +
order-blind alternatives → nearest-career search (`notebooks/trajectory_study.ipynb`, hands-on);
(3) PCA career map; (4) verdict on whether DTW earns its keep at 2–18-season lengths. Watch:
gap seasons (220/388 careers), Barcelona slice bias in long careers.

## Evaluation (the primary deliverable)

E1 held-out-match retrieval · E2 cross-competition retrieval · E3 (dormant, cross-provider) ·
E4 linear probes — position should read HIGH, club LOW (integrity test) · E5 transfer
replacement pairs (men only) · E6 human triplets vs inter-annotator agreement · E7 everything
vs the published baseline · E8 men→women transfer. Plus a fixed sanity suite of pairs any
football watcher agrees on, run on every change.

## Rules that bind every step

1. Every count computed from disk, never asserted; silent skips are bugs (D10).
2. Style features are shares/rates; volume lives only in support columns (§ D6/D11).
3. Data outputs belong to DVC, never `git add`-ed; reports and code belong to git.
4. Negative results are deliverables: club-probe accuracy, encoder-vs-baseline losses,
   human disagreement, wide CIs on low-minute entities — all reported as found.
5. StatsBomb credited on everything published; no scraped sources, ever.

## Top risks

| Risk | Mitigation |
|---|---|
| Tournament/slice entities too sparse | action-level modeling + support-threshold filtering + bootstrap CIs (no shrinkage, D15) |
| Encoder underperforms baselines initially | expected; M4 baselines are the floor; budget debugging |
| Team system leaks into style | E4 club probe as gate; fixed-effect residualization |
| 360 only on 426 matches | optional branch + learned no-360 token; report ablation |
| Teammate co-occurrence poisons player2vec | club probe gate; role-token variant |
