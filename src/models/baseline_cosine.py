"""M4 first cut: baseline style vectors + same-player season similarity.

Vector = the 66 style features of the team-adjusted table (residualized shares/
rates), z-scored over solid entities (>= MIN_MINUTES), NaN -> 0 (= population
mean; a player with no shots is average-on-shot-features, not excluded).
Distance = cosine. Support columns never enter the vector.

Per the 2026-08-11 amendment, seasons are NEVER pooled: each season is a point,
a career is an ordered sequence of points. This stage ranks a player's own
seasons against each other — stylistic drift — and writes:

  data/models/baseline_vectors.parquet             (keys + support + z features)
  data/models/same_player_season_similarity.parquet (all within-player pairs)
  reports/style_drift.md                            (self vs random reference,
      Messi trajectory, biggest drifts / most stable players)

The self-vs-random gap is the first pre-E1 sanity signal: if a player's own
seasons are not far more similar than random same-position pairs, no fancier
model has anything to learn.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from corpus import ROOT, load_config  # noqa: E402

CFG = load_config()
ADJUSTED = ROOT / "data" / "features" / "player_season_team_adjusted.parquet"
OUT_VECTORS = ROOT / "data" / "models" / "baseline_vectors.parquet"
OUT_PAIRS = ROOT / "data" / "models" / "same_player_season_similarity.parquet"

META = {"player_id", "team_id", "season", "context", "matches", "actions",
        "minutes", "player", "team", "position", "pos_group", "team_group_size"}
MIN_MINUTES = 450          # D15: explicit support threshold, stated everywhere
RNG = np.random.default_rng(0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else float("nan")


if __name__ == "__main__":
    df = pd.read_parquet(ADJUSTED)
    feats = [c for c in df.columns if c not in META and df[c].dtype.kind == "f"]
    solid = df.minutes >= MIN_MINUTES
    print(f"{len(df):,} entities, {solid.sum():,} with >= {MIN_MINUTES} min; "
          f"{len(feats)} features")

    mu = df.loc[solid, feats].mean()
    sd = df.loc[solid, feats].std().replace(0, 1)
    Z = ((df[feats] - mu) / sd).fillna(0.0)
    vectors = pd.concat(
        [df[["player_id", "team_id", "season", "context", "player", "team",
             "pos_group", "matches", "actions", "minutes"]], Z], axis=1)
    OUT_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    vectors.to_parquet(OUT_VECTORS, compression="zstd", index=False)

    # --- all within-player season pairs, solid entities only ----------------
    V = Z.to_numpy(dtype=np.float64)
    rows = []
    for pid, g in df[solid].groupby("player_id"):
        if len(g) < 2:
            continue
        idx = g.index.to_numpy()
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = df.loc[idx[i]], df.loc[idx[j]]
                rows.append({
                    "player_id": pid, "player": a.player,
                    "pos_group": a.pos_group,
                    "season_a": a.season, "team_a": a.team, "context_a": a.context,
                    "season_b": b.season, "team_b": b.team, "context_b": b.context,
                    "minutes_a": a.minutes, "minutes_b": b.minutes,
                    "cosine": cosine(V[idx[i]], V[idx[j]]),
                })
    pairs = pd.DataFrame(rows).sort_values(["player", "season_a", "season_b"])
    pairs.to_parquet(OUT_PAIRS, compression="zstd", index=False)

    # --- reference: random different-player pairs within position group ----
    ref = []
    sdf = df[solid]
    for _, g in sdf.groupby("pos_group"):
        idx = g.index.to_numpy()
        if len(idx) < 2:
            continue
        i = RNG.choice(idx, size=min(20000, 4 * len(idx)))
        j = RNG.choice(idx, size=len(i))
        keep = sdf.loc[i, "player_id"].to_numpy() != sdf.loc[j, "player_id"].to_numpy()
        ref += [cosine(V[x], V[y]) for x, y in zip(i[keep], j[keep])]
    ref = np.array(ref)

    self_mean = pairs.cosine.mean()
    print(f"within-player pairs: {len(pairs):,}, mean cosine {self_mean:+.3f}")
    print(f"random same-position pairs: {len(ref):,}, mean cosine {ref.mean():+.3f}")

    # --- adjacent club seasons per player: drift ranking --------------------
    club = df[solid & (df.context == "club")].sort_values("season")
    adj = []
    for pid, g in club.groupby("player_id"):
        if len(g) < 2:
            continue
        idx = g.index.to_numpy()
        for i in range(len(idx) - 1):
            a, b = df.loc[idx[i]], df.loc[idx[i + 1]]
            adj.append({"player": a.player, "pos_group": a.pos_group,
                        "from_s": f"{a.season} {a.team}", "to_s": f"{b.season} {b.team}",
                        "cosine": cosine(V[idx[i]], V[idx[i + 1]])})
    adj = pd.DataFrame(adj)

    messi = pairs[pairs.player.str.contains("Messi Cuccittini", na=False)
                  & (pairs.context_a == "club") & (pairs.context_b == "club")]
    messi_adj = adj[adj.player.str.contains("Messi Cuccittini", na=False)]

    report = ROOT / CFG["paths"]["reports"] / "style_drift.md"
    lines = [
        "# Same-player season similarity (baseline cosine, M4)",
        "",
        "Regenerated by `src/models/baseline_cosine.py` on every `dvc repro`. Do not edit.",
        "",
        f"Vectors: {len(feats)} z-scored residualized style features; entities with >= "
        f"{MIN_MINUTES} minutes; cosine similarity.",
        "",
        f"- within-player season pairs: **{len(pairs):,}**, mean cosine **{self_mean:+.3f}**",
        f"- random same-position-group pairs: mean cosine **{ref.mean():+.3f}** "
        f"(the gap is the identity signal any model must beat)",
        "",
        "## Messi, adjacent club seasons",
        "",
        "| from | to | cosine |",
        "|---|---|--:|",
    ] + [f"| {r.from_s} | {r.to_s} | {r.cosine:+.3f} |"
         for r in messi_adj.itertuples()] + [
        "",
        "## Biggest adjacent-season drifts (style changed most)",
        "",
        "| player | pos | from | to | cosine |",
        "|---|---|---|---|--:|",
    ] + [f"| {r.player} | {r.pos_group} | {r.from_s} | {r.to_s} | {r.cosine:+.3f} |"
         for r in adj.nsmallest(10, "cosine").itertuples()] + [
        "",
        "## Most stable adjacent seasons",
        "",
        "| player | pos | from | to | cosine |",
        "|---|---|---|---|--:|",
    ] + [f"| {r.player} | {r.pos_group} | {r.from_s} | {r.to_s} | {r.cosine:+.3f} |"
         for r in adj.nlargest(10, "cosine").itertuples()]
    report.write_text("\n".join(lines) + "\n")

    print(f"Messi club-season pairs: {len(messi):,}; adjacent: {len(messi_adj)}")
    print(f"-> {OUT_VECTORS.relative_to(ROOT)}, {OUT_PAIRS.relative_to(ROOT)}, "
          f"{report.relative_to(ROOT)}")
