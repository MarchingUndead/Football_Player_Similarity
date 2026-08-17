"""M4: Player Vectors baseline (Decroos & Davis, ECML PKDD 2019).

The published baseline this project must beat. Per action type, an entity's
start locations become a smoothed per-90 heatmap on a 24x16 grid; each action
type's heatmap matrix is compressed with NMF (non-negativity -> parts-based,
interpretable role components); component weights concatenate into one vector
per (player, team, season). Unlike the cosine baseline this KEEPS volume —
per-90 counts, not shares — as the paper specifies.

Also fits KMeans on the vectors: the role-cluster view ("which player types
exist"), with exemplars per cluster and a PCA map.

Outputs:
  data/models/player_vectors.parquet        (keys + 20 component weights + cluster)
  reports/player_vectors.md                 (components, clusters, case studies)
  reports/figures/player_vectors_map.png
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
from corpus import ROOT, load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harmonise"))
from spadl_union import actions_union_sql, comps_union_sql  # noqa: E402

CFG = load_config()
SPADL = ROOT / "data" / "spadl" / "statsbomb"
DB = ROOT / "data" / "football.duckdb"
FEATURES = ROOT / "data" / "features" / "player_season_context.parquet"
OUT = ROOT / "data" / "models" / "player_vectors.parquet"

GX, GY = 24, 16                      # grid cells: x own->attacking, y across
TYPES = ["pass", "cross", "dribble", "take_on", "shot"]   # dribble = carry (SPADL)
K = 4                                # NMF components per action type
N_CLUSTERS = 8
MIN_MINUTES = 450

CELLS_SQL = f"""
WITH raw_actions AS ({actions_union_sql(ROOT)}),
comps AS ({comps_union_sql(ROOT)})
SELECT a.player_id, a.team_id, c.season, t.type_name,
       least(floor(a.start_x / ({105 / GX})), {GX - 1})::INT AS cx,
       least(floor(a.start_y / ({68 / GY})), {GY - 1})::INT AS cy,
       count(*) AS n
FROM raw_actions a
JOIN '{SPADL}/actiontypes.parquet' t USING (type_id)
JOIN comps c USING (competition_id, season_id)
WHERE a.period_id <= 4 AND t.type_name IN ({','.join(f"'{t}'" for t in TYPES)})
GROUP BY ALL
"""

if __name__ == "__main__":
    con = duckdb.connect(str(DB), read_only=True)
    cells = con.sql(CELLS_SQL).df()
    ent = pd.read_parquet(FEATURES, columns=["player_id", "team_id", "season",
                                             "context", "player", "team",
                                             "pos_group", "minutes"])
    ent = ent[ent.minutes >= MIN_MINUTES].reset_index(drop=True)
    ent["row"] = ent.index
    cells = cells.merge(ent[["player_id", "team_id", "season", "row", "minutes"]],
                        on=["player_id", "team_id", "season"])

    # one smoothed per-90 heatmap matrix per action type, NMF'd to K components
    weights, parts = [], {}
    for t in TYPES:
        M = np.zeros((len(ent), GX * GY))
        sub = cells[cells.type_name == t]
        M[sub.row, (sub.cx * GY + sub.cy)] = 90 * sub.n / sub.minutes
        M = np.stack([gaussian_filter(r.reshape(GX, GY), sigma=1).ravel()
                      for r in M])
        nmf = NMF(n_components=K, init="nndsvda", max_iter=600, random_state=0)
        W = nmf.fit_transform(M)
        weights.append(W)
        parts[t] = nmf
        print(f"{t:8s}: recon err {nmf.reconstruction_err_:.1f}")
    V = np.hstack(weights)           # (entities, len(TYPES) * K)
    dims = [f"{t}_c{i}" for t in TYPES for i in range(K)]

    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0)
    # cluster on standardized dims so high-volume types don't own the clustering
    Vz = (V - V.mean(0)) / np.where(V.std(0) > 0, V.std(0), 1)
    ent["cluster"] = km.fit_predict(Vz)

    out = pd.concat([ent.drop(columns="row"),
                     pd.DataFrame(V, columns=dims)], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, compression="zstd", index=False)

    # --- case study: Messi / Neymar / Yamal --------------------------------
    def entity_row(name, season=None, team=None):
        m = ent[ent.player.str.contains(name, na=False)]
        if season is not None:
            m = m[m.season == season]
        if team is not None:
            m = m[m.team == team]
        return m.minutes.idxmax()

    def ranks_from(q_idx, targets):
        d = np.linalg.norm(V - V[q_idx], axis=1)
        order = np.argsort(d)
        rank = np.empty_like(order); rank[order] = np.arange(len(order))
        return [{"target": f"{ent.player[t].split()[0]} {ent.team[t]} {ent.season[t]}",
                 "cluster": int(ent.cluster[t]), "rank": int(rank[t]),
                 "of": len(ent), "dist": float(d[t])} for t in targets]

    messi_club = entity_row("Messi", "2014/2015")
    messi_arg = entity_row("Messi", "2022")   # Copa24 is 447 min < threshold
    targets = [entity_row("Neymar", "2015/2016"), entity_row("Neymar", "2018"),
               entity_row("Yamal")]
    case = {"Messi Barcelona 2014/2015": ranks_from(messi_club, targets),
            "Messi Argentina WC2022": ranks_from(messi_arg, targets)}

    # --- report + map -------------------------------------------------------
    lines = [
        "# Player Vectors baseline (Decroos & Davis 2019) — M4",
        "",
        "Regenerated by `src/models/m1_player_vectors.py` on every `dvc repro`. Do not edit.",
        "",
        f"{len(ent):,} entities (>= {MIN_MINUTES} min) x {len(dims)} dims "
        f"({K} NMF components x {len(TYPES)} action types, {GX}x{GY} per-90 "
        "smoothed heatmaps). Euclidean distance, per the paper. Volume is kept",
        "(per-90 counts, not shares) — differences vs the cosine baseline are",
        "informative, not bugs.",
        "",
        "## Clusters (KMeans on standardized weights)",
        "",
        "| cluster | n | position mix | exemplars (closest to centroid) |",
        "|---|--:|---|---|",
    ]
    for c in range(N_CLUSTERS):
        m = ent.cluster == c
        pos = ent[m].pos_group.value_counts(normalize=True).head(2)
        posmix = ", ".join(f"{i} {v:.0%}" for i, v in pos.items())
        d = np.linalg.norm(Vz[m.values] - km.cluster_centers_[c], axis=1)
        ex = ent[m].iloc[np.argsort(d)[:5]]
        names = "; ".join(f"{r.player.split()[-1]} ({r.team} {r.season})"
                          for r in ex.itertuples())
        lines.append(f"| {c} | {m.sum()} | {posmix} | {names} |")

    lines += ["", "## Case study: is Yamal/Neymar near Messi?", ""]
    for q, rows in case.items():
        lines += [f"**query: {q}** (cluster "
                  f"{int(ent.cluster[messi_club if 'Barcelona' in q else messi_arg])})",
                  "", "| target | cluster | rank | dist |", "|---|--:|--:|--:|"]
        lines += [f"| {r['target']} | {r['cluster']} | {r['rank']}/{r['of']} "
                  f"| {r['dist']:.2f} |" for r in rows]
        lines += [""]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    P = PCA(2, random_state=0).fit_transform(Vz)
    PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    fig, ax = plt.subplots(figsize=(9, 7), dpi=130)
    for c in range(N_CLUSTERS):
        m = (ent.cluster == c).values
        ax.scatter(P[m, 0], P[m, 1], s=7, c=PALETTE[c], linewidths=0,
                   label=f"{c} (n={m.sum()})", alpha=.7)
    for i, lab in [(messi_club, "Messi 14/15"), (targets[0], "Neymar 15/16"),
                   (targets[2], "Yamal E24"), (messi_arg, "Messi WC22")]:
        ax.scatter(*P[i], s=90, facecolor="none", edgecolor="#0b0b0b", lw=1.6)
        ax.annotate(lab, P[i], xytext=(7, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold")
    ax.legend(title="cluster", fontsize=8, markerscale=2, loc="best")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Player Vectors space (PCA), KMeans role clusters")
    figdir = ROOT / CFG["paths"]["reports"] / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "player_vectors_map.png", bbox_inches="tight")
    lines += ["", "![map](figures/player_vectors_map.png)"]

    (ROOT / CFG["paths"]["reports"] / "player_vectors.md") \
        .write_text("\n".join(lines) + "\n")
    print(f"-> {OUT.relative_to(ROOT)}, reports/player_vectors.md")
    for q, rows in case.items():
        print(q, "->", [(r['target'], r['rank']) for r in rows])
