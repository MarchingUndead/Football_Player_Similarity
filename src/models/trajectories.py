"""Career-trajectory data prep, shared by notebooks.

Loading + cleaning only: turn the baseline vector table into per-player ordered
season sequences. Modeling (DTW, alternative distances) deliberately lives in
the notebook — this module keeps analysis code short, it does not analyse.

A "career" here = one player's club entities ordered by season, each season one
z-scored style vector (the 2026-08-11 amendment: seasons are points, never
pooled). A player with two clubs in one season contributes two points; season
gaps are flagged, not hidden.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

META = {"player_id", "team_id", "season", "context", "matches", "actions",
        "minutes", "player", "team", "position", "pos_group"}


def load_vectors(root) -> pd.DataFrame:
    """The M4 baseline table: entity keys + support + 66 z-scored style features."""
    return pd.read_parquet(root / "data" / "models" / "baseline_vectors.parquet")


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META and df[c].dtype.kind == "f"]


def _start_year(season: str) -> int:
    return int(str(season)[:4])          # '2004/2005' -> 2004, '2022' -> 2022


@dataclass
class Career:
    player_id: float
    player: str
    pos_group: str
    seasons: list          # e.g. ['2004/2005', '2005/2006', ...]
    teams: list
    idx: np.ndarray        # row positions in the vectors table (for projections)
    X: np.ndarray          # (n_seasons, n_features) z-scored style matrix

    @property
    def has_gaps(self) -> bool:
        years = [_start_year(s) for s in self.seasons]
        return any(b - a > 1 for a, b in zip(years, years[1:]))


def build_sequences(vec: pd.DataFrame, min_minutes: int = 450,
                    context: str = "club", min_seasons: int = 2) -> dict:
    """player_id -> Career, for players with >= min_seasons solid entities."""
    d = vec[(vec.minutes >= min_minutes) & (vec.context == context)] \
        .sort_values("season")
    F = feature_cols(vec)
    out = {}
    for pid, g in d.groupby("player_id"):
        if len(g) < min_seasons:
            continue
        out[pid] = Career(pid, g.player.iloc[0], g.pos_group.iloc[0],
                          g.season.tolist(), g.team.tolist(),
                          g.index.to_numpy(), g[F].to_numpy(float))
    return out


def pca_project(vec: pd.DataFrame, features: list[str], min_minutes: int = 450):
    """2-D PCA of the style space: fit on solid entities, transform all rows.

    Returns (coords aligned with vec's rows, explained variance ratio).
    """
    from sklearn.decomposition import PCA
    p = PCA(n_components=2).fit(vec.loc[vec.minutes >= min_minutes, features])
    return p.transform(vec[features]), p.explained_variance_ratio_
