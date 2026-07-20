"""Orthogonal ranking features: signals BERT4Rec never sees.

The Phase 3 measurement showed a reranker fed only retriever-derived
features cannot beat the retriever (clean-protocol NDCG@10 went DOWN).
The diagnosis predicts the remedy: features carrying information the
retriever lacks. ML-1M ships several, unused until now:

  movies.dat  genres (18 flags), release year parsed from the title
  users.dat   gender, age bucket, occupation category
  ratings     item-item co-occurrence counts (window-based PMI), a
              count-statistic view of the sequence data that the
              transformer summarizes differently

Per-candidate features produced:
  genre_sim_profile   cosine(candidate genres, mean genre vector of the
                      user's last 20 history items)
  genre_sim_last      jaccard(candidate genres, last history item genres)
  item_year           release year (0 if unparseable)
  cooc_pmi_mean       mean positive PMI between the candidate and the
                      user's last 5 history items
  user_gender, user_age, user_occupation
                      constant within a user's candidate group, so they
                      rank nothing alone -- their value is interactions:
                      tree splits like (age<=24) -> (genre_horror) learn
                      demographic-conditioned genre preferences, which is
                      exactly how GBDT rankers consume user context.

user_occupation is declared categorical to LightGBM (its integer coding
is arbitrary); age buckets and gender are ordinal/binary and stay
numeric.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
_GENRE_IDX = {g: i for i, g in enumerate(GENRES)}
_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

AUX_FEATURE_NAMES = [
    "genre_sim_profile", "genre_sim_last", "item_year", "cooc_pmi_mean",
    "user_gender", "user_age", "user_occupation",
]
AUX_CATEGORICAL = ["user_occupation"]


def load_movie_features(movies_path: str | Path, item_id_map: dict,
                        n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """movies.dat -> (genre matrix (n_items+1, 18), year array (n_items+1,))."""
    genre_mat = np.zeros((n_items + 1, len(GENRES)), dtype=np.float32)
    years = np.zeros(n_items + 1, dtype=np.float32)
    with open(movies_path, encoding="latin-1") as f:
        for line in f:
            movie_id, title, genres = line.rstrip("\n").split("::")
            dense = item_id_map.get(int(movie_id))
            if dense is None:
                continue  # movie exists but has no interactions
            for g in genres.split("|"):
                idx = _GENRE_IDX.get(g)
                if idx is not None:
                    genre_mat[dense, idx] = 1.0
            m = _YEAR_RE.search(title)
            if m:
                years[dense] = float(m.group(1))
    return genre_mat, years


def load_user_features(users_path: str | Path,
                       user_ids: Sequence[int]) -> np.ndarray:
    """users.dat -> (n_users, 3) array of [gender, age, occupation],
    row-aligned with the split's user order."""
    table: dict[int, tuple[float, float, float]] = {}
    with open(users_path, encoding="latin-1") as f:
        for line in f:
            uid, gender, age, occ, _zip = line.rstrip("\n").split("::")
            table[int(uid)] = (0.0 if gender == "M" else 1.0, float(age), float(occ))
    out = np.zeros((len(user_ids), 3), dtype=np.float32)
    for row, uid in enumerate(user_ids):
        out[row] = table.get(uid, (0.0, 0.0, 0.0))
    return out


def build_cooccurrence_pmi(train: Sequence[Sequence[int]], n_items: int,
                           window: int = 5) -> dict[tuple[int, int], float]:
    """Positive PMI over within-window co-occurrences in training
    sequences. Sparse dict keyed by ordered (min,max) item pair; absent
    pairs have PMI 0 by convention."""
    pair_counts: Counter = Counter()
    item_counts: Counter = Counter()
    total_pairs = 0
    for seq in train:
        for i, a in enumerate(seq):
            item_counts[a] += 1
            for b in seq[i + 1 : i + 1 + window]:
                key = (a, b) if a < b else (b, a)
                pair_counts[key] += 1
                total_pairs += 1
    pmi: dict[tuple[int, int], float] = {}
    if total_pairs == 0:
        return pmi
    n_events = sum(item_counts.values())
    for (a, b), c_ab in pair_counts.items():
        val = math.log((c_ab / total_pairs)
                       / ((item_counts[a] / n_events) * (item_counts[b] / n_events)))
        if val > 0:
            pmi[(a, b)] = val
    return pmi


class AuxFeatureContext:
    """Precomputed orthogonal features, consumed by the ranking dataset
    builder: user_context() once per user, features_for() per candidate."""

    def __init__(self, genre_mat: np.ndarray, years: np.ndarray,
                 user_demo: np.ndarray, pmi: dict):
        self.genre_mat = genre_mat
        self.years = years
        self.user_demo = user_demo
        self.pmi = pmi
        norms = np.linalg.norm(genre_mat, axis=1)
        self._genre_norm = np.where(norms > 0, norms, 1.0)

    def user_context(self, user_row: int, hist: Sequence[int]) -> dict:
        recent = list(hist[-20:])
        profile = self.genre_mat[recent].mean(axis=0) if recent else np.zeros(len(GENRES))
        pnorm = np.linalg.norm(profile)
        last_genres = self.genre_mat[hist[-1]] if hist else np.zeros(len(GENRES))
        return {
            "profile": profile, "pnorm": pnorm if pnorm > 0 else 1.0,
            "last_genres": last_genres,
            "last5": list(hist[-5:]),
            "demo": self.user_demo[user_row].tolist(),
        }

    def features_for(self, uctx: dict, item: int) -> list[float]:
        g = self.genre_mat[item]
        sim_profile = float(np.dot(g, uctx["profile"])
                            / (self._genre_norm[item] * uctx["pnorm"]))
        lg = uctx["last_genres"]
        inter = float(np.dot(g, lg))
        union = float(g.sum() + lg.sum() - inter)
        sim_last = inter / union if union > 0 else 0.0
        pmis = []
        for h in uctx["last5"]:
            key = (h, item) if h < item else (item, h)
            pmis.append(self.pmi.get(key, 0.0))
        cooc = float(np.mean(pmis)) if pmis else 0.0
        return [sim_profile, sim_last, float(self.years[item]), cooc] + uctx["demo"]
