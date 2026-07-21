"""Precompute serving artifacts so the API starts in seconds, not minutes.

Serving-time history for each user is their FULL sequence (train + valid
+ test): at serve time "now" is after everything we know about the user,
unlike evaluation where holding items out is the whole point. The
co-occurrence PMI (the slow part, minutes) and all feature tables are
computed once here and pickled.

    python scripts/build_artifacts.py --data data/ml-1m/ratings.dat \
        --movies data/ml-1m/movies.dat --users data/ml-1m/users.dat \
        --out artifacts/serving.pkl
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feedforge.data import build_sequences, load_movielens_1m  # noqa: E402
from feedforge.features import (  # noqa: E402
    build_cooccurrence_pmi,
    load_movie_features,
    load_user_features,
)
from feedforge.ranker import build_item_popularity  # noqa: E402

_TITLE_RE = re.compile(r"^(.*?)(?:\s+\((\d{4})\))?\s*$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--movies", required=True)
    p.add_argument("--users", required=True)
    p.add_argument("--out", default="artifacts/serving.pkl")
    args = p.parse_args()

    split = build_sequences(load_movielens_1m(args.data))

    full_history = [tr + [v, t] for tr, v, t in
                    zip(split.train, split.valid_target, split.test_target)]
    user_row = {uid: i for i, uid in enumerate(split.user_ids)}

    titles: dict[int, str] = {}
    with open(args.movies, encoding="latin-1") as f:
        for line in f:
            movie_id, title, _ = line.rstrip("\n").split("::")
            dense = split.item_id_map.get(int(movie_id))
            if dense is not None:
                titles[dense] = title

    dense_to_orig = {d: o for o, d in split.item_id_map.items()}

    genre_mat, years = load_movie_features(args.movies, split.item_id_map, split.n_items)
    user_demo = load_user_features(args.users, split.user_ids)
    print("building co-occurrence PMI (the slow part)...")
    pmi = build_cooccurrence_pmi(split.train, split.n_items)
    item_pop = build_item_popularity(split.train, split.n_items)

    artifacts = {
        "n_items": split.n_items,
        "mask_token": split.mask_token,
        "vocab_size": split.vocab_size,
        "full_history": full_history,
        "user_row": user_row,
        "titles": titles,
        "dense_to_orig": dense_to_orig,
        "genre_mat": genre_mat,
        "years": years,
        "user_demo": user_demo,
        "pmi": pmi,
        "item_pop": item_pop,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(full_history)} users, {split.n_items} items)")


if __name__ == "__main__":
    main()
