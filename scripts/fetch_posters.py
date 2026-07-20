"""Fetch movie posters for ML-1M items from TMDB.

ML-1M ships no image links, so items are matched to TMDB via the search
API using title + year parsed from movies.dat (titles look like
"Toy Story (1995)"; ML-1M formats some as "Matrix, The (1999)", which is
normalized back to "The Matrix" before searching). Expect a 5-10% miss
rate from title mismatches and obscure entries; misses are logged to
posters/misses.json and downstream code treats those items as having no
content signal rather than fabricating one.

Usage:
    export TMDB_API_KEY=...   # free key from themoviedb.org
    python scripts/fetch_posters.py --movies data/ml-1m/movies.dat --out data/posters
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
IMG_BASE = "https://image.tmdb.org/t/p/w342"  # 342px wide: plenty for ViT-224

TITLE_RE = re.compile(r"^(.*)\s+\((\d{4})\)\s*$")
ARTICLE_RE = re.compile(r"^(.*),\s*(The|A|An|Le|La|Les|Der|Die|Das|Il|El)$", re.IGNORECASE)


def normalize_title(raw: str) -> tuple[str, int | None]:
    """'Matrix, The (1999)' -> ('The Matrix', 1999)"""
    m = TITLE_RE.match(raw)
    title, year = (m.group(1), int(m.group(2))) if m else (raw, None)
    am = ARTICLE_RE.match(title)
    if am:
        title = f"{am.group(2)} {am.group(1)}"
    return title.strip(), year


def search_poster_path(session: requests.Session, api_key: str, title: str, year: int | None) -> str | None:
    params = {"api_key": api_key, "query": title}
    if year:
        params["year"] = year
    r = session.get(SEARCH_URL, params=params, timeout=15)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results and year:
        # Retry without the year: ML-1M years occasionally disagree with TMDB
        params.pop("year")
        r = session.get(SEARCH_URL, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
    for res in results:
        if res.get("poster_path"):
            return res["poster_path"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--movies", required=True, help="path to ml-1m/movies.dat")
    parser.add_argument("--out", default="data/posters")
    parser.add_argument("--sleep", type=float, default=0.05,
                        help="pause between requests; TMDB allows ~50 req/s, stay polite")
    args = parser.parse_args()

    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("Set TMDB_API_KEY (free key: themoviedb.org -> Settings -> API)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    movies = []
    with open(args.movies, encoding="latin-1") as f:
        for line in f:
            movie_id, title, _genres = line.rstrip("\n").split("::")
            movies.append((int(movie_id), title))
    print(f"{len(movies)} movies in {args.movies}")

    session = requests.Session()
    misses: list[dict] = []
    fetched = skipped = 0

    for i, (movie_id, raw_title) in enumerate(movies, 1):
        dest = out_dir / f"{movie_id}.jpg"
        if dest.exists():
            skipped += 1
            continue
        title, year = normalize_title(raw_title)
        try:
            poster_path = search_poster_path(session, api_key, title, year)
            if poster_path is None:
                misses.append({"movie_id": movie_id, "title": raw_title})
            else:
                img = session.get(IMG_BASE + poster_path, timeout=20)
                img.raise_for_status()
                dest.write_bytes(img.content)
                fetched += 1
        except requests.RequestException as e:
            misses.append({"movie_id": movie_id, "title": raw_title, "error": str(e)})
        if i % 200 == 0:
            print(f"  {i}/{len(movies)}  fetched={fetched} skipped={skipped} missed={len(misses)}")
        time.sleep(args.sleep)

    (out_dir / "misses.json").write_text(json.dumps(misses, indent=2))
    print(f"done: fetched={fetched} skipped={skipped} missed={len(misses)} "
          f"({len(misses) / len(movies):.1%} miss rate)")
    print(f"misses logged to {out_dir / 'misses.json'}")


if __name__ == "__main__":
    main()
