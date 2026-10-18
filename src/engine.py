"""
engine.py - the playnext recommendation engine.

Plain Python: no LLM, no MCP, no web framework. Everything else in the project
imports this module; it imports none of them.

Pipeline
--------
1. Retrieval    Cosine similarity between a taste profile and every eligible
                game (TF-IDF over tags, description, genres, developer/publisher).
                The top `n_candidates` move on.
2. Ranking      Blend similarity with review quality (Wilson lower bound of the
                positive-review rate) and popularity (log review count).
3. Re-ranking   MMR for diversity, plus per-developer and per-franchise caps.

Guardrails are hard filters in `Engine._eligible_mask`, applied before anything
is scored, so an ineligible game can never reach the results.

Similarity
----------
Each field gets its own L2-normalised TF-IDF block, scaled by sqrt(weight) and
concatenated, so the dot product of two games equals
    sum over fields of weight * cosine_in_that_field
A game with an empty field (e.g. no tags) simply scores 0 on that part.

Quick try (from the repo root):
    python src/engine.py --like "Hades" "Slay the Spire" --max-price 20
"""

from __future__ import annotations

import argparse
import re
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

PLATFORMS = ("windows", "mac", "linux")


# ---------------------------------------------------------------------------
# Configuration, inputs and outputs
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    # Similarity: relative weight of each field (normalised to sum to 1).
    w_tags: float = 0.5
    w_description: float = 0.3
    w_genres: float = 0.1
    w_credits: float = 0.1            # developers + publishers
    top_tags_per_game: int = 20       # keep only the most-voted tags
    desc_min_df: int = 2              # description vocabulary pruning
    desc_max_df: float = 0.5

    # Guardrails.
    min_reviews: int = 50
    adult_age_cutoff: int = 18        # required_age >= this counts as adult
    adult_tags: tuple = ("Sexual Content", "Nudity", "NSFW", "Hentai", "Mature")
    adult_tag_min_votes: int = 5      # ignore a stray vote or two

    # Ranking: blend of similarity, review quality and popularity.
    n_candidates: int = 100
    w_similarity: float = 0.7
    w_quality: float = 0.2
    w_popularity: float = 0.1

    # Re-ranking.
    mmr_lambda: float = 0.7           # 1.0 = pure relevance, lower = more diversity
    max_per_developer: int = 2
    max_per_franchise: int = 1


@dataclass
class Constraints:
    """Hard filters coming from the user's request ("co-op, under $20, short")."""
    max_price: float | None = None
    platforms: tuple = ()             # game must support all of these
    categories_any: tuple = ()        # e.g. ("co-op",): substring match on categories
    tags_any: tuple = ()              # game must have at least one of these tags
    exclude_tags: tuple = ()
    max_playtime_hours: float | None = None   # games with unknown playtime pass
    min_release_year: int | None = None
    allow_adult: bool = False


@dataclass
class Recommendation:
    app_id: int
    name: str
    score: float                      # final ranking score
    similarity: float                 # raw cosine-style similarity to the profile
    quality: float                    # Wilson lower bound of positive-review rate
    popularity: float                 # 0-1, log-scaled review count
    price: float
    genres: list
    developers: list
    top_tags: list                    # the game's own most-voted tags
    shared_tags: list                 # tags it shares with the user's liked games


@dataclass
class Result:
    items: list = field(default_factory=list)
    unmatched_ids: list = field(default_factory=list)   # liked ids not in the dataset
    n_eligible: int = 0
    timings_ms: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Soundtracks, demos and tools show up as separate "games" in Steam data.
_NOISE = re.compile(
    r"\b(soundtrack|ost|demo|playtest|dedicated server|art ?book|sdk)\b", re.I
)
_SUBTITLE = re.compile(r"\s*:\s*|\s+[-–—]\s+")
_TRAILING_NUMBER = re.compile(r"[\s:.\-]*\b(?:\d+|[ivx]{1,5})\s*$", re.I)


def franchise_key(name: str) -> str:
    """Crude franchise id: drop the subtitle and any trailing sequel number.

    "Left 4 Dead 2" -> "left 4 dead"; "Hades II" -> "hades";
    "Supipara - Chapter 1" -> "supipara".
    """
    lowered = name.lower().strip()
    base = _SUBTITLE.split(lowered, maxsplit=1)[0]
    base = _TRAILING_NUMBER.sub("", base).strip()
    return base or lowered


def wilson_lower_bound(positive, n, z: float = 1.96):
    """Lower bound of the 95% confidence interval on the positive-review rate.

    Unlike the raw ratio, 3/3 positive scores below 900/1000 positive, which is
    what we want: few reviews means we are less sure the game is good.
    """
    positive = np.asarray(positive, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = positive / n
        denom = 1 + z * z / n
        centre = p + z * z / (2 * n)
        margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
        bound = (centre - margin) / denom
    return np.where(n > 0, bound, 0.0)


def _tfidf_from_dicts(dicts):
    counts = DictVectorizer().fit_transform(dicts)
    if counts.shape[1] == 0:
        return counts.tocsr()
    return TfidfTransformer().fit_transform(counts)


def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = values.min(), values.max()
    if hi <= lo:
        return np.ones_like(values, dtype=float)
    return (values - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, games: pd.DataFrame, config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()
        self.games = games.reset_index(drop=True)
        g = self.games

        self._id_to_row = {int(a): i for i, a in enumerate(g["app_id"])}
        self._name_lower = g["name"].str.lower()
        self._categories = g["categories"].tolist()
        self._tags = g["tags"].tolist()
        self._tag_aliases = None    # lowercase tag -> real spellings; built on first use

        if "n_reviews" in g:
            self._n_reviews = g["n_reviews"].to_numpy(dtype=np.int64)
        else:
            self._n_reviews = (g["positive"] + g["negative"]).to_numpy(dtype=np.int64)

        self._build_static_columns()
        self._build_features()

    @classmethod
    def from_pickle(cls, path, config: EngineConfig | None = None) -> "Engine":
        return cls(pd.read_pickle(path), config)

    # -- build -------------------------------------------------------------

    def _build_static_columns(self):
        """Per-game values that never change between requests."""
        g, cfg = self.games, self.cfg
        names = g["name"].tolist()

        self._is_noise = np.array([bool(_NOISE.search(n)) for n in names])
        self._franchise = [franchise_key(n) for n in names]
        self._developers = g["developers"].tolist()

        # One-off tag processing: keep the top-N most-voted tags per game.
        self._top_tags = []       # tag names, most voted first
        self._tag_weights = []    # {tag: log1p(votes)} for the TF-IDF block
        for tag_votes in self._tags:
            ordered = sorted(tag_votes.items(), key=lambda kv: -kv[1])
            ordered = [(t, v) for t, v in ordered[: cfg.top_tags_per_game] if v > 0]
            self._top_tags.append([t for t, _ in ordered])
            self._tag_weights.append({t: float(np.log1p(v)) for t, v in ordered})

        self._has_adult_tag = np.array([
            any(votes.get(t, 0) >= cfg.adult_tag_min_votes for t in cfg.adult_tags)
            for votes in self._tags
        ])

        self._quality = wilson_lower_bound(g["positive"], self._n_reviews)
        self._popularity = np.minimum(
            np.log1p(self._n_reviews) / np.log1p(1_000_000), 1.0
        )

    def _build_features(self):
        g, cfg = self.games, self.cfg

        tags = _tfidf_from_dicts(self._tag_weights)
        genres = _tfidf_from_dicts([{x: 1.0 for x in gs} for gs in g["genres"]])
        credits = _tfidf_from_dicts([
            {name: 1.0 for name in devs + pubs}
            for devs, pubs in zip(g["developers"], g["publishers"])
        ])

        try:
            description = TfidfVectorizer(
                stop_words="english", sublinear_tf=True,
                min_df=cfg.desc_min_df, max_df=cfg.desc_max_df,
            ).fit_transform(g["short_description"].fillna(""))
        except ValueError:  # vocabulary pruned to nothing
            warnings.warn("Description vocabulary is empty; ignoring descriptions.")
            description = sp.csr_matrix((len(g), 0))

        blocks = [
            (tags, cfg.w_tags), (description, cfg.w_description),
            (genres, cfg.w_genres), (credits, cfg.w_credits),
        ]
        blocks = [(b, w) for b, w in blocks if w > 0 and b.shape[1] > 0]
        if not blocks:
            raise ValueError("No usable features: check the data and weights.")
        total = sum(w for _, w in blocks)
        self.X = sp.hstack(
            [b * np.sqrt(w / total) for b, w in blocks], format="csr", dtype=np.float32
        )

    # -- name lookup (for manual picks) -------------------------------------

    def find_games(self, query: str, limit: int = 5) -> list:
        """Find games by name: exact match first, then prefix, then substring.
        Within each tier, more-reviewed games first; soundtracks and demos last."""
        q = query.lower().strip()
        if not q:
            return []
        names = self._name_lower
        contains = names.str.contains(q, regex=False).to_numpy()
        hits = np.flatnonzero(contains)
        if len(hits) == 0:
            return []
        exact = (names == q).to_numpy()
        prefix = names.str.startswith(q).to_numpy()

        def sort_key(i):
            tier = 0 if exact[i] else 1 if prefix[i] else 2
            return (self._is_noise[i], tier, -self._n_reviews[i])

        out = []
        for i in sorted(hits, key=sort_key)[:limit]:
            released = self.games["release_date"].iat[i]
            out.append({
                "app_id": int(self.games["app_id"].iat[i]),
                "name": self.games["name"].iat[i],
                "release_year": None if pd.isna(released) else int(released.year),
                "n_reviews": int(self._n_reviews[i]),
            })
        return out

    # -- recommend -----------------------------------------------------------

    def recommend(self, liked, k: int = 10, owned=(), wishlist=(),
                  constraints: Constraints | None = None,
                  today: date | None = None) -> Result:
        """Recommend `k` games.

        liked        {app_id: weight} or [(app_id, weight), ...]. The taste profile.
        owned        app_ids never to recommend (the user's library).
        wishlist     app_ids never to recommend (already on their radar).
        constraints  hard filters from the user's request.
        today        override "today" (used by tests for the unreleased filter).
        """
        cons = constraints or Constraints()
        today = today or date.today()
        timings = {}

        # 1. Taste profile
        t0 = time.perf_counter()
        pairs = list(liked.items()) if isinstance(liked, dict) else list(liked)
        profile, seed_rows, seed_weights, unmatched = self._build_profile(pairs)
        timings["profile"] = (time.perf_counter() - t0) * 1000

        # 2. Retrieval: eligible games, then the top-N most similar
        t0 = time.perf_counter()
        exclude = set(seed_rows)
        for app_id in (*owned, *wishlist):
            row = self._id_to_row.get(int(app_id))
            if row is not None:
                exclude.add(row)
        mask = self._eligible_mask(cons, exclude, today)
        n_eligible = int(mask.sum())
        result = Result(unmatched_ids=unmatched, n_eligible=n_eligible)

        n = min(self.cfg.n_candidates, n_eligible)
        if n == 0:
            result.timings_ms = timings
            return result
        sims = (self.X @ profile.T).toarray().ravel()
        masked = np.where(mask, sims, -np.inf)
        idx = np.argpartition(-masked, n - 1)[:n]
        idx = idx[np.argsort(-masked[idx])]
        timings["retrieval"] = (time.perf_counter() - t0) * 1000

        # 3. Ranking: similarity + review quality + popularity
        t0 = time.perf_counter()
        cfg = self.cfg
        weight_sum = cfg.w_similarity + cfg.w_quality + cfg.w_popularity
        score = (
            cfg.w_similarity * _minmax(sims[idx])
            + cfg.w_quality * self._quality[idx]
            + cfg.w_popularity * self._popularity[idx]
        ) / weight_sum
        timings["ranking"] = (time.perf_counter() - t0) * 1000

        # 4. Re-ranking: MMR diversity + developer/franchise caps
        t0 = time.perf_counter()
        picked = self._mmr(idx, score, k)
        timings["rerank"] = (time.perf_counter() - t0) * 1000

        # 5. Package the output
        seed_tags = Counter()
        for row, w in zip(seed_rows, seed_weights):
            for tag in self._top_tags[row]:
                seed_tags[tag] += w
        for j in picked:
            i = int(idx[j])
            top_tags = self._top_tags[i]
            result.items.append(Recommendation(
                app_id=int(self.games["app_id"].iat[i]),
                name=self.games["name"].iat[i],
                score=float(score[j]),
                similarity=float(sims[i]),
                quality=float(self._quality[i]),
                popularity=float(self._popularity[i]),
                price=float(self.games["price"].iat[i]),
                genres=list(self.games["genres"].iat[i]),
                developers=list(self._developers[i]),
                top_tags=top_tags[:5],
                shared_tags=[t for t in top_tags if t in seed_tags][:3],
            ))
        result.timings_ms = timings
        return result

    # -- internals -------------------------------------------------------------

    def _build_profile(self, pairs):
        """Weighted average of the liked games' vectors, scaled to unit length."""
        weights_by_row, unmatched = {}, []
        for app_id, weight in pairs:
            row = self._id_to_row.get(int(app_id))
            if row is None:
                unmatched.append(int(app_id))
            elif weight > 0:
                weights_by_row[row] = weights_by_row.get(row, 0.0) + float(weight)
        if not weights_by_row:
            raise ValueError(
                "No usable liked games: none were found in the dataset, or all "
                f"had zero weight. Unmatched ids: {unmatched}"
            )
        rows = list(weights_by_row)
        weights = np.array([weights_by_row[r] for r in rows])
        mix = sp.csr_matrix((weights / weights.sum())[None, :])
        profile = mix @ self.X[rows]
        norm = float(np.sqrt(profile.multiply(profile).sum()))
        if norm == 0:
            raise ValueError("The liked games have no usable features.")
        return profile / norm, rows, weights, unmatched

    def _eligible_mask(self, c: Constraints, exclude_rows: set, today: date):
        """Guardrails and request constraints as one boolean mask over all games."""
        g, cfg = self.games, self.cfg

        mask = self._n_reviews >= cfg.min_reviews                    # guardrail 5
        mask &= ~self._is_noise                                      # soundtracks, demos
        release = g["release_date"]
        mask &= (release.isna() | (release <= pd.Timestamp(today))).to_numpy()  # guardrail 6
        if not c.allow_adult:                                        # guardrail 7
            mask &= (g["required_age"] < cfg.adult_age_cutoff).to_numpy()
            mask &= ~self._has_adult_tag
        if exclude_rows:                                             # guardrail 1
            mask[list(exclude_rows)] = False

        # Hard constraints from the request (guardrail 3).
        if c.max_price is not None:
            mask &= (g["price"] <= c.max_price).to_numpy()
        for platform in c.platforms:
            if platform not in PLATFORMS:
                raise ValueError(f"Unknown platform {platform!r}; use one of {PLATFORMS}")
            mask &= g[platform].to_numpy(dtype=bool)
        if c.min_release_year is not None:
            mask &= (release.dt.year >= c.min_release_year).to_numpy()
        if c.max_playtime_hours is not None:
            playtime = g["average_playtime_forever"]      # minutes; 0 means unknown
            mask &= ((playtime == 0) | (playtime <= c.max_playtime_hours * 60)).to_numpy()

        # List-valued constraints: only checked on games that survived so far.
        if c.categories_any or c.tags_any or c.exclude_tags:
            wanted_cats = [s.lower() for s in c.categories_any]
            wanted_tags = self._resolve_tags(c.tags_any)
            banned_tags = self._resolve_tags(c.exclude_tags)
            for i in np.flatnonzero(mask):
                votes = self._tags[i]
                keep = True
                if wanted_cats:
                    cats = [x.lower() for x in self._categories[i]]
                    keep = any(w in cat for w in wanted_cats for cat in cats)
                if keep and c.tags_any:
                    keep = any(votes.get(t, 0) > 0 for t in wanted_tags)
                if keep and c.exclude_tags:
                    keep = not any(votes.get(t, 0) > 0 for t in banned_tags)
                mask[i] = keep
        return mask

    def _resolve_tags(self, names) -> list:
        """Map case-insensitive tag names to the spellings used in the data, so
        each game needs a couple of dict lookups instead of lowercasing every tag."""
        if self._tag_aliases is None:
            aliases = {}
            for votes in self._tags:
                for tag in votes:
                    aliases.setdefault(tag.lower(), set()).add(tag)
            self._tag_aliases = aliases
        return [real for n in names for real in self._tag_aliases.get(n.lower(), ())]

    def _mmr(self, idx: np.ndarray, score: np.ndarray, k: int) -> list:
        """Maximal Marginal Relevance with developer and franchise caps.

        Repeatedly picks the candidate maximising
            lambda * relevance - (1 - lambda) * (similarity to what is already picked)
        Returns positions into `idx`, in final ranking order.
        """
        cfg = self.cfg
        cand = self.X[idx]
        sim = (cand @ cand.T).toarray()
        norms = np.sqrt(np.clip(np.diag(sim), 0, None))
        norms[norms == 0] = 1.0
        sim = sim / np.outer(norms, norms)            # proper cosine between candidates

        relevance = _minmax(score)
        dev_count, franchise_count = Counter(), Counter()
        picked, remaining = [], list(range(len(idx)))

        def blocked(j):
            i = int(idx[j])
            if franchise_count[self._franchise[i]] >= cfg.max_per_franchise:
                return True
            return any(dev_count[d] >= cfg.max_per_developer for d in self._developers[i])

        while remaining and len(picked) < k:
            remaining = [j for j in remaining if not blocked(j)]   # caps only tighten
            if not remaining:
                break
            best, best_value = None, -np.inf
            for j in remaining:
                redundancy = sim[j, picked].max() if picked else 0.0
                value = cfg.mmr_lambda * relevance[j] - (1 - cfg.mmr_lambda) * redundancy
                if value > best_value:
                    best, best_value = j, value
            picked.append(best)
            remaining.remove(best)
            i = int(idx[best])
            franchise_count[self._franchise[i]] += 1
            for d in self._developers[i]:
                dev_count[d] += 1
        return picked


# ---------------------------------------------------------------------------
# Command-line demo
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(description="Try the playnext engine from the terminal.")
    parser.add_argument("--data", type=Path, default=Path("data/games_slim.pkl"))
    parser.add_argument("--like", nargs="+", required=True, help="game names you like")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-price", type=float)
    parser.add_argument("--platform", action="append", choices=PLATFORMS)
    parser.add_argument("--category", action="append", help='e.g. "co-op"')
    args = parser.parse_args()

    print("Building features ...")
    t0 = time.perf_counter()
    engine = Engine.from_pickle(args.data)
    print(f"Ready in {time.perf_counter() - t0:.1f}s ({len(engine.games):,} games)\n")

    liked = []
    for name in args.like:
        hits = engine.find_games(name, limit=1)
        if not hits:
            print(f"  ! no match for {name!r}, skipping")
            continue
        print(f"  {name!r} -> {hits[0]['name']} (id {hits[0]['app_id']})")
        liked.append((hits[0]["app_id"], 1.0))

    constraints = Constraints(
        max_price=args.max_price,
        platforms=tuple(args.platform or ()),
        categories_any=tuple(args.category or ()),
    )
    result = engine.recommend(liked, k=args.k, constraints=constraints)

    print(f"\n{result.n_eligible:,} eligible games\n")
    for rank, r in enumerate(result.items, 1):
        print(f"{rank:>2}. {r.name}  (id {r.app_id}, ${r.price:.2f})")
        print(f"     score {r.score:.3f} | sim {r.similarity:.3f} | quality {r.quality:.2f}")
        print(f"     tags: {', '.join(r.top_tags)}")
        if r.shared_tags:
            print(f"     shares with your picks: {', '.join(r.shared_tags)}")
    print("\nTimings (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in result.timings_ms.items()))


if __name__ == "__main__":
    _main()