"""
engine.py - the playnext recommendation engine.

Plain Python: no LLM, no MCP, no web framework. Everything else in the project
imports this module; it imports none of them.

Pipeline
--------
1. Retrieval    Cosine similarity between every eligible game and EACH liked game
                (TF-IDF over tags, description, genres, developer/publisher),
                scaled so every liked game counts equally, then combined. The top
                `n_candidates`, plus each liked game's own best matches, move on.
2. Ranking      Blend similarity with review quality (Wilson lower bound of the
                positive-review rate) and popularity (log review count).
3. Re-ranking   MMR for diversity, per-developer and per-franchise caps, and a
                nudge so each liked game gets its share of the final list.

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
from typing import NamedTuple

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
    # Tags that say nothing about taste. Dropped from similarity and from the tags
    # shown; the raw votes still serve tag constraints (e.g. "free to play only").
    tag_stoplist: tuple = ("Early Access", "Singleplayer", "Free to Play")

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

    # Several liked games: each is scored on its own, then combined.
    max_seeds: int = 30               # a big library is cut to its heaviest games
    seed_blend: float = 0.3           # 0 = close to any one seed is enough; 1 = must fit all
    per_seed_candidates: int = 10     # each seed always contributes its own best matches

    # Re-ranking.
    mmr_lambda: float = 0.7           # 1.0 = pure relevance, lower = more diversity
    max_per_developer: int = 2
    max_per_franchise: int = 2
    seed_balance: float = 0.2         # nudge toward giving every liked game its share

    # Games suggested from outside (e.g. found on the web): extra score for a game
    # that passes the guardrails, 1 - 0.5**mentions of it, times this weight.
    w_candidates: float = 0.3


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
    allow_vr_only: bool = False       # VR-only games are hidden unless asked for


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
    shared_tags: list                 # tags it shares with the liked game it matches best
    because_of: str = ""              # name of the liked game it matches best ("" if unrelated)
    mentions: int = 0                 # how many outside sources suggested it (see `candidates`)


@dataclass
class Result:
    items: list = field(default_factory=list)
    unmatched_ids: list = field(default_factory=list)   # liked ids not in the dataset
    n_eligible: int = 0
    timings_ms: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)        # e.g. unknown tag_boosts names


class _Seeds(NamedTuple):
    rows: list                        # liked games used for scoring, heaviest first
    weights: np.ndarray
    norms: np.ndarray                 # length of each seed's feature vector
    excluded: list                    # every matched liked game: never recommended
    unmatched: list                   # liked ids not in the dataset


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


def _top_indices(values: np.ndarray, n: int) -> np.ndarray:
    """Positions of the `n` largest values, best first."""
    n = min(n, len(values))
    top = np.argpartition(-values, n - 1)[:n]
    return top[np.argsort(-values[top])]


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
        stoplist = {t.lower() for t in cfg.tag_stoplist}
        self._top_tags = []       # tag names, most voted first
        self._tag_weights = []    # {tag: log1p(votes)} for the TF-IDF block
        for tag_votes in self._tags:
            kept = [(t, v) for t, v in tag_votes.items() if t.lower() not in stoplist]
            kept.sort(key=lambda kv: -kv[1])
            ordered = [(t, v) for t, v in kept[: cfg.top_tags_per_game] if v > 0]
            self._top_tags.append([t for t, _ in ordered])
            self._tag_weights.append({t: float(np.log1p(v)) for t, v in ordered})

        self._has_adult_tag = np.array([
            any(votes.get(t, 0) >= cfg.adult_tag_min_votes for t in cfg.adult_tags)
            for votes in self._tags
        ])
        self._is_vr_only = np.array([
            self._looks_vr_only(cats, votes)
            for cats, votes in zip(self._categories, self._tags)
        ])

        self._quality = wilson_lower_bound(g["positive"], self._n_reviews)
        self._popularity = np.minimum(
            np.log1p(self._n_reviews) / np.log1p(1_000_000), 1.0
        )

    @staticmethod
    def _looks_vr_only(categories, tag_votes) -> bool:
        """Steam's own "VR Only" category, or a clear VR tag on a game that does
        not claim to be merely VR-supported (the categories can be incomplete)."""
        lowered = [c.lower() for c in categories]
        if "vr only" in lowered:
            return True
        return tag_votes.get("VR", 0) >= 5 and not any("vr support" in c for c in lowered)

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
                  tag_boosts: dict | None = None,
                  candidates: dict | None = None,
                  today: date | None = None) -> Result:
        """Recommend `k` games.

        liked        {app_id: weight} or [(app_id, weight), ...]. The taste seeds.
        owned        app_ids never to recommend (the user's library).
        wishlist     app_ids never to recommend (already on their radar).
        constraints  hard filters from the user's request.
        tag_boosts   soft preferences, e.g. {"Great Soundtrack": 0.2}. A game that
                     strongly carries the tag gains up to that much score (negative
                     values subtract). Only reorders candidates; never filters.
        candidates   {app_id: mentions}: games suggested from outside, e.g. found by a
                     web search, with how many sources named each. They join the pool
                     and get a score bonus, but the guardrails and constraints still
                     apply: an ineligible candidate is dropped, never forced in.
        today        override "today" (used by tests for the unreleased filter).
        """
        cons = constraints or Constraints()
        cfg = self.cfg
        today = today or date.today()
        boosts, boost_warnings = self._check_boosts(tag_boosts)
        timings = {}

        # 1. Seeds: the liked games, strongest first
        t0 = time.perf_counter()
        pairs = list(liked.items()) if isinstance(liked, dict) else list(liked)
        seeds = self._resolve_seeds(pairs)
        timings["profile"] = (time.perf_counter() - t0) * 1000

        # 2. Retrieval: eligible games, scored against each seed separately
        t0 = time.perf_counter()
        exclude = set(seeds.excluded)
        for app_id in (*owned, *wishlist):
            row = self._id_to_row.get(int(app_id))
            if row is not None:
                exclude.add(row)
        mask = self._eligible_mask(cons, exclude, today)
        n_eligible = int(mask.sum())
        result = Result(unmatched_ids=seeds.unmatched, n_eligible=n_eligible,
                        warnings=boost_warnings)

        n = min(cfg.n_candidates, n_eligible)
        if n == 0:
            result.timings_ms = timings
            return result

        seed_matrix = self.X[seeds.rows]
        raw = (self.X @ seed_matrix.T).toarray()                  # games x seeds
        raw = raw / np.where(seeds.norms > 0, seeds.norms, 1.0)   # cosine to each seed
        # Put the seeds on one scale: each seed's best eligible match scores 1.0.
        # Otherwise a seed with a dense neighbourhood (many close matches) would
        # crowd out a niche seed whose closest matches are less similar.
        best_per_seed = np.where(mask[:, None], raw, 0.0).max(axis=0)
        scaled = raw / np.where(best_per_seed > 0, best_per_seed, 1.0)
        share = seeds.weights / seeds.weights.sum()
        weighted = scaled * (seeds.weights / seeds.weights.max())
        # A game is relevant if it is close to ANY seed (max), with a bonus for
        # fitting several at once (weighted mean).
        combined = ((1 - cfg.seed_blend) * weighted.max(axis=1)
                    + cfg.seed_blend * (scaled @ share))
        nearest_seed = weighted.argmax(axis=1)

        # Candidate pool: the best overall, plus each seed's own best matches.
        pool = set(_top_indices(np.where(mask, combined, -np.inf), n).tolist())
        if len(seeds.rows) > 1:
            per_seed = min(cfg.per_seed_candidates, n_eligible)
            for j in range(len(seeds.rows)):
                column = np.where(mask, weighted[:, j], -np.inf)
                pool.update(int(i) for i in _top_indices(column, per_seed)
                            if weighted[i, j] > 0)
        mentions = {}
        for app_id, count in (candidates or {}).items():
            row = self._id_to_row.get(int(app_id))
            if row is not None and mask[row] and count > 0:
                mentions[row] = float(count)
        if candidates and len(mentions) < len(candidates):
            result.warnings.append(
                f"{len(candidates) - len(mentions)} of {len(candidates)} candidate games were "
                "ignored: not in the dataset, already owned or liked, or filtered out.")
        pool.update(mentions)
        idx = np.array(sorted(pool, key=lambda i: -combined[i]), dtype=np.int64)
        timings["retrieval"] = (time.perf_counter() - t0) * 1000

        # 3. Ranking: similarity + review quality + popularity (+ tag boosts)
        t0 = time.perf_counter()
        weight_sum = cfg.w_similarity + cfg.w_quality + cfg.w_popularity
        score = (
            cfg.w_similarity * _minmax(combined[idx])
            + cfg.w_quality * self._quality[idx]
            + cfg.w_popularity * self._popularity[idx]
        ) / weight_sum
        score = score + self._boost_bonus(idx, boosts)
        if mentions:
            counts = np.array([mentions.get(int(i), 0.0) for i in idx])
            score = score + cfg.w_candidates * (1 - 0.5 ** counts)    # 0 when never mentioned
        timings["ranking"] = (time.perf_counter() - t0) * 1000

        # 4. Re-ranking: MMR diversity + developer/franchise caps + fair seed shares
        t0 = time.perf_counter()
        quotas = np.maximum(1.0, k * share)
        picked = self._mmr(idx, score, k, nearest_seed[idx], quotas)
        timings["rerank"] = (time.perf_counter() - t0) * 1000

        # 5. Package the output
        for j in picked:
            i = int(idx[j])
            s = int(nearest_seed[i])
            seed_row = seeds.rows[s]
            seed_tag_set = set(self._top_tags[seed_row])
            top_tags = self._top_tags[i]
            related = raw[i, s] > 0      # a web-suggested game can share nothing with the seeds
            result.items.append(Recommendation(
                app_id=int(self.games["app_id"].iat[i]),
                name=self.games["name"].iat[i],
                score=float(score[j]),
                similarity=float(raw[i, s]),
                quality=float(self._quality[i]),
                popularity=float(self._popularity[i]),
                price=float(self.games["price"].iat[i]),
                genres=list(self.games["genres"].iat[i]),
                developers=list(self._developers[i]),
                top_tags=top_tags[:5],
                shared_tags=[t for t in top_tags if t in seed_tag_set][:3] if related else [],
                because_of=self.games["name"].iat[seed_row] if related else "",
                mentions=int(mentions.get(i, 0)),
            ))
        result.timings_ms = timings
        return result

    # -- internals -------------------------------------------------------------

    def _resolve_seeds(self, pairs) -> "_Seeds":
        """Match liked app_ids to rows. Keeps the `max_seeds` heaviest for scoring,
        but every matched game stays excluded from the results."""
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
        ordered = sorted(weights_by_row, key=lambda r: -weights_by_row[r])
        rows = ordered[: self.cfg.max_seeds]
        matrix = self.X[rows]
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        if not norms.any():
            raise ValueError("The liked games have no usable features.")
        weights = np.array([weights_by_row[r] for r in rows])
        return _Seeds(rows, weights, norms, ordered, unmatched)

    def _check_boosts(self, tag_boosts):
        """Validate tag_boosts and map the tag names to the data's spellings."""
        resolved, warnings_ = [], []
        for name, amount in (tag_boosts or {}).items():
            amount = float(amount)
            if not np.isfinite(amount) or abs(amount) > 1:
                raise ValueError(f"tag_boosts[{name!r}] must be between -1 and 1, got {amount}")
            real = self._resolve_tags([name])
            if real:
                resolved.append((real, amount))
            else:
                warnings_.append(f"Unknown tag {name!r} in tag_boosts; ignored.")
        return resolved, warnings_

    def _boost_bonus(self, rows, resolved) -> np.ndarray:
        """Score added to each candidate: amount x how prominent the tag is for
        that game (its votes relative to the game's most-voted tag, 0 to 1)."""
        bonus = np.zeros(len(rows))
        if not resolved:
            return bonus
        for pos, i in enumerate(rows):
            votes = self._tags[int(i)]
            top = max(votes.values(), default=0)
            if top <= 0:
                continue
            for real, amount in resolved:
                strength = max((votes.get(t, 0) for t in real), default=0) / top
                bonus[pos] += amount * min(strength, 1.0)
        return bonus

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
        if not c.allow_vr_only:                                      # hardware most people lack
            mask &= ~self._is_vr_only
        if exclude_rows:                                           # guardrail 1
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

    def _mmr(self, idx: np.ndarray, score: np.ndarray, k: int,
             seed_of: np.ndarray | None = None, quotas: np.ndarray | None = None) -> list:
        """Maximal Marginal Relevance with developer and franchise caps.

        Repeatedly picks the candidate maximising
            lambda * relevance - (1 - lambda) * (similarity to what is already picked)
        and, when several liked games are involved, a small penalty on candidates
        whose nearest liked game already has its share of the picks so far.
        Returns positions into `idx`, in final ranking order.
        """
        cfg = self.cfg
        cand = self.X[idx]
        sim = (cand @ cand.T).toarray()
        norms = np.sqrt(np.clip(np.diag(sim), 0, None))
        norms[norms == 0] = 1.0
        sim = sim / np.outer(norms, norms)            # proper cosine between candidates

        relevance = _minmax(score)
        dev_count, franchise_count, seed_picks = Counter(), Counter(), Counter()
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
                if seed_of is not None:
                    s = int(seed_of[j])
                    value -= cfg.seed_balance * seed_picks[s] / quotas[s]
                if value > best_value:
                    best, best_value = j, value
            picked.append(best)
            remaining.remove(best)
            if seed_of is not None:
                seed_picks[int(seed_of[best])] += 1
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
    parser.add_argument("--boost", action="append", metavar="TAG=AMOUNT",
                        help='soft preference, e.g. "Great Soundtrack=0.2" (-1 to 1)')
    parser.add_argument("--allow-vr", action="store_true", help="include VR-only games")
    args = parser.parse_args()
    boosts = {}
    for item in args.boost or []:
        tag, _, amount = item.rpartition("=")
        if not tag:
            parser.error(f"--boost expects TAG=AMOUNT, got {item!r}")
        boosts[tag] = float(amount)

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
        allow_vr_only=args.allow_vr,
    )
    result = engine.recommend(liked, k=args.k, constraints=constraints, tag_boosts=boosts)

    print(f"\n{result.n_eligible:,} eligible games")
    for warning in result.warnings:
        print(f"  ! {warning}")
    print()
    for rank, r in enumerate(result.items, 1):
        print(f"{rank:>2}. {r.name}  (id {r.app_id}, ${r.price:.2f})")
        print(f"     score {r.score:.3f} | sim {r.similarity:.3f} | quality {r.quality:.2f}")
        print(f"     tags: {', '.join(r.top_tags)}")
        if r.shared_tags:
            print(f"     like {r.because_of}: shares {', '.join(r.shared_tags)}")
        else:
            print(f"     closest to {r.because_of}")
    print("\nTimings (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in result.timings_ms.items()))


if __name__ == "__main__":
    _main()