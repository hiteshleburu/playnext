"""Tests for the recommendation engine, on a small synthetic catalogue.

The catalogue has three clear clusters (roguelike deckbuilders, farming sims,
horror) so "did it recommend the right kind of game" is easy to assert. Each
guardrail from the README has its own test.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engine import Constraints, Engine, EngineConfig, franchise_key  # noqa: E402

TODAY = date(2026, 10, 18)

ROGUE_DESC = "Build a deck of cards and fight through a roguelike dungeon run with new relics"
FARM_DESC = "Plant crops and raise animals on your farm as the seasons change in a cozy village"
HORROR_DESC = "Survive a haunted house at night while a terrifying monster hunts you in the dark"

SEED_TAGS = {"Roguelike": 100, "Card Game": 80, "Deckbuilding": 60}
ROGUE_TAGS = {"Roguelike": 90, "Card Game": 70}
FARM_TAGS = {"Farming Sim": 90, "Cozy": 70}
HORROR_TAGS = {"Horror": 90, "Atmospheric": 70}

ROGUE_NAMES = ["Ember Crown", "Hollow Spire", "Iron Relic", "Ashen Vault",
               "Gilded Maze", "Cinder Hand", "Frost Ledger", "Obsidian Draw"]
FARM_NAMES = ["Meadow Haven", "Willow Creek", "Sunny Orchard", "Clover Barn",
              "Maple Acres", "Honey Field", "Pebble Farm", "Daisy Valley"]
HORROR_NAMES = ["Silent Manor", "Crimson Hall", "Whisper Cellar", "Pale Corridor",
                "Dread Asylum", "Gloom Cabin", "Hollow Mist", "Raven Crypt"]


def make_game(app_id, name, tags, desc, **overrides):
    positive = overrides.pop("positive", 450)
    negative = overrides.pop("negative", 50)
    row = {
        "app_id": app_id, "name": name,
        "release_date": pd.Timestamp("2020-01-01"),
        "required_age": 0, "price": 10.0, "short_description": desc,
        "genres": ["Indie"], "categories": ["Single-player"], "tags": tags,
        "developers": [f"Dev {app_id}"], "publishers": [f"Pub {app_id}"],
        "positive": positive, "negative": negative, "recommendations": 100,
        "average_playtime_forever": 600, "estimated_owners": "0 - 20000",
        "owners_low": 0, "owners_high": 20000,
        "windows": True, "mac": True, "linux": True,
    }
    row.update(overrides)
    row["n_reviews"] = row["positive"] + row["negative"]
    row["review_ratio"] = row["positive"] / row["n_reviews"] if row["n_reviews"] else None
    return row


def base_rows():
    rows = [make_game(1, "Seed Deckbuilder", SEED_TAGS, ROGUE_DESC, genres=["Indie", "Strategy"])]
    for offset, name in enumerate(ROGUE_NAMES):
        rows.append(make_game(2 + offset, name, ROGUE_TAGS, ROGUE_DESC, genres=["Indie", "Strategy"]))
    for offset, name in enumerate(FARM_NAMES):
        rows.append(make_game(11 + offset, name, FARM_TAGS, FARM_DESC, genres=["Simulation"]))
    for offset, name in enumerate(HORROR_NAMES):
        rows.append(make_game(21 + offset, name, HORROR_TAGS, HORROR_DESC, genres=["Horror"]))
    return rows


def seed_twin(app_id, name, **overrides):
    """A game with exactly the seed's features, so it is the best possible match."""
    return make_game(app_id, name, SEED_TAGS, ROGUE_DESC, genres=["Indie", "Strategy"], **overrides)


def build_engine(extra=(), **config):
    config.setdefault("desc_max_df", 1.0)   # tiny corpus: don't prune common words
    return Engine(pd.DataFrame(base_rows() + list(extra)), EngineConfig(**config))


def recommend_ids(engine, k=100, **kwargs):
    kwargs.setdefault("today", TODAY)
    result = engine.recommend([(1, 1.0)], k=k, **kwargs)
    return [r.app_id for r in result.items]


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------

def test_recommends_the_same_kind_of_game():
    result = build_engine().recommend([(1, 1.0)], k=5, today=TODAY)
    assert len(result.items) == 5
    assert all(2 <= r.app_id <= 9 for r in result.items)   # all roguelike deckbuilders


def test_shared_tags_explain_the_match():
    result = build_engine().recommend([(1, 1.0)], k=3, today=TODAY)
    assert set(result.items[0].shared_tags) == {"Roguelike", "Card Game"}


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def test_never_returns_liked_owned_or_wishlisted():
    ids = recommend_ids(build_engine(), owned=[2, 3], wishlist=[4])
    assert not {1, 2, 3, 4} & set(ids)


def test_min_reviews_filter():
    extra = [make_game(40, "Barely Reviewed", ROGUE_TAGS, ROGUE_DESC, positive=9, negative=1)]
    assert 40 not in recommend_ids(build_engine(extra))
    assert 40 in recommend_ids(build_engine(extra, min_reviews=5))


def test_unreleased_games_excluded():
    extra = [make_game(41, "Future Hit", ROGUE_TAGS, ROGUE_DESC,
                       release_date=pd.Timestamp("2099-01-01"))]
    assert 41 not in recommend_ids(build_engine(extra))


def test_adult_content_filtered_by_tag_and_by_age():
    extra = [
        make_game(42, "Spicy One", {**ROGUE_TAGS, "Nudity": 20}, ROGUE_DESC),
        make_game(43, "Mature Rated", ROGUE_TAGS, ROGUE_DESC, required_age=18),
        make_game(44, "Barely Tagged", {**ROGUE_TAGS, "Nudity": 2}, ROGUE_DESC),
    ]
    engine = build_engine(extra)
    ids = recommend_ids(engine)
    assert 42 not in ids and 43 not in ids
    assert 44 in ids                                   # 2 stray votes is not enough
    opted_in = recommend_ids(engine, constraints=Constraints(allow_adult=True))
    assert 42 in opted_in and 43 in opted_in


def test_soundtracks_and_demos_excluded_but_lookalike_names_are_not():
    extra = [
        make_game(50, "Seed Deckbuilder Soundtrack", ROGUE_TAGS, ROGUE_DESC),
        make_game(51, "Demolition Deck", ROGUE_TAGS, ROGUE_DESC),
    ]
    ids = recommend_ids(build_engine(extra))
    assert 50 not in ids
    assert 51 in ids


def test_unmatched_ids_are_reported_not_fatal():
    result = build_engine().recommend([(1, 1.0), (999999, 1.0)], k=3, today=TODAY)
    assert result.unmatched_ids == [999999]
    assert len(result.items) == 3


def test_no_usable_liked_games_raises():
    engine = build_engine()
    with pytest.raises(ValueError):
        engine.recommend([(999999, 1.0)], today=TODAY)
    with pytest.raises(ValueError):
        engine.recommend([(1, 0.0)], today=TODAY)


# ---------------------------------------------------------------------------
# Request constraints
# ---------------------------------------------------------------------------

def test_max_price():
    extra = [make_game(45, "Pricey Deck", ROGUE_TAGS, ROGUE_DESC, price=60.0)]
    engine = build_engine(extra)
    assert 45 in recommend_ids(engine)
    assert 45 not in recommend_ids(engine, constraints=Constraints(max_price=20))


def test_platform():
    extra = [make_game(46, "Windows Only Deck", ROGUE_TAGS, ROGUE_DESC, mac=False, linux=False)]
    ids = recommend_ids(build_engine(extra), constraints=Constraints(platforms=("mac",)))
    assert 46 not in ids and len(ids) > 0


def test_category_match_is_case_insensitive_substring():
    extra = [
        make_game(47, "Coop Deck One", ROGUE_TAGS, ROGUE_DESC, categories=["Single-player", "Online Co-op"]),
        make_game(48, "Coop Deck Two", ROGUE_TAGS, ROGUE_DESC, categories=["Shared/Split Screen Co-op"]),
    ]
    ids = recommend_ids(build_engine(extra), constraints=Constraints(categories_any=("co-op",)))
    assert set(ids) == {47, 48}


def test_tag_include_and_exclude():
    engine = build_engine()
    cozy = recommend_ids(engine, constraints=Constraints(tags_any=("Cozy",)))
    assert cozy and all(11 <= i <= 18 for i in cozy)
    no_cards = recommend_ids(engine, constraints=Constraints(exclude_tags=("card game",)))
    assert no_cards and not any(1 <= i <= 9 for i in no_cards)


def test_playtime_filter_lets_unknown_playtime_through():
    extra = [
        make_game(52, "Long Deck", ROGUE_TAGS, ROGUE_DESC, average_playtime_forever=3000),
        make_game(53, "Short Deck", ROGUE_TAGS, ROGUE_DESC, average_playtime_forever=120),
        make_game(54, "Unknown Deck", ROGUE_TAGS, ROGUE_DESC, average_playtime_forever=0),
    ]
    ids = recommend_ids(build_engine(extra), constraints=Constraints(max_playtime_hours=5))
    assert 52 not in ids
    assert 53 in ids and 54 in ids


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def test_developer_cap():
    extra = [seed_twin(60 + n, f"Mega Title {w}", developers=["MegaCorp"])
             for n, w in enumerate(["Alpha", "Bravo", "Charlie", "Delta", "Echo"])]
    capped = build_engine(extra).recommend([(1, 1.0)], k=15, today=TODAY)
    assert sum("MegaCorp" in r.developers for r in capped.items) == 2
    uncapped = build_engine(extra, max_per_developer=99).recommend([(1, 1.0)], k=15, today=TODAY)
    assert sum("MegaCorp" in r.developers for r in uncapped.items) == 5


def test_franchise_cap():
    extra = [seed_twin(70, "Deck Saga 1"), seed_twin(71, "Deck Saga 2"),
             seed_twin(72, "Deck Saga: Remastered")]
    capped = recommend_ids(build_engine(extra), k=15)
    assert len({70, 71, 72} & set(capped)) == 1
    uncapped = recommend_ids(build_engine(extra, max_per_franchise=99), k=15)
    assert {70, 71, 72} <= set(uncapped)


def test_mmr_trades_relevance_for_diversity():
    extra = [seed_twin(80, "Alpha Quest"), seed_twin(81, "Beta Voyage")]
    pure_relevance = recommend_ids(build_engine(extra, mmr_lambda=1.0), k=2)
    assert set(pure_relevance) == {80, 81}             # identical twins rank 1 and 2
    diverse = recommend_ids(build_engine(extra, mmr_lambda=0.3), k=2)
    assert not {80, 81} <= set(diverse)                # penalised for being redundant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_find_games_orders_exact_then_prefix_then_substring_and_noise_last():
    extra = [
        make_game(90, "Hades", ROGUE_TAGS, ROGUE_DESC),
        make_game(91, "Hades II", ROGUE_TAGS, ROGUE_DESC),
        make_game(92, "Escape from Hades", ROGUE_TAGS, ROGUE_DESC),
        make_game(93, "Hades Soundtrack", ROGUE_TAGS, ROGUE_DESC),
    ]
    hits = build_engine(extra).find_games("hades", limit=10)
    assert [h["app_id"] for h in hits] == [90, 91, 92, 93]
    assert build_engine(extra).find_games("no such game") == []


def test_result_reports_timings_and_eligible_count():
    result = build_engine().recommend([(1, 1.0)], k=3, today=TODAY)
    assert {"profile", "retrieval", "ranking", "rerank"} <= set(result.timings_ms)
    assert result.n_eligible == len(base_rows()) - 1   # everything except the liked game


@pytest.mark.parametrize("name,expected", [
    ("Supipara - Chapter 1 Spring Has Come!", "supipara"),
    ("Left 4 Dead 2", "left 4 dead"),
    ("Civilization VI", "civilization"),
    ("Portal 2", "portal"),
    ("Deck Saga: Remastered", "deck saga"),
    ("Hades", "hades"),
])
def test_franchise_key(name, expected):
    assert franchise_key(name) == expected