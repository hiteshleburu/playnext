# playnext

Agentic Steam game recommender: TF-IDF engine + MCP server + multi-agent reranking + FastAPI.

## Objective

- **Input:** a taste profile built from one or more sources (manual picks, Steam library, wishlist), plus an optional free-text request such as "short, co-op, under $20".
- **Output:** 5-10 games from the dataset, each with a short reason.
- **Success:** the picks are relevant to the profile, varied rather than near-duplicates, and match any constraints in the request.

### Input sources and weighting

All sources are merged into a single weighted list of liked games, `(app_id, weight)`, so the engine doesn't care where a game came from.

| Source | Signal | Weighting |
|---|---|---|
| Manual picks | Strongest: the user chose them explicitly | Highest |
| Library | Owning is not liking, so weight by playtime | Scales with hours played |
| Wishlist | Interest without confirmed enjoyment | Moderate |

## Guardrails

1. Never recommend a game the user already owns or has wishlisted.
2. Only recommend games that exist in the dataset. No invented titles.
3. Respect hard constraints from the request (price cap, platform, required age) as filters, not soft preferences.
4. Don't return near-duplicates, such as several entries from the same franchise or developer.
5. Exclude games with too few reviews (`MIN_REVIEWS = 50`, configurable) and downrank games with a low review ratio.
6. Exclude unreleased games (`release_date` later than today).
7. Filter adult content using tags (e.g. "Sexual Content", "Nudity") as well as `required_age`, since `required_age` alone is unreliable.
8. Handle failures gracefully: unmatched games are skipped with a note, and a private Steam profile returns a clear error instead of crashing.

## Architecture

The engine is plain Python and knows nothing about agents, MCP, or the API. Each later layer imports the one below it, never the reverse. That keeps a clean baseline for measuring what the agents add.

1. **Engine:** TF-IDF retrieval over tags, genres, and descriptions; a scoring function for ranking; MMR plus safety filters for re-ranking.
2. **MCP server:** exposes the engine's functions as tools.
3. **Agents:** Interpreter (request to filters), Critic, and Explainer, layered on top of the engine.
4. **FastAPI:** the `/recommend` endpoint.
5. **Eval:** compare engine-only against engine plus agents on the same test requests.


## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install "pandas>=2.0" scikit-learn
```

Download the Steam games dataset from Kaggle (a JSON file keyed by Steam app ID) and save it as `data/games.json`. The `data/` folder is gitignored.

Build the slimmed table:

```bash
python src/prepare_data.py --input data/games.json --output data/games_slim.pkl
```

This keeps only the fields the engine needs, adds `n_reviews` and `review_ratio`, saves the result, and prints a report on missing fields, review counts, prices, and release dates.