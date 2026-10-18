"""
prepare_data.py - slim the raw Steam games JSON down to what the engine needs.

The raw file is a dict keyed by Steam app ID, with ~40 fields per game
(screenshots, movies, packages, URLs, ...). Most of that is dead weight for a
recommender, so this script keeps only the useful fields, adds two derived
review columns, saves the result, and prints a report so you can decide
thresholds from real numbers.

Usage (from the repo root):
    python src/prepare_data.py --input data/games.json --output data/games_slim.pkl

Requires pandas >= 2.0.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Small defensive helpers: real-world data has gaps and odd types.
# ---------------------------------------------------------------------------

def to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_list(value):
    """Return a clean list of non-empty strings; anything else becomes []."""
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if v and str(v).strip()]


def as_tags(value):
    """Tags are {tag: votes} in the raw data; tolerate a bare list or junk."""
    if isinstance(value, dict):
        return {str(k): to_int(v) for k, v in value.items()}
    if isinstance(value, list):
        return {str(t): 0 for t in value}
    return {}


def parse_owners(value):
    """'0 - 20000' -> (0, 20000); anything unparseable -> (None, None)."""
    try:
        low, high = (int(p.replace(",", "").strip()) for p in str(value).split("-"))
        return low, high
    except ValueError:
        return None, None


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------

def load_raw(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        sys.exit(f"Expected a JSON object keyed by app ID, got {type(data).__name__}.")
    return data


def slim_record(app_id, game):
    """Keep only the fields the engine needs from one raw game entry."""
    owners_low, owners_high = parse_owners(game.get("estimated_owners"))
    return {
        "app_id": to_int(app_id, default=-1),
        "name": str(game.get("name") or "").strip(),
        "release_date": game.get("release_date"),
        "required_age": to_int(game.get("required_age")),
        "price": to_float(game.get("price")),
        "short_description": str(game.get("short_description") or "").strip(),
        "genres": as_list(game.get("genres")),
        "categories": as_list(game.get("categories")),
        "tags": as_tags(game.get("tags")),
        "developers": as_list(game.get("developers")),
        "publishers": as_list(game.get("publishers")),
        "positive": to_int(game.get("positive")),
        "negative": to_int(game.get("negative")),
        "recommendations": to_int(game.get("recommendations")),
        "average_playtime_forever": to_int(game.get("average_playtime_forever")),
        "estimated_owners": game.get("estimated_owners"),
        "owners_low": owners_low,
        "owners_high": owners_high,
        "windows": bool(game.get("windows")),
        "mac": bool(game.get("mac")),
        "linux": bool(game.get("linux")),
    }


def build_table(raw):
    rows = [
        slim_record(app_id, game)
        for app_id, game in raw.items()
        if isinstance(game, dict)
    ]
    df = pd.DataFrame(rows)

    # Release dates come in mixed formats ("Jul 29, 2016", "Oct 2020",
    # "Coming soon", ""). Anything unparseable becomes NaT.
    df["release_date"] = pd.to_datetime(
        df["release_date"], errors="coerce", format="mixed"
    )

    # Derived review columns, used later for ranking and filtering.
    df["n_reviews"] = df["positive"] + df["negative"]
    df["review_ratio"] = (df["positive"] / df["n_reviews"]).where(df["n_reviews"] > 0)
    return df


# ---------------------------------------------------------------------------
# Exploration report
# ---------------------------------------------------------------------------

def print_report(df):
    total = len(df)
    if total == 0:
        print("No games found.")
        return

    def pct(n):
        return f"{n:,} ({n / total:.1%})"

    print("\n=== Dataset report ===")
    print(f"Total games:                {total:,}")
    print(f"Bad / missing app_id:       {pct(int((df['app_id'] < 0).sum()))}")

    print("\n-- Missing fields --")
    print(f"Empty short_description:    {pct(int((df['short_description'] == '').sum()))}")
    print(f"No tags:                    {pct(int((df['tags'].str.len() == 0).sum()))}")
    print(f"No genres:                  {pct(int((df['genres'].str.len() == 0).sum()))}")
    print(f"No developer:               {pct(int((df['developers'].str.len() == 0).sum()))}")

    print("\n-- Reviews (positive + negative) --")
    r = df["n_reviews"]
    print(f"Median: {r.median():,.0f}   Mean: {r.mean():,.1f}   "
          f"p75: {r.quantile(0.75):,.0f}   p90: {r.quantile(0.90):,.0f}   Max: {r.max():,}")
    for threshold in (1, 10, 50, 100):
        print(f"Fewer than {threshold:>3} reviews:      {pct(int((r < threshold).sum()))}")

    print("\n-- Price --")
    free = int((df["price"] == 0).sum())
    paid = df.loc[df["price"] > 0, "price"]
    print(f"Free (price == 0):          {pct(free)}")
    if len(paid):
        print(f"Median paid price:          ${paid.median():.2f}")

    print("\n-- Release date --")
    unparsed = int(df["release_date"].isna().sum())
    print(f"Could not parse:            {pct(unparsed)}")
    if unparsed < total:
        print(f"Range:                      {df['release_date'].min().date()} "
              f"to {df['release_date'].max().date()}")

    print("\n-- Age rating --")
    print(f"required_age > 0:           {pct(int((df['required_age'] > 0).sum()))}")
    print()


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=Path("data/games.json"),
                        help="raw Steam games JSON (default: data/games.json)")
    parser.add_argument("--output", type=Path, default=Path("data/games_slim.pkl"),
                        help="where to save the slim table (default: data/games_slim.pkl)")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")

    print(f"Loading {args.input} ...")
    raw = load_raw(args.input)
    df = build_table(raw)
    del raw  # free the big dict before saving

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.output)
    print(f"Saved {len(df):,} games, {df.shape[1]} columns -> {args.output}")

    print_report(df)


if __name__ == "__main__":
    main()