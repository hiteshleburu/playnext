"""Matching game titles that come from the web or an LLM against the dataset."""

import re
import unicodedata

# Words that can follow a title in a re-release without making it a different game.
EDITION_WORDS = {
    "definitive", "edition", "remastered", "remaster", "hd", "enhanced", "game", "of", "the",
    "year", "goty", "deluxe", "complete", "directors", "cut", "anniversary", "special",
    "ultimate", "gold", "legendary",
}


def normalize(text) -> str:
    """Lowercase, no accents, apostrophes or punctuation, single spaces."""
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", re.sub(r"['’`]", "", text)).strip()


def resolve_title(title: str, hits: list):
    """The dataset game `title` means, or None. Strict: an exact match, or the same title
    plus edition words only. "Ori" is not "Ori and the Will of the Wisps"."""
    want = normalize(title)
    if not want:
        return None
    for hit in hits:
        if normalize(hit["name"]) == want:
            return hit
    for hit in hits:
        have = normalize(hit["name"])
        if have.startswith(want + " ") and all(w in EDITION_WORDS for w in have[len(want):].split()):
            return hit
    return None


def count_mentions(title: str, texts: list) -> int:
    """How many of the texts contain the title as whole words."""
    want = normalize(title)
    return sum(f" {want} " in f" {normalize(t)} " for t in texts) if want else 0


def clean_titles(raw, max_chars: int = 80, limit: int = 40) -> list:
    """De-duplicated, sane-length titles from an LLM reply."""
    if not isinstance(raw, list):
        return []
    titles, seen = [], set()
    for item in raw:
        title = str(item).strip()
        key = normalize(title)
        if key and len(title) <= max_chars and key not in seen:
            seen.add(key)
            titles.append(title)
    return titles[:limit]