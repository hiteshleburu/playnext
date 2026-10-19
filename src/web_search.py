"""
web_search.py - the web search tool for playnext, backed by a local SearXNG.

Returns short, cleaned snippets (title, url, snippet) that a small local model
can read. Everything it returns is UNTRUSTED text from the open internet: pages
can contain instructions aimed at the model. Callers must only use results as
data (e.g. "which game titles are mentioned?") and must never let them change
guardrails, tool permissions or prompts.

Features
  * Disk cache (default 7 days), so repeated queries and re-run tests cost nothing.
  * A few retries with backoff, because SearXNG's upstream engines are flaky.
  * A clear error when JSON output is not enabled (HTTP 403).
  * Standard library only.

Configuration (environment variables)
  SEARXNG_URL        base URL of the instance        (default http://localhost:8080)
  WEB_SEARCH_CACHE   cache folder                    (default .cache/web_search)

Try it:
    docker compose up -d
    python src/web_search.py "games like Chants of Sennaar"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_URL = "http://localhost:8080"
DEFAULT_CACHE_DIR = ".cache/web_search"
CACHE_TTL_SECONDS = 7 * 24 * 3600

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


class WebSearchError(RuntimeError):
    """The search backend could not be used (misconfigured, down, or blocked)."""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text, limit: int) -> str:
    """Collapse whitespace, drop control characters and truncate."""
    text = _CONTROL_CHARS.sub("", str(text or ""))
    text = _WHITESPACE.sub(" ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _cache_path(cache_dir: Path, provider: str, query: str) -> Path:
    key = hashlib.sha256(f"{provider}\n{query.strip().lower()}".encode()).hexdigest()
    return cache_dir / f"{key}.json"


def _read_cache(path: Path, ttl: float):
    try:
        if time.time() - path.stat().st_mtime > ttl:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_cache(path: Path, results: list) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results), encoding="utf-8")
    except OSError:
        pass  # a cache that cannot be written is not worth failing a search over


# ---------------------------------------------------------------------------
# Provider: SearXNG
# ---------------------------------------------------------------------------

def _search_searxng(query: str, base_url: str, timeout: float, **_) -> list:
    url = base_url.rstrip("/") + "/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json"}
    )
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 403:
            raise WebSearchError(
                "SearXNG refused the JSON request (HTTP 403). Make sure searxng/settings.yml "
                "lists `json` under search.formats, then restart: docker compose restart"
            ) from error
        raise
    return payload.get("results") or []


PROVIDERS = {"searxng": _search_searxng}   # add "tavily" / "brave" here later


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def web_search(
    query: str,
    max_results: int = 5,
    snippet_chars: int = 300,
    *,
    provider: str = "searxng",
    base_url: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    cache_ttl: float = CACHE_TTL_SECONDS,
    retries: int = 2,
    timeout: float = 20.0,
    backoff: float = 1.0,
) -> list[dict]:
    """Search the web and return up to `max_results` dicts: title, url, snippet.

    Returns [] if the backend answers but has nothing (after retrying).
    Raises WebSearchError if the backend cannot be reached or is misconfigured.
    """
    query = query.strip()
    if not query:
        return []
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")

    base_url = base_url or os.environ.get("SEARXNG_URL", DEFAULT_URL)
    cache_dir = Path(cache_dir or os.environ.get("WEB_SEARCH_CACHE", DEFAULT_CACHE_DIR))
    cache_file = _cache_path(cache_dir, provider, query)

    cached = _read_cache(cache_file, cache_ttl)
    if cached is not None:
        return cached[:max_results]

    raw, last_error = [], None
    for attempt in range(retries + 1):
        try:
            raw = PROVIDERS[provider](query, base_url=base_url, timeout=timeout)
            last_error = None
        except WebSearchError:
            raise
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
            last_error = error
            raw = []
        if raw:
            break
        if attempt < retries:
            time.sleep(backoff * (2 ** attempt))
    if last_error is not None and not raw:
        raise WebSearchError(
            f"Could not reach {provider} at {base_url} ({last_error}). "
            "Is it running? Try: docker compose up -d"
        ) from last_error

    results, seen = [], set()
    for item in raw:
        link = str(item.get("url") or "").strip()
        if not link or link in seen:
            continue
        seen.add(link)
        results.append(asdict(SearchResult(
            title=clean_text(item.get("title"), 150),
            url=link,
            snippet=clean_text(item.get("content"), snippet_chars),
        )))
        if len(results) >= max_results:
            break

    if results:                      # never cache an empty answer: it is probably a hiccup
        _write_cache(cache_file, results)
    return results


def games_like_queries(game_name: str) -> list[str]:
    """Query templates for finding games similar to one the user likes."""
    return [f"games like {game_name}", f"{game_name} similar games"]


# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(description="Try the web search tool.")
    parser.add_argument("query")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true", help="ignore cached results")
    args = parser.parse_args()
    try:
        results = web_search(args.query, args.max_results,
                             cache_ttl=0 if args.no_cache else CACHE_TTL_SECONDS)
    except WebSearchError as error:
        raise SystemExit(f"Search failed: {error}")
    if not results:
        raise SystemExit("No results.")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}\n")


if __name__ == "__main__":
    _main()
