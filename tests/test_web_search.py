"""Tests for web_search.py against a fake SearXNG server running on localhost.

No Docker or internet needed: a tiny HTTP server in a background thread plays
the part of SearXNG, so we can check parsing, caching, retries and errors.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import web_search as ws  # noqa: E402


class FakeSearxng:
    """A fake SearXNG. `script` is a list of (status, payload) replies, used in
    order; the last one repeats. `hits` records every request made."""

    def __init__(self, script):
        self.script = list(script)
        self.hits = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                parsed = urlparse(self.path)
                outer.hits.append(parse_qs(parsed.query))
                index = min(len(outer.hits) - 1, len(outer.script) - 1)
                status, payload = outer.script[index]
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake():
    created = []

    def make(script):
        server = FakeSearxng(script)
        created.append(server)
        return server

    yield make
    for server in created:
        server.close()


def search(server, query="games like X", tmp_path=None, **kwargs):
    kwargs.setdefault("backoff", 0.01)
    return ws.web_search(query, base_url=server.url, cache_dir=tmp_path, **kwargs)


RESULTS = {"results": [
    {"title": "Heaven's Vault", "url": "https://a.example/1", "content": "An archaeology adventure."},
    {"title": "Obra Dinn", "url": "https://a.example/2", "content": "A first-person mystery."},
    {"title": "Duplicate", "url": "https://a.example/1", "content": "Same URL again."},
]}


def test_parses_results_and_removes_duplicate_urls(fake, tmp_path):
    server = fake([(200, RESULTS)])
    results = search(server, tmp_path=tmp_path)
    assert [r["url"] for r in results] == ["https://a.example/1", "https://a.example/2"]
    assert results[0] == {"title": "Heaven's Vault", "url": "https://a.example/1",
                          "snippet": "An archaeology adventure."}


def test_requests_json_format_with_the_query(fake, tmp_path):
    server = fake([(200, RESULTS)])
    search(server, "games like Chants of Sennaar", tmp_path=tmp_path)
    assert server.hits[0]["q"] == ["games like Chants of Sennaar"]
    assert server.hits[0]["format"] == ["json"]


def test_max_results_and_snippet_truncation(fake, tmp_path):
    payload = {"results": [{"title": f"T{i}", "url": f"https://x.example/{i}",
                            "content": "word " * 200} for i in range(10)]}
    results = search(fake([(200, payload)]), tmp_path=tmp_path, max_results=3, snippet_chars=50)
    assert len(results) == 3
    assert all(len(r["snippet"]) <= 50 for r in results)
    assert results[0]["snippet"].endswith("…")


def test_untrusted_text_is_cleaned(fake, tmp_path):
    payload = {"results": [{"title": "Line\nbreak\x00 title", "url": "https://x.example/1",
                            "content": "  lots   of\n\n whitespace\x07  "}]}
    result = search(fake([(200, payload)]), tmp_path=tmp_path)[0]
    assert result["title"] == "Line break title"
    assert result["snippet"] == "lots of whitespace"


def test_results_without_urls_are_skipped(fake, tmp_path):
    payload = {"results": [{"title": "No url", "content": "x"},
                           {"title": "Has url", "url": "https://x.example/1", "content": "y"}]}
    assert len(search(fake([(200, payload)]), tmp_path=tmp_path)) == 1


def test_second_identical_search_is_served_from_cache(fake, tmp_path):
    server = fake([(200, RESULTS)])
    first = search(server, "Same Query", tmp_path=tmp_path)
    second = search(server, "  same query ", tmp_path=tmp_path)   # case/space-insensitive
    assert first == second
    assert len(server.hits) == 1


def test_expired_cache_is_refetched(fake, tmp_path):
    server = fake([(200, RESULTS)])
    search(server, tmp_path=tmp_path)
    time.sleep(0.05)
    search(server, tmp_path=tmp_path, cache_ttl=0.01)
    assert len(server.hits) == 2


def test_empty_results_are_retried_and_then_returned_empty_and_not_cached(fake, tmp_path):
    server = fake([(200, {"results": []})])
    assert search(server, tmp_path=tmp_path, retries=2) == []
    assert len(server.hits) == 3
    search(server, tmp_path=tmp_path, retries=0)
    assert len(server.hits) == 4                                  # empty answer was not cached


def test_retry_succeeds_after_a_flaky_failure(fake, tmp_path):
    server = fake([(500, {}), (200, RESULTS)])
    assert len(search(server, tmp_path=tmp_path, retries=2)) == 2
    assert len(server.hits) == 2


def test_403_gives_a_helpful_error_without_retrying(fake, tmp_path):
    server = fake([(403, {})])
    with pytest.raises(ws.WebSearchError, match="search.formats"):
        search(server, tmp_path=tmp_path, retries=2)
    assert len(server.hits) == 1


def test_unreachable_server_raises_a_clear_error(tmp_path):
    with pytest.raises(ws.WebSearchError, match="docker compose up"):
        ws.web_search("anything", base_url="http://127.0.0.1:9", cache_dir=tmp_path,
                      retries=1, backoff=0.01, timeout=2)


def test_blank_query_makes_no_request(fake, tmp_path):
    server = fake([(200, RESULTS)])
    assert search(server, "   ", tmp_path=tmp_path) == []
    assert server.hits == []


def test_unknown_provider_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        ws.web_search("x", provider="nope", cache_dir=tmp_path)


def test_games_like_queries():
    assert ws.games_like_queries("Hades") == ["games like Hades", "Hades similar games"]
