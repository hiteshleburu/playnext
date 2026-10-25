"""The Interpreter agent: free text in, a verified request out.

The model proposes a GameRequest as JSON. Then `ground` checks it against the dataset and
the person's own words: tags must exist and be mentioned, numbers must be written by the
person, and the model can never set anything the schema lacks (there is no adult switch)."""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import AgentSpec, run_agent  # noqa: E402
from connection import PipelineError, call  # noqa: E402
from llm import LLM, extract_json  # noqa: E402
from titles import count_mentions, normalize, resolve_title  # noqa: E402

MAX_TEXT_CHARS = 2000


class InterpretError(RuntimeError):
    """The request could not be turned into something the engine can use."""


class LikedTitle(BaseModel):
    name: str
    weight: float = Field(1.0, gt=0)


class GameRequest(BaseModel):
    liked: list[LikedTitle] = []
    already_played: list[str] = []
    max_price: Optional[float] = Field(None, ge=0)
    platforms: list[Literal["windows", "mac", "linux"]] = []
    categories_any: list[str] = []
    tags_any: list[str] = []
    exclude_tags: list[str] = []
    max_playtime_hours: Optional[float] = Field(None, gt=0)
    min_release_year: Optional[int] = Field(None, ge=1970, le=2100)
    allow_vr_only: bool = False
    tag_boosts: dict[str, float] = {}
    k: int = 10

    @model_validator(mode="before")
    @classmethod
    def _null_means_unset(cls, data):
        return {key: v for key, v in data.items() if v is not None} if isinstance(data, dict) else data

    @field_validator("k")
    @classmethod
    def _clamp_k(cls, value):
        return max(1, min(value, 25))

    @field_validator("tag_boosts")
    @classmethod
    def _clamp_boosts(cls, value):
        return {tag: max(-1.0, min(amount, 1.0)) for tag, amount in value.items()}


SYSTEM_PROMPT = """\
You turn a request for video game recommendations into a JSON object.

Most important: include a field ONLY if the person's own words ask for it. When in doubt,
leave it out. Never add tags, platforms or numbers they did not say.

Fields:
  liked           [{"name": "<game title>", "weight": 1.0}]  weight 2.0 for "love"/"favorite", 0.5 for "kind of"
  already_played  ["<game title>"]  games they say they played or own
  max_price       number in USD, only if they gave a budget
  platforms       "windows", "mac" or "linux", only if named
  categories_any  Steam categories, only if asked (e.g. co-op)
  tags_any        tags a game MUST have, only for firm requirements
  exclude_tags    tags a game must NOT have ("no X", "without X")
  tag_boosts      {"<tag>": 0.1 to 1.0} for softer wishes ("would be nice")
  max_playtime_hours, min_release_year, allow_vr_only (only if they ask for VR), k

Game titles: copy them exactly as the person wrote them, even if misspelt, abbreviated or
unfamiliar to you. Never drop or change a title; the code corrects titles afterwards.

Tools: search_tags finds exact tag names. Tag fields need exact tag names, so call
search_tags for each tag the person mentioned and copy its spelling. If no tag matches,
leave it out.

Example. Request: "like Hades, no horror, I played Celeste"
Reply: {"liked": [{"name": "Hades", "weight": 1.0}], "already_played": ["Celeste"],
        "exclude_tags": ["Horror"]}

Reply with ONLY the JSON object."""

SPEC = AgentSpec("interpreter", SYSTEM_PROMPT, tools=("search_tags",), max_steps=8)

PRICE_PATTERNS = (
    r"(?:under|below|less than|up to|at most|max(?:imum)?(?: price)?(?: of)?|budget of|<=?)\s*\$?\s*(\d+(?:\.\d+)?)",
    r"\$\s*(\d+(?:\.\d+)?)\s*(?:or less|or under|or below|max|and under|tops)",
)


@dataclass
class Interpretation:
    request: GameRequest
    liked: list                 # [{"name", "weight"}] for the pipeline
    k: int
    options: dict               # arguments for the recommend tool
    warnings: list = field(default_factory=list)
    trace: list = field(default_factory=list)


def _mentioned(phrase: str, said: str) -> bool:
    """Is the phrase in the person's text (whole words, a plural counts)?"""
    p = normalize(phrase)
    return bool(p) and (f" {p} " in f" {said} " or f" {p}s " in f" {said} ")


def _number_in(value, text: str) -> bool:
    return any(f"{float(n):g}" == f"{value:g}" for n in re.findall(r"\d+(?:\.\d+)?", text))


def _price_in(text: str):
    for pattern in PRICE_PATTERNS:
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1))
    return None


def _site_title(raw: str) -> str:
    """"MIO: Memories in Orbit on Steam" -> "MIO: Memories in Orbit"."""
    return re.split(r"\s+(?:on Steam\b|[-–|·]\s)", raw)[0].strip()


async def title_from_web(client, title: str):
    """The dataset game a misspelt or abbreviated title means, found through a web search.
    Search text is untrusted: a game counts only if it exists in the dataset (strict match)
    and its name appears in at least two of the results."""
    try:
        results = (await call(client, "web_search", query=f"{title} steam game", max_results=5))["results"]
    except PipelineError:
        return None
    texts = [f"{r['title']} {r['snippet']}" for r in results]
    best, best_count, tried = None, 1, set()
    for r in results:
        candidate = _site_title(r["title"])
        if not normalize(candidate) or normalize(candidate) in tried:
            continue
        tried.add(normalize(candidate))
        matches = (await call(client, "find_games", name=candidate, limit=5))["matches"]
        hit = resolve_title(candidate, matches)
        count = count_mentions(hit["name"], texts) if hit else 0
        if count > best_count:
            best, best_count = hit, count
    return best


async def find_title(client, title: str, use_web: bool):
    """Dataset game for a title the person typed: exact first, then the web if allowed."""
    matches = (await call(client, "find_games", name=title, limit=5))["matches"]
    return resolve_title(title, matches) or (await title_from_web(client, title) if use_web else None)


async def ground(client, request: GameRequest, text: str, use_web: bool = False) -> Interpretation:
    """Check a proposed request against the dataset and the person's words."""
    warnings, real_tags, said = [], {}, normalize(text)

    def said_only(role, items):
        for item in items:
            if item.strip() and _mentioned(item, said):
                yield item
            else:
                warnings.append(f"Ignored {role} {item!r}: you did not ask for it.")

    async def tags(role, names):
        out = []
        for tag in said_only(role, names):
            key = tag.strip().lower()
            if key not in real_tags:
                found = (await call(client, "search_tags", query=tag, limit=10))["tags"]
                real_tags[key] = next((t["tag"] for t in found if t["tag"].lower() == key), None)
            if real_tags[key] is None:
                warnings.append(f"Ignored {role} {tag!r}: no such tag in the dataset.")
            elif real_tags[key] not in out:
                out.append(real_tags[key])
        return out

    def number(label, value):
        if value is None or _number_in(value, text):
            return value
        warnings.append(f"Ignored {label} {value:g}: you did not give that number.")

    liked, seen = [], set()
    for item in request.liked:
        if normalize(item.name) and normalize(item.name) not in seen:
            seen.add(normalize(item.name))
            entry = {"name": item.name.strip(), "weight": item.weight}
            hit = await find_title(client, entry["name"], True) if use_web else None
            if hit:
                if normalize(hit["name"]) != normalize(entry["name"]):
                    warnings.append(f"Read {entry['name']!r} as {hit['name']!r}.")
                entry = {"app_id": hit["app_id"], "name": hit["name"], "weight": item.weight}
            liked.append(entry)
    if not liked:
        raise InterpretError('Name at least one game you like, e.g. "games like Hades".')

    owned = []
    for title in request.already_played:
        hit = await find_title(client, title, use_web)
        if hit is None:
            warnings.append(f"Could not find {title!r} among the games to skip.")
        elif hit["app_id"] not in owned:
            owned.append(hit["app_id"])

    boosts = {}
    for tag, amount in request.tag_boosts.items():
        for real in await tags("preferred tag", [tag]):
            boosts[real] = amount

    options = {
        "owned_app_ids": owned,
        "tags_any": await tags("required tag", request.tags_any),
        "exclude_tags": await tags("excluded tag", request.exclude_tags),
        "categories_any": list(said_only("category", request.categories_any)),
        "platforms": list(said_only("platform", request.platforms)),
        "tag_boosts": boosts,
        "allow_vr_only": request.allow_vr_only and _mentioned("vr", said),
    }
    options = {key: value for key, value in options.items() if value}
    price = number("max price", request.max_price)
    numbers = {"max_price": _price_in(text) if price is None else price,
               "max_playtime_hours": number("max_playtime_hours", request.max_playtime_hours),
               "min_release_year": number("min_release_year", request.min_release_year)}
    options.update({key: value for key, value in numbers.items() if value is not None})
    return Interpretation(request, liked, request.k, options, warnings)


async def interpret(llm: LLM, client, text: str, use_web: bool = False) -> Interpretation:
    """Free text -> verified Interpretation. With use_web, misspelt or abbreviated titles are
    corrected through a web search. Raises InterpretError (or LLMError / PipelineError)."""
    text = (text or "").strip()
    if not text:
        raise InterpretError("The request is empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise InterpretError(f"The request is too long ({len(text)} characters; limit {MAX_TEXT_CHARS}).")

    task, trace = text, []
    for attempt in range(2):                 # one retry if the JSON is unusable
        run = await run_agent(llm, client, SPEC, task)
        trace += run.trace
        try:
            request = GameRequest.model_validate(extract_json(run.answer))
            break
        except (ValueError, ValidationError) as error:
            problem = str(error)[:300]
            task = (f"{text}\n\nYour previous reply was not a valid request ({problem}). "
                    "Reply with ONLY the corrected JSON object.")
    else:
        raise InterpretError(f"Could not understand the request: {problem}")

    interpretation = await ground(client, request, text, use_web)
    interpretation.trace = trace
    return interpretation