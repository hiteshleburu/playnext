"""
interpreter.py - the Interpreter agent: free text in, a validated request out.

    "something like Hades and Hollow Knight, under $20, no horror, I've played Celeste"
        -> liked games with weights, a price cap, tag filters, games to skip ...

Two stages, and the split matters:
  1. The agent (LLM, with the find_games and search_tags tools) proposes a GameRequest
     as JSON. If the JSON does not validate it gets one more try with the error shown.
  2. Code grounds the proposal. Tags, categories, platforms and numbers must appear in
     the person's own words (small models invent preferences), tag names must exist in the
     dataset (checked with search_tags), "already played" titles must strictly resolve to
     dataset games, and "under $20" is also read straight from the text if the model
     missed it. Liked titles are resolved
     later by the pipeline, which already does that.

The model proposes; code decides. It cannot set anything the schema lacks (there is no
adult-content switch here) and a made-up tag or title never reaches the engine.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents import AgentSpec, run_agent  # noqa: E402
from llm import LLM, LLMError, extract_json  # noqa: E402
from pipeline import PipelineError, call, normalize, resolve_title  # noqa: E402

MAX_TEXT_CHARS = 2000
MAX_LIKED = 20
MAX_TAGS = 10
MAX_PLAYED = 30


class InterpretError(RuntimeError):
    """The request could not be turned into something the engine can use."""


class LikedTitle(BaseModel):
    name: str
    weight: float = Field(1.0, gt=0)

    @field_validator("weight")
    @classmethod
    def _clamp(cls, value):
        return min(value, 5.0)


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
    def _nulls_mean_unset(cls, data):
        # Small models write null for "nothing"; treat it as the default.
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if value is not None}
        return data

    @field_validator("k")
    @classmethod
    def _clamp_k(cls, value):
        return max(1, min(value, 25))

    @field_validator("tag_boosts")
    @classmethod
    def _clamp_boosts(cls, value):
        return {tag: max(-1.0, min(amount, 1.0)) for tag, amount in value.items()}


SYSTEM_PROMPT = """\
You turn a person's request for video game recommendations into a JSON object.

The most important rule: include a field ONLY if the person's own words ask for it. When
in doubt, leave it out. Never add preferences, tags, platforms or numbers they did not
say. An empty result for a field is the normal case.

Fields:
  liked           [{"name": "<game title>", "weight": 1.0}]  games they like. Weight is 1.0
                  unless they say how much: 2.0 for "love"/"favorite", 0.5 for "kind of".
  already_played  ["<game title>"]  games they say they have played or own.
  max_price       number in USD, only if they gave a budget.
  platforms       "windows", "mac" or "linux", only if they named one.
  categories_any  Steam categories, only if they asked (e.g. co-op).
  tags_any        tags a game MUST have, only for firm requirements they stated.
  exclude_tags    tags a game must NOT have, from "no X" / "without X" / "not X".
  tag_boosts      {"<tag>": 0.1 to 1.0} for softer wishes ("would be nice").
  max_playtime_hours, min_release_year, allow_vr_only (only if they ask for VR), k.

Tools: find_games checks a title; search_tags finds exact tag names. Tag fields only work
with exact tag names, so call search_tags for each tag the person mentioned and copy the
spelling it returns. If no tag matches, leave it out. Do not invent games.

Example. Request: "like Hades, no horror, I played Celeste"
Reply: {"liked": [{"name": "Hades", "weight": 1.0}], "already_played": ["Celeste"],
        "exclude_tags": ["Horror"]}

When you are done, reply with ONLY the JSON object, no other text."""


@dataclass
class Interpretation:
    request: GameRequest
    liked: list                 # [{"name", "weight"}] for the pipeline
    k: int
    options: dict               # arguments for the recommend tool
    warnings: list = field(default_factory=list)
    trace: list = field(default_factory=list)


SPEC = AgentSpec("interpreter", SYSTEM_PROMPT, tools=("find_games", "search_tags"), max_steps=8)


async def _propose(llm: LLM, client, text: str):
    """Run the agent; returns (GameRequest, trace). One retry if the JSON is unusable."""
    task, trace, last_error = text, [], None
    for _ in range(2):
        run = await run_agent(llm, client, SPEC, task)
        trace += run.trace
        try:
            return GameRequest.model_validate(extract_json(run.answer)), trace
        except (ValueError, ValidationError) as error:
            last_error = error
            task = (f"{text}\n\nYour previous reply was not a valid request ({str(error)[:300]}). "
                    "Reply with ONLY the corrected JSON object.")
    raise InterpretError(f"Could not understand the request: {str(last_error)[:300]}")


def _mentioned(phrase: str, text_norm: str) -> bool:
    """Does the person's text contain this phrase (whole words; a plural counts)?"""
    p = normalize(phrase)
    padded = f" {text_norm} "
    return bool(p) and (f" {p} " in padded or f" {p}s " in padded)


def _number_in(value, text: str) -> bool:
    """Did the person write this number? (Models sometimes invent a budget or a year.)"""
    wanted = f"{value:g}"
    return any(f"{float(n):g}" == wanted for n in re.findall(r"\d+(?:\.\d+)?", text))


_PRICE_PATTERNS = (
    r"(?:under|below|less than|up to|at most|max(?:imum)?(?: price)?(?: of)?|budget of|<=?)\s*\$?\s*(\d+(?:\.\d+)?)",
    r"\$\s*(\d+(?:\.\d+)?)\s*(?:or less|or under|or below|max|and under|tops)",
)


def _price_in(text: str):
    """A budget stated in the text ("under $20"), or None."""
    for pattern in _PRICE_PATTERNS:
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1))
    return None


async def _real_tag(client, tag: str):
    """The dataset's spelling of `tag`, or None if no such tag exists."""
    found = (await call(client, "search_tags", query=tag, limit=10))["tags"]
    want = tag.strip().lower()
    return next((t["tag"] for t in found if t["tag"].lower() == want), None)


async def _ground_tags(client, tags: list, warnings: list, role: str, cache: dict,
                       text_norm: str) -> list:
    out = []
    for tag in tags[:MAX_TAGS]:
        if not _mentioned(tag, text_norm):
            warnings.append(f"Ignored {role} {tag!r}: you did not ask for it.")
            continue
        key = tag.strip().lower()
        if key not in cache:
            cache[key] = await _real_tag(client, tag)
        real = cache[key]
        if real is None:
            warnings.append(f"Ignored {role} {tag!r}: no such tag in the dataset.")
        elif real not in out:
            out.append(real)
    return out


async def ground(client, request: GameRequest, text: str) -> Interpretation:
    """Check a proposed request against the dataset and the person's own words, and build
    the recommend options."""
    warnings, cache, text_norm = [], {}, normalize(text)

    liked, seen = [], set()
    for item in request.liked:
        key = normalize(item.name)
        if key and key not in seen:
            seen.add(key)
            liked.append({"name": item.name.strip(), "weight": item.weight})
    if len(liked) > MAX_LIKED:
        warnings.append(f"Only the first {MAX_LIKED} liked games were used.")
        liked = liked[:MAX_LIKED]
    if not liked:
        raise InterpretError("Name at least one game you like, e.g. \"games like Hades\".")

    options = {}
    owned = []
    for title in request.already_played[:MAX_PLAYED]:
        matches = (await call(client, "find_games", name=title, limit=5))["matches"]
        hit = resolve_title(title, matches)
        if hit is None:
            warnings.append(f"Could not find {title!r} among the games to skip.")
        elif hit["app_id"] not in owned:
            owned.append(hit["app_id"])
    if owned:
        options["owned_app_ids"] = owned

    tags_any = await _ground_tags(client, request.tags_any, warnings, "required tag", cache, text_norm)
    exclude = await _ground_tags(client, request.exclude_tags, warnings, "excluded tag", cache, text_norm)
    boosts = {}
    for tag, amount in list(request.tag_boosts.items())[:MAX_TAGS]:
        real = (await _ground_tags(client, [tag], warnings, "preferred tag", cache, text_norm) or [None])[0]
        if real:
            boosts[real] = amount

    categories = []
    for category in request.categories_any:
        if category.strip() and _mentioned(category, text_norm):
            categories.append(category)
        elif category.strip():
            warnings.append(f"Ignored category {category!r}: you did not ask for it.")
    platforms = []
    for platform in request.platforms:
        if _mentioned(platform, text_norm):
            platforms.append(platform)
        else:
            warnings.append(f"Ignored platform {platform!r}: you did not ask for it.")

    for name, value in (("tags_any", tags_any), ("exclude_tags", exclude),
                        ("categories_any", categories), ("platforms", platforms)):
        if value:
            options[name] = value
    if boosts:
        options["tag_boosts"] = boosts

    max_price = request.max_price
    if max_price is not None and not _number_in(max_price, text):
        warnings.append(f"Ignored max price {max_price:g}: you did not give that number.")
        max_price = None
    if max_price is None:
        max_price = _price_in(text)
    if max_price is not None:
        options["max_price"] = max_price
    for name in ("max_playtime_hours", "min_release_year"):
        value = getattr(request, name)
        if value is None:
            continue
        if _number_in(value, text):
            options[name] = value
        else:
            warnings.append(f"Ignored {name} {value:g}: you did not give that number.")
    if request.allow_vr_only and _mentioned("vr", text_norm):
        options["allow_vr_only"] = True

    return Interpretation(request, liked, request.k, options, warnings)


async def interpret(llm: LLM, client, text: str) -> Interpretation:
    """Free text -> grounded Interpretation. Raises InterpretError, LLMError or PipelineError."""
    text = (text or "").strip()
    if not text:
        raise InterpretError("The request is empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise InterpretError(f"The request is too long ({len(text)} characters; the limit is "
                             f"{MAX_TEXT_CHARS}).")
    request, trace = await _propose(llm, client, text)
    interpretation = await ground(client, request, text)
    interpretation.trace = trace
    return interpretation