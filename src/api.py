"""FastAPI front door for playnext. A website talks to this, never to MCP directly.

    pip install fastapi uvicorn
    python src/api.py            then open http://127.0.0.1:8001/docs

POST /recommend       structured input: liked games ({app_id, name, weight}) plus filters
POST /recommend/text  free text, understood by the Interpreter agent
GET  /games/search    title search for a dropdown
GET  /health
GET  /             the web page (web/index.html)

One MCP connection is opened at startup and shared by all requests, so the engine is built
once. The LLM comes from .env (LLM_PROVIDER, OPENROUTER_API_KEY, ...). Set MCP_SERVER_URL to use
a server you started yourself, PLAYNEXT_DATA for the dataset, PLAYNEXT_CORS_ORIGINS for your site.
"""

import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from connection import PipelineError, call, server_target  # noqa: E402
from interpreter import InterpretError  # noqa: E402
from llm import LLM, LLMError, load_dotenv  # noqa: E402
from orchestrator import answer_text  # noqa: E402
from pipeline import run_pipeline  # noqa: E402

from mcp.client import Client  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    app.state.llm = LLM()
    target = server_target(os.environ.get("PLAYNEXT_DATA", "data/games_slim.pkl"),
                           os.environ.get("MCP_SERVER_URL"))
    async with AsyncExitStack() as stack:
        app.state.client = await stack.enter_async_context(Client(target))
        yield


app = FastAPI(title="playnext", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_methods=["*"], allow_headers=["*"],
    allow_origins=os.environ.get("PLAYNEXT_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(","))


@app.exception_handler(PipelineError)
@app.exception_handler(InterpretError)
async def bad_request(request: Request, error: Exception):
    return JSONResponse({"detail": str(error)}, status_code=422)


@app.exception_handler(LLMError)
async def model_unavailable(request: Request, error: LLMError):
    return JSONResponse({"detail": str(error)}, status_code=502)


class LikedIn(BaseModel):
    app_id: int
    name: str
    weight: float = Field(1.0, gt=0)


class RecommendIn(BaseModel):
    liked: list[LikedIn] = Field(min_length=1, max_length=50)
    k: int = Field(10, ge=1, le=25)
    use_web: bool = True
    owned_app_ids: list[int] = []
    wishlist_app_ids: list[int] = []
    max_price: Optional[float] = Field(None, ge=0)
    platforms: list[Literal["windows", "mac", "linux"]] = []
    categories_any: list[str] = []
    tags_any: list[str] = []
    exclude_tags: list[str] = []
    max_playtime_hours: Optional[float] = Field(None, gt=0)
    min_release_year: Optional[int] = None
    tag_boosts: dict[str, float] = {}


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    use_web: bool = True


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(Path(__file__).resolve().parent.parent / "web" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/games/search")
async def search_games(request: Request, q: str = Query(min_length=1, max_length=80),
                       limit: int = Query(8, ge=1, le=10)):
    return (await call(request.app.state.client, "find_games", name=q, limit=limit))["matches"]


@app.post("/recommend")
async def recommend(body: RecommendIn, request: Request):
    options = body.model_dump(exclude={"liked", "k", "use_web"}, exclude_defaults=True)
    return await run_pipeline(request.app.state.client, request.app.state.llm,
                              [g.model_dump() for g in body.liked], k=body.k, options=options,
                              use_web=body.use_web)


@app.post("/recommend/text")
async def recommend_from_text(body: TextIn, request: Request):
    return await answer_text(request.app.state.client, request.app.state.llm, body.text,
                             use_web=body.use_web)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8001)))