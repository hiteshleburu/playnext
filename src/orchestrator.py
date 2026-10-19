"""
orchestrator.py - free-text request in, recommendations out. The multi-agent flow.

    request text
      -> Interpreter agent   text -> validated, grounded request (tools: find_games, search_tags)
      -> pipeline            web search + LLM title extraction + engine ranking
      -> result

Every step talks to the playnext MCP server through one shared connection. The agents
never see more tools than their allowlist, and everything a model proposes is checked by
code before it reaches the engine (see interpreter.py and pipeline.py).

Try it (needs Ollama, and `docker compose up -d` for web search):
    python src/orchestrator.py "something like Hades and Hollow Knight under $20, no horror"
    python src/orchestrator.py "cozy games like Stardew Valley" --no-web --trace
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interpreter import InterpretError, interpret  # noqa: E402
from llm import LLM, LLMError  # noqa: E402
from pipeline import PipelineError, _print_report, run_pipeline, server_target, with_client  # noqa: E402


async def recommend_from_text(target, llm: LLM, text: str, *, use_web: bool = True,
                              results_per_query: int = 5, log=lambda message: None) -> dict:
    """Run the whole flow over one MCP connection. `target` is whatever `with_client` takes."""

    async def work(client):
        interpretation = await interpret(llm, client, text)
        log("Understood: " + ", ".join(f"{g['name']} (x{g['weight']:g})" for g in interpretation.liked)
            + (f" | {interpretation.options}" if interpretation.options else ""))
        report = await run_pipeline(
            client, llm, interpretation.liked, k=interpretation.k, options=interpretation.options,
            use_web=use_web, results_per_query=results_per_query, log=log)
        report["warnings"] = interpretation.warnings + report["warnings"]
        report["interpretation"] = {"liked": interpretation.liked, "k": interpretation.k,
                                    "options": interpretation.options}
        report["trace"] = interpretation.trace
        return report

    return await with_client(target, work)


def _print_trace(report: dict) -> None:
    print("\nAgent tool calls:")
    for step in report["trace"]:
        status = f"ERROR {step['error']}" if step["error"] else "ok"
        print(f"  {step['agent']}: {step['tool']}({step['arguments']}) -> {status}")


async def _main(args) -> None:
    llm = LLM(model=args.model) if args.model else LLM()
    log = lambda message: print(message, file=sys.stderr, flush=True)  # noqa: E731
    report = await recommend_from_text(
        server_target(args.data, args.server_url), llm, args.request,
        use_web=not args.no_web, log=log)
    _print_report(report)
    if args.trace:
        _print_trace(report)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Recommend games from a plain-English request.")
    parser.add_argument("request", help='e.g. "games like Hades, under $20, no horror"')
    parser.add_argument("--no-web", action="store_true", help="skip web search: the engine alone")
    parser.add_argument("--trace", action="store_true", help="show the agent's tool calls")
    parser.add_argument("--data", type=Path, default=Path("data/games_slim.pkl"))
    parser.add_argument("--server-url", help="use a running server, e.g. http://127.0.0.1:8000/mcp")
    parser.add_argument("--model", help="override LLM_MODEL")
    args = parser.parse_args()
    try:
        asyncio.run(_main(args))
    except (InterpretError, PipelineError, LLMError) as error:
        raise SystemExit(f"Error: {error}")


if __name__ == "__main__":
    main()