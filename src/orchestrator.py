"""Free-text request in, recommendations out.

    python src/orchestrator.py 'games like Hades, under $20, no horror' --trace
    python src/orchestrator.py '...' --provider openrouter --model <model-id> --no-web

In PowerShell use single quotes: inside double quotes "$20" is a variable and disappears.
Flow: Interpreter agent (text -> verified filters) -> pipeline (web + engine) -> result."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from connection import PipelineError, server_target, with_client  # noqa: E402
from interpreter import InterpretError, interpret  # noqa: E402
from llm import LLM, LLMError, load_dotenv  # noqa: E402
from pipeline import run_pipeline  # noqa: E402


async def answer_text(client, llm: LLM, text: str, *, use_web: bool = True,
                      results_per_query: int = 5, log=lambda message: None) -> dict:
    """Free text -> recommendations, using an already-connected MCP client."""
    plan = await interpret(llm, client, text, use_web)
    log("Understood: " + ", ".join(f"{g['name']} (x{g['weight']:g})" for g in plan.liked)
        + (f" | {plan.options}" if plan.options else ""))
    report = await run_pipeline(client, llm, plan.liked, k=plan.k, options=plan.options,
                                use_web=use_web, results_per_query=results_per_query, log=log)
    report["warnings"] = plan.warnings + report["warnings"]
    report["interpretation"] = {"liked": plan.liked, "k": plan.k, "options": plan.options}
    report["trace"] = plan.trace
    return report


async def recommend_from_text(target, llm: LLM, text: str, **kwargs) -> dict:
    """Connect to the MCP server `target` and answer. Keyword arguments go to answer_text."""
    return await with_client(target, lambda client: answer_text(client, llm, text, **kwargs))


def print_report(report: dict, trace: bool = False) -> None:
    print()
    for rank, r in enumerate(report["recommendations"], 1):
        flag = f"  [web x{r['mentions']}]" if r["mentions"] else ""
        print(f"{rank:>2}. {r['name']}  (${r['price']:.2f}){flag}")
        reason = f"like {r['because_of']}" if r["because_of"] else "suggested by the web"
        print(f"     {reason} | tags: {', '.join(r['top_tags'])}")
    web = report["web"]
    if web:
        print(f"\nWeb: {web['results']} results read, {web['titles_found']} titles found, "
              f"{web['verified']} in the dataset.")
        if web["dropped"]:
            print("Not in the dataset (dropped): " + ", ".join(web["dropped"][:15]))
    for warning in report["warnings"]:
        print(f"! {warning}")
    if trace:
        print("\nAgent tool calls:")
        for step in report["trace"]:
            print(f"  {step['agent']}: {step['tool']}({step['arguments']}) -> "
                  + (f"ERROR {step['error']}" if step["error"] else "ok"))


async def amain(args) -> None:
    llm = LLM(provider=args.provider, model=args.model)
    log = lambda message: print(message, file=sys.stderr, flush=True)  # noqa: E731
    log(f"Model: {llm.describe()}")
    report = await recommend_from_text(server_target(args.data, args.server_url), llm, args.request,
                                       use_web=not args.no_web, log=log)
    print_report(report, args.trace)


def main() -> None:
    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Recommend games from a plain-English request.")
    parser.add_argument("request", help='e.g. \'games like Hades, under $20, no horror\'')
    parser.add_argument("--no-web", action="store_true", help="skip web search: the engine alone")
    parser.add_argument("--trace", action="store_true", help="show the agent's tool calls")
    parser.add_argument("--data", type=Path, default=Path("data/games_slim.pkl"))
    parser.add_argument("--server-url", help="use a running server, e.g. http://127.0.0.1:8000/mcp")
    parser.add_argument("--provider", choices=["ollama", "openrouter"],
                        help="which LLM to use (default: LLM_PROVIDER or ollama)")
    parser.add_argument("--model", help="override the provider's model")
    args = parser.parse_args()
    try:
        asyncio.run(amain(args))
    except (InterpretError, PipelineError, LLMError) as error:
        raise SystemExit(f"Error: {error}")


if __name__ == "__main__":
    main()