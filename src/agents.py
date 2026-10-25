"""A small generic agent loop over MCP tools.

An agent is a system prompt, an allowlist of tool names and a step budget. Only allowlisted
tools are shown to the model, calls to anything else are refused, repeated identical calls
are refused, and results are truncated. Errors go back to the model as text so it can retry."""

import asyncio
import json
from dataclasses import dataclass, field

from llm import LLM


@dataclass(frozen=True)
class AgentSpec:
    name: str
    system: str
    tools: tuple = ()
    max_steps: int = 6
    max_result_chars: int = 4000


@dataclass
class AgentRun:
    answer: str
    trace: list = field(default_factory=list)      # one entry per tool call


def _result_text(result, limit: int) -> str:
    if result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)
    else:
        text = " ".join(getattr(c, "text", "") for c in result.content)
    return text if len(text) <= limit else f"{text[:limit]}... [truncated, {len(text) - limit} more characters]"


async def run_agent(llm: LLM, client, spec: AgentSpec, task: str) -> AgentRun:
    available = {t.name: t for t in (await client.list_tools()).tools}
    missing = set(spec.tools) - set(available)
    if missing:
        raise RuntimeError(f"Agent {spec.name!r} needs tools the server lacks: {sorted(missing)}")
    schemas = [{"type": "function", "function": {
        "name": t.name, "description": t.description or "", "parameters": t.input_schema}}
        for t in (available[n] for n in spec.tools)]

    messages = [{"role": "system", "content": spec.system}, {"role": "user", "content": task}]
    trace, done = [], set()

    for _ in range(spec.max_steps):
        reply = await asyncio.to_thread(llm.chat_message, messages, **({"tools": schemas} if schemas else {}))
        if not reply["tool_calls"]:
            return AgentRun(reply["content"], trace)
        messages.append({"role": "assistant", "content": reply["content"], "tool_calls": reply["tool_calls"]})
        for call in reply["tool_calls"]:
            function = call.get("function", {})
            name = function.get("name", "")
            entry = {"agent": spec.name, "tool": name, "arguments": None, "error": None}
            trace.append(entry)
            try:
                if name not in spec.tools:
                    raise ValueError(f"Unknown tool {name!r}. Available: {sorted(spec.tools)}")
                arguments = function.get("arguments") or "{}"
                arguments = arguments if isinstance(arguments, dict) else json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
                entry["arguments"] = arguments
                key = (name, json.dumps(arguments, sort_keys=True))
                if key in done:
                    raise ValueError("You already made this exact call; its result is above. "
                                     "Use it, or answer now if you have what you need.")
                done.add(key)
                result = await client.call_tool(name, arguments)
                text = _result_text(result, spec.max_result_chars)
                if result.is_error:
                    entry["error"], text = text, f"Error: {text}"
            except ValueError as error:                 # includes JSONDecodeError
                entry["error"], text = str(error), f"Error: {error}"
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": text})

    messages.append({"role": "user", "content": "You are out of tool calls. Answer now using what you have."})
    reply = await asyncio.to_thread(llm.chat_message, messages)
    return AgentRun(reply["content"], trace)