"""
agents.py - a small generic agent runner.

An agent here is a system prompt, an allowlist of MCP tool names and a step budget.
`run_agent` is the loop: show the model the allowed tools (discovered from the MCP
server, not hard-coded), run the tool calls it asks for, feed the results back, and
stop when it answers in plain text.

Safety notes
  * Only allowlisted tools are shown to the model, and a call to any other name is
    refused (models do invent tool names), with the error fed back to it.
  * Tool results are cut to `max_result_chars` so one big result cannot fill the
    context window.
  * Tool errors and bad arguments go back to the model as text so it can retry;
    only a broken model server or MCP connection raises.
  * Repeating an identical tool call is refused (small models loop); the refusal tells
    the model to use the earlier result or answer.
  * The step budget bounds the loop; when it runs out the model is told to answer
    with what it has.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from llm import LLM


@dataclass(frozen=True)
class AgentSpec:
    name: str
    system: str
    tools: tuple = ()          # names of MCP tools this agent may use
    max_steps: int = 6         # model calls that may request tools
    max_result_chars: int = 4000


@dataclass
class AgentRun:
    answer: str
    trace: list = field(default_factory=list)   # one entry per tool call


def _tool_schema(tool) -> dict:
    return {"type": "function",
            "function": {"name": tool.name, "description": tool.description or "",
                         "parameters": tool.input_schema}}


def _result_text(result, limit: int) -> str:
    if result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)
    else:
        text = " ".join(getattr(c, "text", "") for c in result.content)
    if len(text) > limit:
        text = text[:limit] + f"... [truncated, {len(text) - limit} more characters]"
    return text


def _parse_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    value = json.loads(raw or "{}")
    if not isinstance(value, dict):
        raise ValueError("arguments must be a JSON object")
    return value


async def run_agent(llm: LLM, client, spec: AgentSpec, task: str) -> AgentRun:
    """Run one agent on `task` using tools from the connected MCP `client`."""
    available = {t.name: t for t in (await client.list_tools()).tools}
    allowed = {name: available[name] for name in spec.tools if name in available}
    missing = set(spec.tools) - set(allowed)
    if missing:
        raise RuntimeError(f"Agent {spec.name!r} needs tools the server lacks: {sorted(missing)}")
    schemas = [_tool_schema(t) for t in allowed.values()]

    messages = [{"role": "system", "content": spec.system},
                {"role": "user", "content": task}]
    trace: list = []
    done: dict = {}            # (tool, arguments) already called in this run

    for _ in range(spec.max_steps):
        reply = await asyncio.to_thread(
            llm.chat_message, messages, **({"tools": schemas} if schemas else {}))
        calls = reply["tool_calls"]
        if not calls:
            return AgentRun(reply["content"], trace)
        messages.append({"role": "assistant", "content": reply["content"], "tool_calls": calls})
        for call in calls:
            function = call.get("function", {})
            name = function.get("name", "")
            entry = {"agent": spec.name, "tool": name, "arguments": None, "error": None}
            trace.append(entry)
            try:
                if name not in allowed:
                    raise ValueError(f"Unknown tool {name!r}. Available: {sorted(allowed)}")
                arguments = _parse_arguments(function.get("arguments"))
                entry["arguments"] = arguments
                key = (name, json.dumps(arguments, sort_keys=True))
                if key in done:
                    raise ValueError("You already made this exact call; its result is above. "
                                     "Use it, or answer now if you have what you need.")
                done[key] = True
                result = await client.call_tool(name, arguments)
                text = _result_text(result, spec.max_result_chars)
                if result.is_error:
                    entry["error"] = text
                    text = f"Error: {text}"
            except (ValueError, json.JSONDecodeError) as error:
                entry["error"] = str(error)
                text = f"Error: {error}"
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": text})

    # Budget spent: ask for an answer without offering tools.
    messages.append({"role": "user", "content":
                     "You are out of tool calls. Answer now using what you have."})
    reply = await asyncio.to_thread(llm.chat_message, messages)
    return AgentRun(reply["content"], trace)