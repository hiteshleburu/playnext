"""Connecting to the playnext MCP server and calling its tools."""

import os
import sys
from pathlib import Path

# The MCP client gives a subprocess only a few safe env vars, so pass these explicitly.
SERVER_ENV_VARS = ("TAVILY_API_KEY", "SEARXNG_URL", "WEB_SEARCH_CACHE", "PLAYNEXT_ALLOW_ADULT")


class PipelineError(RuntimeError):
    """The request cannot be answered (e.g. none of the liked games exist)."""


async def call(client, tool: str, **arguments):
    """Call an MCP tool and return its structured result; raise PipelineError on failure."""
    result = await client.call_tool(tool, arguments)
    if result.is_error:
        raise PipelineError(f"{tool} failed: " + " ".join(getattr(c, "text", "") for c in result.content))
    return result.structured_content


def server_target(data_path, server_url=None):
    """A running server's URL, or a subprocess running mcp_server.py over stdio."""
    from mcp import StdioServerParameters

    if server_url:
        return server_url
    env = {name: os.environ[name] for name in SERVER_ENV_VARS if name in os.environ}
    return StdioServerParameters(
        command=sys.executable, env=env or None,
        args=[str(Path(__file__).resolve().parent / "mcp_server.py"), "--data", str(data_path)])


async def with_client(target, work):
    """Connect to the server and `await work(client)`. The MCP client wraps errors in an
    ExceptionGroup; this unwraps it so callers can catch PipelineError / LLMError."""
    from mcp.client import Client

    try:
        async with Client(target) as client:
            return await work(client)
    except BaseExceptionGroup as group:
        error = group
        while isinstance(error, BaseExceptionGroup) and error.exceptions:
            error = error.exceptions[0]
        raise error from None