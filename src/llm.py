"""
llm.py - a minimal client for any OpenAI-compatible chat endpoint.

One small class, standard library only. Pointing it somewhere else is a matter of
environment variables, so switching between a local model and a hosted one needs
no code change:

    LLM_BASE_URL   default http://localhost:11434/v1   (Ollama)
    LLM_MODEL      default llama3.1:8b
    LLM_API_KEY    default "ollama" (Ollama ignores it; hosted providers need a real key)

Examples
    OpenRouter  LLM_BASE_URL=https://openrouter.ai/api/v1  LLM_MODEL=<model id>  LLM_API_KEY=...
    Gemini      LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ ...

Ollama note: the context length is a server setting, not a request one. If prompts are
longer than it, Ollama silently cuts them off. Set it in a Modelfile
(PARAMETER num_ctx 8192) or with the OLLAMA_CONTEXT_LENGTH environment variable.

Everything an LLM returns is treated as untrusted text by the callers.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.1:8b"


class LLMError(RuntimeError):
    """The model could not be reached, or did not answer in a usable way."""


def strip_thinking(text: str) -> str:
    """Drop <think>...</think> blocks that reasoning models put before the answer."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def extract_json(text: str) -> dict:
    """Parse the JSON object in a model reply, tolerating code fences and chatter."""
    text = strip_thinking(text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object, got {type(value).__name__}")
    return value


@dataclass
class LLM:
    base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", "ollama"))
    timeout: float = 300.0        # seconds; a CPU-only machine can be slow
    temperature: float = 0.0

    def chat(self, messages: list, **extra) -> str:
        """Send messages, return the reply text (thinking blocks removed)."""
        return self.chat_message(messages, **extra)["content"]

    def chat_message(self, messages: list, **extra) -> dict:
        """Like chat, but returns {"content": str, "tool_calls": list}. Pass
        tools=[...] (OpenAI function schemas) to let the model ask for tool calls."""
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, **extra}
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
            message = data["choices"][0]["message"]
            return {"content": strip_thinking(message.get("content")),
                    "tool_calls": message.get("tool_calls") or []}
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            raise LLMError(f"The model server answered HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LLMError(
                f"Could not reach the model at {self.base_url} ({error}). For Ollama: is it "
                f"running, and did you `ollama pull {self.model}`?") from error
        except (KeyError, IndexError, ValueError) as error:
            raise LLMError(f"Unexpected reply from the model server: {error}") from error

    def json(self, messages: list, retries: int = 1) -> dict:
        """Ask for a JSON object and return it parsed. Retries once if the reply is not
        valid JSON, which small models occasionally do."""
        last_error = None
        for _ in range(retries + 1):
            text = self.chat(messages, response_format={"type": "json_object"})
            try:
                return extract_json(text)
            except ValueError as error:
                last_error = error
        raise LLMError(f"The model did not return valid JSON: {last_error}")