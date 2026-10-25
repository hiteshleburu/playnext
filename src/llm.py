"""A small client for any OpenAI-compatible chat endpoint (Ollama, OpenRouter, ...).

Pick the provider with --provider or LLM_PROVIDER (default ollama). OpenRouter needs
OPENROUTER_API_KEY and uses OPENROUTER_MODEL if set. LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
override either provider, and constructor arguments (--model) override everything.
Ollama silently cuts prompts longer than its context length (set num_ctx or
OLLAMA_CONTEXT_LENGTH). Hosted providers receive whatever is sent."""

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROVIDERS = {
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.1:8b", "api_key": "ollama"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.1-8b-instruct",
                   "model_env": "OPENROUTER_MODEL", "key_env": "OPENROUTER_API_KEY"},
}


class LLMError(RuntimeError):
    """The model could not be reached, or did not answer in a usable way."""


def load_dotenv(path=None) -> None:
    """Load KEY=VALUE lines from .env (working directory, then repo root) into os.environ.
    Variables already set are never overridden."""
    for file in [Path(path)] if path else [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if not file.is_file():
            continue
        for line in file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def strip_thinking(text: str) -> str:
    """Drop <think>...</think> blocks that reasoning models put before the answer."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def extract_json(text: str) -> dict:
    """The JSON object in a model reply, tolerating code fences and chatter."""
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
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout: float = 300.0
    temperature: float = 0.0

    def __post_init__(self):
        self.provider = (self.provider or os.environ.get("LLM_PROVIDER") or "ollama").lower()
        if self.provider not in PROVIDERS:
            raise LLMError(f"Unknown LLM provider {self.provider!r}. Choose from: {', '.join(PROVIDERS)}.")
        preset, env = PROVIDERS[self.provider], os.environ.get
        self.base_url = self.base_url or env("LLM_BASE_URL") or preset["base_url"]
        self.model = self.model or env("LLM_MODEL") or env(preset.get("model_env", "")) or preset["model"]
        self.api_key = self.api_key or env("LLM_API_KEY") or env(preset.get("key_env", "")) or preset.get("api_key")
        if not self.api_key:
            raise LLMError(f"No API key for {self.provider}: set {preset['key_env']} (shell or .env file).")

    def describe(self) -> str:
        return f"{self.provider}: {self.model}"

    def chat_message(self, messages: list, **extra) -> dict:
        """Send messages; return {"content": text, "tool_calls": list}. Pass tools=[...]
        (OpenAI function schemas) to let the model ask for tool calls."""
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, **extra}
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                message = json.load(response)["choices"][0]["message"]
        except urllib.error.HTTPError as error:
            hint = f" Check the API key for {self.provider}." if error.code in (401, 403) else ""
            detail = error.read().decode(errors="replace")[:300]
            raise LLMError(f"The model server answered HTTP {error.code}: {detail}{hint}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            advice = (f"For Ollama: is it running, and did you `ollama pull {self.model}`?"
                      if self.provider == "ollama" else "Check your network connection.")
            raise LLMError(f"Could not reach the model at {self.base_url} ({error}). {advice}") from error
        except (KeyError, IndexError, ValueError) as error:
            raise LLMError(f"Unexpected reply from the model server: {error}") from error
        return {"content": strip_thinking(message.get("content")),
                "tool_calls": message.get("tool_calls") or []}

    def chat(self, messages: list, **extra) -> str:
        return self.chat_message(messages, **extra)["content"]

    def json(self, messages: list, retries: int = 1) -> dict:
        """Ask for a JSON object and parse it; retry if a small model returns invalid JSON."""
        for _ in range(retries + 1):
            try:
                return extract_json(self.chat(messages, response_format={"type": "json_object"}))
            except ValueError as error:
                last_error = error
        raise LLMError(f"The model did not return valid JSON: {last_error}")