"""A thin LLM interface so the pipeline can run on Claude, a local Ollama model,
any OpenAI-compatible endpoint (Groq, OpenRouter, DeepSeek, ...), or a fake in tests."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Protocol

from .config import DEFAULT_MODELS, settings


class LLM(Protocol):
    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 4096) -> str: ...


class LLMError(RuntimeError):
    pass


class ClaudeLLM:
    """Anthropic Messages API wrapper."""

    def __init__(self, api_key: str | None = None, model: str | None = None, temperature: float | None = None):
        import anthropic

        key = api_key or settings.anthropic_api_key
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Add it to .env or run in offline mode.")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model or (settings.model if settings.provider == "anthropic" else DEFAULT_MODELS["anthropic"])
        self.temperature = temperature

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 4096) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if self.temperature is not None:  # some models reject it, so only send when asked
            kwargs["temperature"] = self.temperature
        msg = self.client.messages.create(**kwargs)
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


def _post_json(url: str, payload: dict, headers: dict[str, str] | None = None, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        raise LLMError(f"{url} returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Could not reach {url}: {exc.reason}. Is the server running?") from exc


class OllamaLLM:
    """A local model served by Ollama (https://ollama.com), e.g. qwen2.5:7b. Free, no key.

    Uses Ollama's native /api/chat so the context window can be raised: the default
    (2-4k tokens) silently truncates a prompt with a dozen passages.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None, num_ctx: int | None = None,
                 json_mode: bool = True, temperature: float = 0.0):
        self.model = model or settings.model or DEFAULT_MODELS["ollama"]
        self.base_url = (base_url or settings.llm_base_url or "http://localhost:11434").rstrip("/")
        self.base_url = re.sub(r"/v1$", "", self.base_url)
        self.num_ctx = num_ctx or settings.ollama_num_ctx
        self.json_mode = json_mode   # every Citebench prompt asks for JSON
        self.temperature = temperature

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 4096) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"num_ctx": self.num_ctx, "num_predict": max_tokens, "temperature": self.temperature},
        }
        if self.json_mode:
            payload["format"] = "json"
        data = _post_json(f"{self.base_url}/api/chat", payload)
        return str(data.get("message", {}).get("content", ""))


class OpenAICompatibleLLM:
    """Any /v1/chat/completions endpoint: Groq, OpenRouter, DeepSeek, Together, vLLM, LM Studio..."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 json_mode: bool = True, temperature: float = 0.0):
        self.model = model or settings.model
        self.base_url = (base_url or settings.llm_base_url or "").rstrip("/")
        if not self.base_url or not self.model:
            raise LLMError("Set CITEBENCH_BASE_URL and CITEBENCH_MODEL for an OpenAI-compatible provider.")
        self.api_key = api_key or settings.llm_api_key
        self.json_mode = json_mode
        self.temperature = temperature

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 4096) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                   "temperature": self.temperature}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = _post_json(f"{self.base_url}/chat/completions", payload, headers)
        try:
            return str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {str(data)[:200]}") from exc


def get_llm() -> LLM | None:
    """The configured LLM, or None for offline mode.

    CITEBENCH_PROVIDER=anthropic (default when ANTHROPIC_API_KEY is set) | ollama | openai_compatible
    """
    if not settings.has_llm:
        return None
    if settings.provider == "ollama":
        return OllamaLLM()
    if settings.provider == "openai_compatible":
        return OpenAICompatibleLLM()
    return ClaudeLLM()


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> Any:
    """Read a JSON value from an LLM reply: whole reply, a fenced block, or the outermost {...}/[...]."""
    text = text.strip()
    candidates = [text]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    for open_c, close_c in ("{}", "[]"):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
    raise LLMError(f"Could not parse JSON from model reply: {text[:200]!r}")


def complete_json(llm: LLM, prompt: str, *, system: str | None = None, max_tokens: int = 4096) -> Any:
    """Ask for JSON; on a malformed reply, ask once more to repair it."""
    reply = llm.complete(prompt, system=system, max_tokens=max_tokens)
    try:
        return parse_json(reply)
    except LLMError:
        repair = (
            "Your previous reply was not valid JSON. Return the same content as ONE valid JSON value, "
            "with no other text.\n\nPrevious reply:\n" + reply
        )
        return parse_json(llm.complete(repair, system=system, max_tokens=max_tokens))
