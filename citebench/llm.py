"""A thin LLM interface so the pipeline can run on Claude, or on a fake in tests."""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .config import settings


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
        self.model = model or settings.model
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


def get_llm() -> LLM | None:
    """Claude if an API key is configured, otherwise None (callers fall back to offline mode)."""
    return ClaudeLLM() if settings.has_llm else None


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
