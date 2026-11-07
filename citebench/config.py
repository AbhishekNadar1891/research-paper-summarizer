"""Settings loaded from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at import time
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Default model per provider, used when CITEBENCH_MODEL is not set.
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "ollama": "qwen2.5:7b",
    "openai_compatible": "",
}


@dataclass(frozen=True)
class Settings:
    provider: str              # "anthropic" | "ollama" | "openai_compatible" | "none"
    anthropic_api_key: str | None
    llm_base_url: str | None   # Ollama or any OpenAI-compatible endpoint
    llm_api_key: str | None    # key for an OpenAI-compatible endpoint (Groq, OpenRouter, ...)
    model: str
    top_k: int
    max_per_paper: int
    embedding_model: str
    nli_model: str
    ollama_num_ctx: int

    @classmethod
    def from_env(cls) -> "Settings":
        anthropic_key = os.getenv("ANTHROPIC_API_KEY") or None
        provider = (os.getenv("CITEBENCH_PROVIDER") or ("anthropic" if anthropic_key else "none")).lower()
        return cls(
            provider=provider,
            anthropic_api_key=anthropic_key,
            llm_base_url=os.getenv("CITEBENCH_BASE_URL") or None,
            llm_api_key=os.getenv("CITEBENCH_API_KEY") or None,
            model=os.getenv("CITEBENCH_MODEL") or DEFAULT_MODELS.get(provider, ""),
            top_k=int(os.getenv("CITEBENCH_TOP_K", "12")),
            max_per_paper=int(os.getenv("CITEBENCH_MAX_PER_PAPER", "2")),
            embedding_model=os.getenv("CITEBENCH_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            nli_model=os.getenv("CITEBENCH_NLI_MODEL", "cross-encoder/nli-deberta-v3-small"),
            ollama_num_ctx=int(os.getenv("CITEBENCH_OLLAMA_NUM_CTX", "8192")),
        )

    @property
    def has_llm(self) -> bool:
        if self.provider == "anthropic":
            return bool(self.anthropic_api_key)
        return self.provider in ("ollama", "openai_compatible")


settings = Settings.from_env()
