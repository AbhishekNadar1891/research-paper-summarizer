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


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    model: str
    top_k: int
    max_per_paper: int
    embedding_model: str
    nli_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=os.getenv("CITEBENCH_MODEL", "claude-haiku-5-5"),
            top_k=int(os.getenv("CITEBENCH_TOP_K", "12")),
            max_per_paper=int(os.getenv("CITEBENCH_MAX_PER_PAPER", "2")),
            embedding_model=os.getenv("CITEBENCH_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            nli_model=os.getenv("CITEBENCH_NLI_MODEL", "cross-encoder/nli-deberta-v3-small"),
        )

    @property
    def has_llm(self) -> bool:
        return bool(self.anthropic_api_key)


settings = Settings.from_env()
