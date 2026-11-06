import json

import pytest

from citebench.ingest import load_starter
from citebench.retrieval import HybridRetriever


class FakeLLM:
    """Returns queued replies in order and records every prompt it was sent."""

    def __init__(self, *replies):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.prompts: list[str] = []

    def complete(self, prompt, *, system=None, max_tokens=4096):
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("FakeLLM ran out of replies")
        return self.replies.pop(0)


@pytest.fixture(scope="session")
def collection():
    return load_starter()


@pytest.fixture(scope="session")
def retriever(collection):
    return HybridRetriever(collection, use_embeddings=False)


@pytest.fixture
def fake_llm():
    return FakeLLM
