"""End-to-end pipeline:

    retrieve -> generate draft -> validate -> Reviewer 2 (critique + revise) -> validate again
             -> drop invalid citations -> truth-highlight -> compare with a memory-only answer
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

from .compare import Comparison, compare
from .generate import generate_grounded
from .ingest import Collection, load_starter
from .llm import LLM, get_llm
from .models import Answer, RetrievedPassage
from .nli import HighlightedAnswer, Scorer, highlight_answer, load_scorer
from .retrieval import HybridRetriever
from .review import ReviewResult, review
from .validate import ValidationReport, drop_invalid_citations, validate

log = logging.getLogger(__name__)

_AUTO = object()


@dataclass
class PipelineResult:
    question: str
    retrieved: list[RetrievedPassage]
    answer: Answer                       # final answer shown to the user
    report: ValidationReport             # citation check of the final answer
    draft: Answer | None = None          # before Reviewer 2 (None if review was skipped)
    draft_report: ValidationReport | None = None
    review: ReviewResult | None = None
    highlights: HighlightedAnswer | None = None
    comparison: Comparison | None = None
    timings: dict[str, float] = field(default_factory=dict)
    retrieval_mode: str = "bm25"
    llm_used: bool = False


class Pipeline:
    def __init__(self, collection: Collection | None = None, llm: LLM | None | object = _AUTO,
                 scorer: Scorer | None = None, use_embeddings: bool = True,
                 retriever: HybridRetriever | None = None):
        self.collection = collection or load_starter()
        self.llm: LLM | None = get_llm() if llm is _AUTO else llm  # type: ignore[assignment]
        self.retriever = retriever or HybridRetriever(self.collection, use_embeddings=use_embeddings)
        self._scorer = scorer

    @property
    def scorer(self) -> Scorer:
        if self._scorer is None:
            self._scorer = load_scorer()
        return self._scorer

    def rebuild_index(self, use_embeddings: bool = True) -> None:
        """Call after adding or removing papers."""
        self.retriever = HybridRetriever(self.collection, use_embeddings=use_embeddings)

    def run(self, question: str, *, with_review: bool = True, with_highlights: bool = True,
            with_comparison: bool = True, top_k: int | None = None) -> PipelineResult:
        t: dict[str, float] = {}

        def timed(name, fn):
            start = time.perf_counter()
            out = fn()
            t[name] = round(time.perf_counter() - start, 3)
            return out

        retrieved = timed("retrieve", lambda: self.retriever.retrieve(question, top_k=top_k))
        draft = timed("generate", lambda: generate_grounded(question, retrieved, self.llm))
        draft_report = timed("validate", lambda: validate(draft, retrieved))

        final, rev = draft, None
        if with_review:
            rev = timed("review", lambda: review(copy.deepcopy(draft), retrieved, draft_report, self.llm))
            if rev.revised is not None:
                final = rev.revised
        report = draft_report if final is draft else timed("revalidate", lambda: validate(final, retrieved))
        final = drop_invalid_citations(copy.deepcopy(final), report)

        highlights = timed("highlight", lambda: highlight_answer(final, retrieved, self.scorer)) \
            if with_highlights else None

        comparison = None
        if with_comparison and self.llm is not None:
            comparison = timed("compare", lambda: compare(
                question, retrieved, self.llm, scorer=self.scorer, grounded=final,
                claim_retriever=lambda claim: self.retriever.retrieve(claim, top_k=3)))

        return PipelineResult(
            question=question, retrieved=retrieved, answer=final, report=report,
            draft=draft if with_review else None, draft_report=draft_report if with_review else None,
            review=rev, highlights=highlights, comparison=comparison, timings=t,
            retrieval_mode=self.retriever.mode, llm_used=self.llm is not None,
        )
