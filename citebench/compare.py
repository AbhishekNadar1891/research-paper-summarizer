"""With docs vs. without docs: what the model says from memory, checked against the papers.

The same question is answered closed-book (no passages). That answer is
broken into atomic claims, in the spirit of FActScore, and each claim is
checked with the NLI scorer against every retrieved passage:

  backed          a passage entails it
  partial         a passage partly supports it
  not_in_papers   no retrieved passage supports it (unverifiable here, not necessarily false)
  contradicted    a passage contradicts it

Shown beside the grounded answer, this makes the value of retrieval visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from .ingest import split_sentences
from .llm import LLM, complete_json
from .models import Answer, RetrievedPassage
from .nli import Evidence, Scorer, Support, best_support, classify, load_scorer

CLOSED_BOOK_SYSTEM = "You are a knowledgeable research assistant answering from memory."

CLOSED_BOOK_PROMPT = """Answer this research question from your own knowledge. You have no documents.
Be specific: name methods, papers and numbers where you know them.

QUESTION: {question}

Return only a JSON object:
{{"answer": "a 4-6 sentence answer",
 "claims": ["one short, self-contained factual claim from your answer", "..."]}}

Each claim must make sense on its own (no "it" or "this method"), and together they must cover every
factual statement in the answer. 5-12 claims."""


class ClaimStatus(str, Enum):
    BACKED = "backed"
    PARTIAL = "partial"
    NOT_IN_PAPERS = "not_in_papers"
    CONTRADICTED = "contradicted"


_FROM_SUPPORT = {
    Support.SUPPORTED: ClaimStatus.BACKED,
    Support.PARTIAL: ClaimStatus.PARTIAL,
    Support.UNSUPPORTED: ClaimStatus.NOT_IN_PAPERS,
    Support.CONTRADICTED: ClaimStatus.CONTRADICTED,
    Support.UNCITED: ClaimStatus.NOT_IN_PAPERS,
}


@dataclass
class ClaimCheck:
    claim: str
    status: ClaimStatus
    evidence: Evidence | None = None


@dataclass
class Comparison:
    question: str
    closed_book_answer: str
    claims: list[ClaimCheck] = field(default_factory=list)
    grounded: Answer | None = None
    papers_unused_by_memory: list[str] = field(default_factory=list)  # cited with docs, never matched from memory
    scorer: str = ""

    def count(self, status: ClaimStatus) -> int:
        return sum(c.status is status for c in self.claims)

    @property
    def backed_rate(self) -> float:
        """Share of memory-only claims that a retrieved passage backs (fully or partly)."""
        if not self.claims:
            return 0.0
        return (self.count(ClaimStatus.BACKED) + self.count(ClaimStatus.PARTIAL)) / len(self.claims)

    def stats(self) -> dict[str, float | int]:
        return {**{s.value: self.count(s) for s in ClaimStatus},
                "claims": len(self.claims), "backed_rate": round(self.backed_rate, 3),
                "papers_unused_by_memory": len(self.papers_unused_by_memory)}


def extract_claims(answer_text: str, claims: Sequence[str] | None = None) -> list[str]:
    """Use the model's own atomic claims when given, else fall back to the answer's sentences."""
    cleaned = [str(c).strip() for c in (claims or []) if str(c).strip()]
    return cleaned or split_sentences(answer_text)


ClaimRetriever = Callable[[str], Sequence[RetrievedPassage]]


def _merge(base: Sequence[RetrievedPassage], extra: Sequence[RetrievedPassage]) -> list[RetrievedPassage]:
    """Union by passage, keeping the base labels and labelling new passages after them."""
    seen = {r.passage.uid for r in base}
    out = list(base)
    n = len(base)
    for r in extra:
        if r.passage.uid not in seen:
            seen.add(r.passage.uid)
            n += 1
            out.append(type(r)(label=f"P{n}", passage=r.passage, paper=r.paper, score=r.score))
    return out


def check_claims(claims: Sequence[str], retrieved: Sequence[RetrievedPassage], scorer: Scorer,
                 claim_retriever: ClaimRetriever | None = None) -> list[ClaimCheck]:
    """Check each claim against the question's passages, plus (optionally) passages retrieved for
    that claim itself, so "not in papers" means not in the collection, not just "not in the top k"."""
    out = []
    for claim in claims:
        pool = _merge(retrieved, claim_retriever(claim)) if claim_retriever else list(retrieved)
        ev = best_support(claim, pool, scorer) if pool else None
        status = _FROM_SUPPORT[classify(ev.score, scorer)] if ev else ClaimStatus.NOT_IN_PAPERS
        out.append(ClaimCheck(claim=claim, status=status, evidence=ev))
    return out


def closed_book(question: str, llm: LLM) -> tuple[str, list[str]]:
    data = complete_json(llm, CLOSED_BOOK_PROMPT.format(question=question), system=CLOSED_BOOK_SYSTEM)
    if not isinstance(data, dict):
        data = {}
    answer = str(data.get("answer", "")).strip()
    claims = data.get("claims") if isinstance(data.get("claims"), list) else None
    return answer, extract_claims(answer, claims)


def compare(question: str, retrieved: Sequence[RetrievedPassage], llm: LLM | None,
            scorer: Scorer | None = None, grounded: Answer | None = None,
            claim_retriever: ClaimRetriever | None = None) -> Comparison | None:
    """Run the closed-book answer and check it against the papers. Needs an LLM; returns None without one."""
    if llm is None:
        return None
    scorer = scorer or load_scorer()
    answer, claims = closed_book(question, llm)
    checks = check_claims(claims, retrieved, scorer, claim_retriever)

    unused: list[str] = []
    if grounded is not None:
        by_label = {r.label: r.paper.key for r in retrieved}
        matched = {c.evidence.paper_key for c in checks
                   if c.evidence and c.status in (ClaimStatus.BACKED, ClaimStatus.PARTIAL)}
        cited = [f.paper_key for f in grounded.findings if f.paper_key in by_label.values()]
        unused = [k for k in dict.fromkeys(cited) if k not in matched]

    return Comparison(question=question, closed_book_answer=answer, claims=checks, grounded=grounded,
                      papers_unused_by_memory=unused, scorer=scorer.name)
