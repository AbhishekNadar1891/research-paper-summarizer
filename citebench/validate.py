"""Citation validation: check every citation the LLM wrote against the passages it was given.

Grounding instructions reduce hallucination but do not remove it. Models
still cite passages that were never retrieved, attach one paper's passage to
another paper's finding, cite a real passage that says something else, and
call a single study "strong evidence". All of that is checkable in code.

Checks
  invented       label not among the retrieved passages            (error)
  misattributed  finding for paper A cites a passage from paper B  (error)
  unknown_paper  finding about a paper that was not retrieved      (error)
  weak           claim shares little wording with its passage      (warning)
  uncited        a finding, or a factual summary sentence, has no citation (warning)
  downgraded     evidence strength above what its distinct papers allow (warning, auto-fixed)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .ingest import split_sentences
from .models import Answer, RetrievedPassage, Strength
from .retrieval import tokenize

_INLINE = re.compile(r"\[(P\d+)\]")
_INLINE_ANY = re.compile(r"\[\s*P\d+(?:\s*,\s*P\d+)*\s*\]")


class IssueKind(str, Enum):
    INVENTED = "invented"
    MISATTRIBUTED = "misattributed"
    UNKNOWN_PAPER = "unknown_paper"
    WEAK = "weak"
    UNCITED = "uncited"
    DOWNGRADED = "downgraded"

    @property
    def is_error(self) -> bool:
        return self in (IssueKind.INVENTED, IssueKind.MISATTRIBUTED, IssueKind.UNKNOWN_PAPER)


@dataclass
class Issue:
    kind: IssueKind
    location: str          # "summary", "finding:<paper>", "evidence[0]", "tension[1]"
    message: str
    label: str | None = None


@dataclass
class CitationCheck:
    location: str
    label: str
    status: str            # "ok" | "weak" | "invented" | "misattributed"
    overlap: float | None = None


@dataclass
class ValidationReport:
    checks: list[CitationCheck] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def _count(self, status: str) -> int:
        return sum(c.status == status for c in self.checks)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def resolved(self) -> int:
        """Citations that point at a real retrieved passage from the right paper (ok + weak)."""
        return self._count("ok") + self._count("weak")

    @property
    def invented(self) -> int:
        return self._count("invented")

    @property
    def misattributed(self) -> int:
        return self._count("misattributed")

    @property
    def weak(self) -> int:
        return self._count("weak")

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.kind.is_error]

    @property
    def ok(self) -> bool:
        return not self.errors

    def status_of(self, location: str, label: str) -> str | None:
        for c in self.checks:
            if c.location == location and c.label == label:
                return c.status
        return None

    def summary(self) -> dict[str, int]:
        return {"total": self.total, "resolved": self.resolved, "weak": self.weak,
                "invented": self.invented, "misattributed": self.misattributed,
                "issues": len(self.issues)}


def support_overlap(claim: str, passage: str) -> float:
    """Share of the claim's content words that also appear in the passage (0..1)."""
    claim_terms = set(tokenize(_INLINE_ANY.sub("", claim)))
    if not claim_terms:
        return 1.0
    return len(claim_terms & set(tokenize(passage))) / len(claim_terms)


class _Checker:
    def __init__(self, retrieved: list[RetrievedPassage], weak_threshold: float):
        self.by_label = {r.label: r for r in retrieved}
        self.paper_keys = {r.paper.key for r in retrieved}
        self.weak_threshold = weak_threshold
        self.report = ValidationReport()

    def cite(self, location: str, label: str, claim: str | None, paper_key: str | None = None) -> bool:
        """Check one citation; record it; return True if it resolves to a valid passage."""
        r = self.by_label.get(label)
        rep = self.report
        if r is None:
            rep.checks.append(CitationCheck(location, label, "invented"))
            rep.issues.append(Issue(IssueKind.INVENTED, location,
                                    f"Cites {label}, which is not one of the retrieved passages.", label))
            return False
        if paper_key and r.paper.key != paper_key:
            rep.checks.append(CitationCheck(location, label, "misattributed"))
            rep.issues.append(Issue(IssueKind.MISATTRIBUTED, location,
                                    f"{label} is from {r.paper.citation_label()}, not the paper this finding is about.",
                                    label))
            return False
        overlap = support_overlap(claim, r.passage.text) if claim else None
        if overlap is not None and overlap < self.weak_threshold:
            rep.checks.append(CitationCheck(location, label, "weak", overlap))
            rep.issues.append(Issue(IssueKind.WEAK, location,
                                    f"The claim shares little wording with {label} ({overlap:.0%}); "
                                    "read the passage to confirm it supports the claim.", label))
        else:
            rep.checks.append(CitationCheck(location, label, "ok", overlap))
        return True


def validate(answer: Answer, retrieved: list[RetrievedPassage], *, weak_threshold: float = 0.2,
             fix_strength: bool = True) -> ValidationReport:
    """Check all citations in `answer`. With fix_strength, inflated evidence ratings are lowered in place."""
    ck = _Checker(retrieved, weak_threshold)
    rep = ck.report

    # Summary: each inline [P#] is checked against the sentence it sits in.
    for sentence in split_sentences(answer.summary):
        labels = _INLINE.findall(sentence)
        for label in dict.fromkeys(labels):
            ck.cite("summary", label, sentence)
        if not labels and len(tokenize(sentence)) >= 6 and answer.grounded and answer.mode != "extractive":
            rep.issues.append(Issue(IssueKind.UNCITED, "summary",
                                    f'Summary sentence has no citation: "{sentence[:80]}"'))

    # Per-paper findings: must cite their own paper.
    for f in answer.findings:
        loc = f"finding:{f.paper_key}"
        if f.paper_key not in ck.paper_keys:
            rep.issues.append(Issue(IssueKind.UNKNOWN_PAPER, loc,
                                    f'Finding is about "{f.paper_key}", which was not among the retrieved papers.'))
        if not f.citations:
            rep.issues.append(Issue(IssueKind.UNCITED, loc, "Finding has no citation."))
        for label in f.citations:
            ck.cite(loc, label, f.finding, paper_key=f.paper_key if f.paper_key in ck.paper_keys else None)

    # Cross-paper evidence: strength must match the number of distinct papers actually cited.
    for i, e in enumerate(answer.evidence):
        loc = f"evidence[{i}]"
        valid_papers = set()
        for label in e.citations:
            if ck.cite(loc, label, None):
                valid_papers.add(ck.by_label[label].paper.key)
        n = len(valid_papers)
        if not e.citations:
            rep.issues.append(Issue(IssueKind.UNCITED, loc, "Evidence claim has no citation."))
        if n < e.strength.min_papers:
            allowed = Strength.from_paper_count(n)
            rep.issues.append(Issue(IssueKind.DOWNGRADED, loc,
                                    f'Rated "{e.strength.value}" but cites {n} distinct paper'
                                    f'{"" if n == 1 else "s"}; '
                                    f'{"lowered" if fix_strength else "should be"} "{allowed.value}".'))
            if fix_strength and allowed is not e.strength:
                e.strength = allowed
                e.downgraded = True

    for i, t in enumerate(answer.tensions):
        for label in t.citations:
            ck.cite(f"tension[{i}]", label, None)

    return rep


def drop_invalid_citations(answer: Answer, report: ValidationReport) -> Answer:
    """Remove invented and misattributed labels from the answer (in place) so the UI never links them."""
    bad = {(c.location, c.label) for c in report.checks if c.status in ("invented", "misattributed")}
    for f in answer.findings:
        f.citations = [c for c in f.citations if (f"finding:{f.paper_key}", c) not in bad]
    for i, e in enumerate(answer.evidence):
        e.citations = [c for c in e.citations if (f"evidence[{i}]", c) not in bad]
    for i, t in enumerate(answer.tensions):
        t.citations = [c for c in t.citations if (f"tension[{i}]", c) not in bad]
    bad_summary = {lbl for loc, lbl in bad if loc == "summary"}
    if bad_summary:
        answer.summary = _INLINE.sub(lambda m: "" if m.group(1) in bad_summary else m.group(0), answer.summary)
        answer.summary = re.sub(r"\s+([.,;])", r"\1", re.sub(r"  +", " ", answer.summary)).strip()
    return answer
