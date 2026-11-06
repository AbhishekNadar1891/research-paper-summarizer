"""Truth highlighter: score every answer sentence against the passages it cites.

Natural language inference (NLI) asks whether a premise (a source sentence)
entails, contradicts, or is neutral towards a hypothesis (a claim from the
answer). Unlike word overlap, NLI notices when a claim reuses the source's
words but changes what they say ("reduced errors" vs "did not reduce errors").

Each sentence gets a label and the single source sentence that supports it
best, so the UI can colour the answer and show the evidence on hover.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence

import numpy as np

from .config import settings
from .ingest import split_sentences
from .models import Answer, RetrievedPassage
from .validate import support_overlap

log = logging.getLogger(__name__)
_INLINE = re.compile(r"\[(P\d+)\]")
_INLINE_ANY = re.compile(r"\s*\[\s*P\d+(?:\s*,\s*P\d+)*\s*\]")


@dataclass(frozen=True)
class NLIScore:
    entailment: float
    contradiction: float
    neutral: float


class Scorer(Protocol):
    name: str
    supported_at: float   # entailment at or above this -> supported
    partial_at: float     # entailment at or above this -> partial
    contradicted_at: float

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScore]:
        """Score (premise, hypothesis) pairs."""
        ...


class CrossEncoderNLI:
    """A sentence-transformers NLI cross-encoder, e.g. cross-encoder/nli-deberta-v3-small."""

    supported_at, partial_at, contradicted_at = 0.6, 0.3, 0.5

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self.name = model_name or settings.nli_model
        self.model = CrossEncoder(self.name)
        id2label = getattr(getattr(self.model, "config", None), "id2label", None) or {
            0: "contradiction", 1: "entailment", 2: "neutral"}
        self._idx = {str(v).lower(): int(k) for k, v in id2label.items()}

    def score(self, pairs):
        if not pairs:
            return []
        logits = np.asarray(self.model.predict(list(pairs), show_progress_bar=False), dtype=float)
        logits = np.atleast_2d(logits)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        i = self._idx
        return [NLIScore(entailment=float(p[i["entailment"]]), contradiction=float(p[i["contradiction"]]),
                         neutral=float(p[i["neutral"]])) for p in probs]


_NEGATION = re.compile(r"\b(not|no|never|neither|nor|without|fails?|failed|cannot|didn't|doesn't|isn't|aren't)\b", re.I)


class LexicalScorer:
    """Model-free fallback: word overlap as entailment, negation mismatch as a contradiction hint."""

    name = "lexical-overlap"
    supported_at, partial_at, contradicted_at = 0.55, 0.3, 0.5

    def score(self, pairs):
        out = []
        for premise, hypothesis in pairs:
            ent = support_overlap(hypothesis, premise)
            neg_mismatch = bool(_NEGATION.search(premise)) != bool(_NEGATION.search(hypothesis))
            contra = 0.6 if (neg_mismatch and ent >= 0.5) else 0.0
            if contra:
                ent = min(ent, 0.2)
            out.append(NLIScore(entailment=ent, contradiction=contra, neutral=max(0.0, 1 - ent - contra)))
        return out


def load_scorer(prefer_model: bool = True) -> Scorer:
    """The NLI cross-encoder if it loads (needs sentence-transformers + model download), else lexical."""
    if prefer_model:
        try:
            return CrossEncoderNLI()
        except Exception as exc:  # noqa: BLE001
            log.warning("NLI model unavailable (%s); using lexical scorer.", exc)
    return LexicalScorer()


class Support(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNCITED = "uncited"


@dataclass
class Evidence:
    label: str            # passage label, e.g. "P3"
    sentence: str         # the best-matching source sentence within it
    score: NLIScore


@dataclass
class HighlightedSentence:
    text: str                       # sentence without the [P#] markers
    citations: list[str] = field(default_factory=list)
    support: Support = Support.UNCITED
    entailment: float = 0.0
    best: Evidence | None = None


def classify(s: NLIScore, scorer: Scorer) -> Support:
    if s.contradiction >= scorer.contradicted_at and s.contradiction > s.entailment:
        return Support.CONTRADICTED
    if s.entailment >= scorer.supported_at:
        return Support.SUPPORTED
    if s.entailment >= scorer.partial_at:
        return Support.PARTIAL
    return Support.UNSUPPORTED


def best_support(claim: str, passages: Sequence[RetrievedPassage], scorer: Scorer) -> Evidence | None:
    """Best evidence for `claim` among `passages`: each source sentence, and each whole passage
    (claims often merge two sentences). Strongest entailment wins; contradiction breaks ties."""
    claim = _INLINE_ANY.sub("", claim).strip()
    units: list[tuple[str, str]] = []
    for r in passages:
        sents = split_sentences(r.passage.text)
        units += [(r.label, s) for s in sents]
        if len(sents) > 1:
            units.append((r.label, r.passage.text))
    if not units or not claim:
        return None
    scores = scorer.score([(u[1], claim) for u in units])
    best_i = max(range(len(units)), key=lambda i: (scores[i].entailment, -scores[i].contradiction))
    worst_c = max(range(len(units)), key=lambda i: scores[i].contradiction)
    i = best_i
    if scores[worst_c].contradiction >= scorer.contradicted_at and scores[best_i].entailment < scorer.partial_at:
        i = worst_c  # nothing supports it and something contradicts it: show the contradiction
    label, text = units[i]
    if text == next(r.passage.text for r in passages if r.label == label):
        # Whole passage won; point at its closest single sentence for display.
        sents = split_sentences(text)
        text = max(sents, key=lambda s: support_overlap(claim, s))
    return Evidence(label=label, sentence=text, score=scores[i])


def highlight_text(text: str, retrieved: Sequence[RetrievedPassage], scorer: Scorer,
                   default_citations: Sequence[str] = ()) -> list[HighlightedSentence]:
    """Split `text` into sentences and label each by how well its cited passages support it.
    `default_citations` apply to sentences without inline [P#] markers (e.g. a finding's citations)."""
    by_label = {r.label: r for r in retrieved}
    out = []
    for sent in split_sentences(text):
        cites = list(dict.fromkeys(_INLINE.findall(sent))) or list(default_citations)
        clean = _INLINE_ANY.sub("", sent).strip()
        hs = HighlightedSentence(text=clean, citations=cites)
        cited = [by_label[c] for c in cites if c in by_label]
        if cited:
            ev = best_support(clean, cited, scorer)
            if ev:
                hs.best, hs.entailment, hs.support = ev, ev.score.entailment, classify(ev.score, scorer)
        elif cites:
            hs.support = Support.UNSUPPORTED  # cites only labels that don't exist
        out.append(hs)
    return out


@dataclass
class HighlightedAnswer:
    summary: list[HighlightedSentence]
    findings: dict[str, list[HighlightedSentence]]
    scorer: str

    def counts(self) -> dict[str, int]:
        all_s = self.summary + [s for v in self.findings.values() for s in v]
        return {k.value: sum(s.support is k for s in all_s) for k in Support}


def highlight_answer(answer: Answer, retrieved: Sequence[RetrievedPassage], scorer: Scorer | None = None) -> HighlightedAnswer:
    scorer = scorer or load_scorer()
    return HighlightedAnswer(
        summary=highlight_text(answer.summary, retrieved, scorer),
        findings={f.paper_key: highlight_text(f.finding, retrieved, scorer, default_citations=f.citations)
                  for f in answer.findings},
        scorer=scorer.name,
    )
