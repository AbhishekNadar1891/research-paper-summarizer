"""Grounded generation: turn retrieved passages into a cited, evidence-rated answer.

The LLM is anchored to the passages: it sees only them, must cite their
labels (P1, P2, ...), and must keep "what one paper found" separate from
"what several papers together support". Its output is still checked later
by citebench.validate, because instructions alone do not stop hallucinated
citations.
"""
from __future__ import annotations

import re
from typing import Any

from .ingest import split_sentences
from .llm import LLM, complete_json
from .models import (
    Answer,
    EvidenceClaim,
    Family,
    Finding,
    RetrievedPassage,
    Strength,
    StudyType,
    Tension,
)
from .retrieval import expand_query, tokenize

SYSTEM = (
    "You are a careful research assistant. You answer only from the passages you are given, "
    "cite them precisely, and never add facts from memory."
)

GROUNDED_PROMPT = """QUESTION: {question}

PASSAGES:
{passages}

Return only a JSON object with this shape:
{{"summary": "3-5 sentence direct answer. Put a citation like [P2] after every claim.",
 "findings": [{{"paper": "<paper key exactly as given>", "finding": "what THIS paper found, 1-2 sentences", "citations": ["P1"], "study_type": "method|evaluation|analysis|theory|survey"}}],
 "evidence": [{{"claim": "a conclusion supported across papers", "citations": ["P1", "P4"], "strength": "strong|moderate|limited", "why": "one sentence explaining the rating"}}],
 "tensions": [{{"point": "a disagreement, trade-off or caveat between papers", "citations": ["P3"]}}],
 "gaps": ["part of the question these passages do not answer"]}}

Rules:
- Use ONLY the passages. Cite only the labels listed above.
- A paper's finding may cite only that paper's own passages.
- One "findings" entry per relevant paper.
- strength: "strong" needs 3+ independent papers with consistent results, "moderate" 2, "limited" 1.
  A benchmark that only measures the problem is not evidence that a fix works.
- Report numbers exactly as written in the passages.
- If the passages do not answer the question, say so in the summary and list it under gaps."""

# Used by the offline (extractive) mode to describe what a group of papers supports.
FAMILY_CLAIMS: dict[Family, tuple[str, str]] = {
    Family.RETRIEVAL: ("Retrieval / grounding",
                       "Grounding generation in retrieved documents reduces hallucination compared with relying on model weights alone."),
    Family.VERIFICATION: ("Verification / revision",
                          "Checking a draft against verification questions or retrieved evidence, then revising it, reduces unsupported claims."),
    Family.INFERENCE: ("Decoding / activations",
                       "Changing decoding or steering internal activations at inference time, without new training data, improves truthfulness benchmarks."),
    Family.DETECTION: ("Detection / uncertainty",
                       "A model's own uncertainty or self-consistency can flag likely hallucinations for checking."),
    Family.TRAINING: ("Training / fine-tuning",
                      "What and how a model is fine-tuned changes how often it hallucinates, in both directions."),
    Family.MEASUREMENT: ("Benchmark / measurement",
                         "Measured hallucination remains substantial even in strong models."),
    Family.FRAMING: ("Theory / survey",
                     "Some hallucination of rare facts is expected from how models are trained, so mitigation is needed beyond pretraining."),
}


def format_passages(retrieved: list[RetrievedPassage]) -> str:
    blocks = []
    for r in retrieved:
        p = r.paper
        blocks.append(
            f"[{r.label}] paper={p.key} | {p.authors} ({p.year}) \"{p.title}\" | study type: {p.study_type.value}\n"
            f"{r.passage.text}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, retrieved: list[RetrievedPassage]) -> str:
    return GROUNDED_PROMPT.format(question=question, passages=format_passages(retrieved))


# --------------------------------------------------------------------------
# Parsing the model's JSON into typed objects
# --------------------------------------------------------------------------
_LABEL = re.compile(r"^P\d+$")


def _labels(value: Any) -> list[str]:
    if isinstance(value, str):
        value = re.findall(r"P\d+", value)
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        s = str(v).strip().strip("[]")
        if _LABEL.match(s) and s not in out:
            out.append(s)
    return out


def _strength(value: Any) -> Strength:
    try:
        return Strength(str(value).lower().strip())
    except ValueError:
        return Strength.LIMITED


def _study_type(value: Any) -> StudyType | None:
    try:
        return StudyType(str(value).lower().strip())
    except ValueError:
        return None


def answer_from_json(question: str, data: Any) -> Answer:
    """Coerce the model's JSON into an Answer. Keeps citation labels as written (validation judges them)."""
    if not isinstance(data, dict):
        data = {}
    findings = [
        Finding(paper_key=str(f.get("paper", "")).strip(), finding=str(f.get("finding", "")).strip(),
                citations=_labels(f.get("citations")), study_type=_study_type(f.get("study_type")))
        for f in data.get("findings") or [] if isinstance(f, dict) and f.get("finding")
    ]
    evidence = [
        EvidenceClaim(claim=str(e.get("claim", "")).strip(), citations=_labels(e.get("citations")),
                      strength=_strength(e.get("strength")), why=str(e.get("why", "")).strip())
        for e in data.get("evidence") or [] if isinstance(e, dict) and e.get("claim")
    ]
    tensions = [
        Tension(point=str(t.get("point", "")).strip(), citations=_labels(t.get("citations")))
        for t in data.get("tensions") or [] if isinstance(t, dict) and t.get("point")
    ]
    gaps = [str(g).strip() for g in data.get("gaps") or [] if str(g).strip()]
    return Answer(question=question, summary=str(data.get("summary", "")).strip(), findings=findings,
                  evidence=evidence, tensions=tensions, gaps=gaps, grounded=True, mode="llm")


# --------------------------------------------------------------------------
# Offline mode
# --------------------------------------------------------------------------
def extractive_answer(question: str, retrieved: list[RetrievedPassage]) -> Answer:
    """No-LLM fallback: best-matching sentence per paper, evidence grouped by approach family."""
    if not retrieved:
        return Answer(question=question, summary="No passages in the collection match this question.",
                      gaps=["The collection has no relevant papers. Add papers and try again."], mode="extractive")
    q = expand_query(tokenize(question))

    def sentence_score(s: str) -> float:
        toks = tokenize(s)
        return sum(q.get(t, 0) for t in toks) / (len(toks) + 1) ** 0.5 + (0.15 if re.search(r"\d", s) else 0)

    by_paper: dict[str, list[RetrievedPassage]] = {}
    for r in retrieved:
        by_paper.setdefault(r.paper.key, []).append(r)

    findings = []
    for key, items in by_paper.items():
        best = max(((sentence_score(s), s, r.label) for r in items for s in split_sentences(r.passage.text)),
                   key=lambda x: x[0])
        findings.append(Finding(paper_key=key, finding=best[1], citations=[best[2]],
                                study_type=items[0].paper.study_type))

    groups: dict[Family, list[RetrievedPassage]] = {}
    for key, items in by_paper.items():
        groups.setdefault(items[0].paper.family, []).append(items[0])
    evidence = []
    for fam, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        label, claim = FAMILY_CLAIMS[fam]
        n = len(items)
        evidence.append(EvidenceClaim(claim=claim, citations=[r.label for r in items],
                                      strength=Strength.from_paper_count(n),
                                      why=f'{n} paper{"s" if n > 1 else ""} in the "{label}" group were retrieved.'))

    methods = [FAMILY_CLAIMS[f][0].lower() for f in groups if f not in (Family.MEASUREMENT, Family.FRAMING)]
    lead = " ".join(f"[{f.citations[0]}]" for f in findings[:3])
    summary = (f"The retrieved papers cover {len(methods)} kinds of approach: {', '.join(methods)}. "
               f"The closest matches are {lead}." if methods else f"The closest matches are {lead}.")
    return Answer(question=question, summary=summary, findings=findings, evidence=evidence,
                  gaps=["Offline mode groups papers by approach type; it does not compare effect sizes "
                        "or check whether studies used comparable setups."],
                  mode="extractive")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def generate_grounded(question: str, retrieved: list[RetrievedPassage], llm: LLM | None = None) -> Answer:
    """Cited answer from the passages. Uses the LLM when given one, else the extractive fallback."""
    if llm is None or not retrieved:
        return extractive_answer(question, retrieved)
    data = complete_json(llm, build_prompt(question, retrieved), system=SYSTEM)
    return answer_from_json(question, data)
