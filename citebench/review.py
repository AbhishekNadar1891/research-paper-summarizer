"""Reviewer 2: a critique-and-revise pass over a draft answer.

A second LLM call reads the draft, the passages and the citation report,
and plays a sceptical peer reviewer: it looks for claims stronger than the
passages allow, papers that were retrieved but ignored, numbers that were
misread, and evidence ratings that are too generous. A third call revises
the draft to address those notes. Without an LLM, the notes come from the
citation report alone and no revision is made.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from .generate import SYSTEM, answer_from_json, format_passages
from .llm import LLM, complete_json
from .models import Answer, RetrievedPassage
from .validate import IssueKind, ValidationReport


class NoteKind(str, Enum):
    OVERCLAIM = "overclaim"
    CHERRY_PICK = "cherry_pick"
    INFLATED_STRENGTH = "inflated_strength"
    MISSING_PAPER = "missing_paper"
    MISREAD_NUMBER = "misread_number"
    BAD_CITATION = "bad_citation"
    OTHER = "other"

    @classmethod
    def parse(cls, value: Any) -> "NoteKind":
        try:
            return cls(str(value).lower().strip())
        except ValueError:
            return cls.OTHER


@dataclass
class ReviewNote:
    kind: NoteKind
    target: str      # "summary", "finding:<paper>", "evidence[i]", "overall"
    comment: str
    citations: list[str] = field(default_factory=list)


@dataclass
class ReviewResult:
    verdict: str                     # "accept" | "minor" | "major"
    notes: list[ReviewNote] = field(default_factory=list)
    revised: Answer | None = None    # None when nothing needed changing or no LLM
    by_llm: bool = True


def answer_to_json(answer: Answer) -> dict:
    return {
        "summary": answer.summary,
        "findings": [{"paper": f.paper_key, "finding": f.finding, "citations": f.citations,
                      "study_type": f.study_type.value if f.study_type else None} for f in answer.findings],
        "evidence": [{"claim": e.claim, "citations": e.citations, "strength": e.strength.value, "why": e.why}
                     for e in answer.evidence],
        "tensions": [{"point": t.point, "citations": t.citations} for t in answer.tensions],
        "gaps": answer.gaps,
    }


REVIEW_PROMPT = """You are Reviewer 2: a sceptical, fair peer reviewer. Review this draft answer against the passages.

QUESTION: {question}

PASSAGES:
{passages}

DRAFT ANSWER (JSON):
{draft}

AUTOMATED CITATION CHECK FOUND:
{issues}

Look for:
- overclaim: a claim stronger or broader than its passages say (e.g. "solves" for "reduces", one setting generalised to all)
- cherry_pick: relevant retrieved papers ignored, or caveats/trade-offs in the passages left out
- inflated_strength: an evidence rating higher than the independent, comparable papers justify
  (a benchmark measuring the problem is not evidence that a fix works)
- missing_paper: a retrieved paper that clearly answers the question but has no finding
- misread_number: a number that differs from the passage
- bad_citation: a citation that does not support its claim

Return only JSON:
{{"verdict": "accept|minor|major",
 "notes": [{{"kind": "overclaim|cherry_pick|inflated_strength|missing_paper|misread_number|bad_citation|other",
            "target": "summary | finding:<paper key> | evidence[<index>] | overall",
            "comment": "what is wrong and how to fix it, one or two sentences",
            "citations": ["P3"]}}]}}
Only raise real problems. If the draft is sound, return verdict "accept" with no notes."""

REVISE_PROMPT = """Revise the draft answer to address every reviewer note. Use ONLY the passages.

QUESTION: {question}

PASSAGES:
{passages}

DRAFT ANSWER (JSON):
{draft}

REVIEWER NOTES:
{notes}

Return the full revised answer as JSON in exactly the same shape as the draft
(summary, findings, evidence, tensions, gaps). Keep what was correct. Cite only labels from the passages;
a finding may cite only its own paper's passages; strength "strong" needs 3+ independent papers,
"moderate" 2, "limited" 1."""


def _issues_text(report: ValidationReport | None) -> str:
    if not report or not report.issues:
        return "No problems found by the automated check."
    return "\n".join(f"- [{i.kind.value}] {i.location}: {i.message}" for i in report.issues)


_ISSUE_TO_NOTE = {
    IssueKind.INVENTED: NoteKind.BAD_CITATION,
    IssueKind.MISATTRIBUTED: NoteKind.BAD_CITATION,
    IssueKind.UNKNOWN_PAPER: NoteKind.BAD_CITATION,
    IssueKind.WEAK: NoteKind.BAD_CITATION,
    IssueKind.UNCITED: NoteKind.OVERCLAIM,
    IssueKind.DOWNGRADED: NoteKind.INFLATED_STRENGTH,
    IssueKind.NUMBER: NoteKind.MISREAD_NUMBER,
}


def notes_from_report(report: ValidationReport) -> list[ReviewNote]:
    return [ReviewNote(kind=_ISSUE_TO_NOTE[i.kind], target=i.location, comment=i.message,
                       citations=[i.label] if i.label else []) for i in report.issues]


def _parse_notes(data: Any) -> tuple[str, list[ReviewNote]]:
    if not isinstance(data, dict):
        return "minor", []
    verdict = str(data.get("verdict", "minor")).lower().strip()
    if verdict not in ("accept", "minor", "major"):
        verdict = "minor"
    notes = []
    for n in data.get("notes") or []:
        if isinstance(n, dict) and str(n.get("comment", "")).strip():
            cites = n.get("citations") if isinstance(n.get("citations"), list) else []
            notes.append(ReviewNote(kind=NoteKind.parse(n.get("kind")), target=str(n.get("target", "overall")),
                                    comment=str(n["comment"]).strip(), citations=[str(c) for c in cites]))
    return verdict, notes


def review(answer: Answer, retrieved: Sequence[RetrievedPassage], report: ValidationReport | None,
           llm: LLM | None, revise: bool = True) -> ReviewResult:
    if llm is None:
        notes = notes_from_report(report) if report else []
        verdict = "accept" if not notes else ("major" if report and report.errors else "minor")
        return ReviewResult(verdict=verdict, notes=notes, revised=None, by_llm=False)

    passages = format_passages(list(retrieved))
    draft = json.dumps(answer_to_json(answer), indent=1)
    data = complete_json(llm, REVIEW_PROMPT.format(question=answer.question, passages=passages, draft=draft,
                                                   issues=_issues_text(report)), system=SYSTEM)
    verdict, notes = _parse_notes(data)
    # Problems the validator proved (bad citations, unknown papers, wrong numbers) are fixed even if the
    # reviewing model waved them through; small models tend to accept everything.
    proven = [n for n in notes_from_report(report) if n.kind in (NoteKind.BAD_CITATION, NoteKind.MISREAD_NUMBER)
              and not n.comment.startswith("The claim shares little")] if report else []
    seen = {(n.target, n.comment) for n in notes}
    notes += [n for n in proven if (n.target, n.comment) not in seen]
    if proven and verdict == "accept":
        verdict = "minor"
    result = ReviewResult(verdict=verdict, notes=notes)
    if revise and notes and verdict != "accept":
        notes_text = "\n".join(f"- [{n.kind.value}] {n.target}: {n.comment}" for n in notes)
        revised_data = complete_json(llm, REVISE_PROMPT.format(question=answer.question, passages=passages,
                                                               draft=draft, notes=notes_text), system=SYSTEM)
        revised = answer_from_json(answer.question, revised_data, list(retrieved))
        if revised.summary or revised.findings:  # ignore an empty revision rather than lose the draft
            result.revised = revised
    return result
