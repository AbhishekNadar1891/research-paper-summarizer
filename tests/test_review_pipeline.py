import json

import pytest

from citebench.generate import extractive_answer
from citebench.models import Answer, EvidenceClaim, Finding, Strength
from citebench.nli import LexicalScorer
from citebench.pipeline import Pipeline
from citebench.review import NoteKind, answer_to_json, review
from citebench.validate import validate

Q = "What approaches have researchers used to reduce hallucinations in language models?"


@pytest.fixture
def res(retriever):
    return retriever.retrieve(Q, top_k=6)


def draft_for(res):
    a = res[0]
    return Answer(question=Q, summary=f"Hallucination is solved by every method [{a.label}].",
                  findings=[Finding(paper_key=a.paper.key, finding=a.passage.text, citations=[a.label])],
                  evidence=[EvidenceClaim(claim="All methods work.", citations=[a.label], strength=Strength.STRONG)])


def test_answer_to_json_roundtrip(res):
    d = answer_to_json(draft_for(res))
    assert d["evidence"][0]["strength"] == "strong"
    json.dumps(d)  # serialisable


def test_offline_review_uses_validation_report(res):
    draft = draft_for(res)
    rep = validate(draft, res)
    result = review(draft, res, rep, llm=None)
    assert not result.by_llm and result.revised is None
    assert NoteKind.INFLATED_STRENGTH in {n.kind for n in result.notes}


def test_llm_review_and_revise(res, fake_llm):
    draft = draft_for(res)
    a = res[0]
    notes = {"verdict": "major", "notes": [
        {"kind": "overclaim", "target": "summary", "comment": "'Solved by every method' is not in the passage.",
         "citations": [a.label]},
        {"kind": "weird_kind", "target": "overall", "comment": "Mention trade-offs."}]}
    revised = {"summary": f"One approach reduced errors [{a.label}].",
               "findings": [{"paper": a.paper.key, "finding": a.passage.text, "citations": [a.label]}],
               "evidence": [{"claim": "One method helped.", "citations": [a.label], "strength": "limited"}]}
    llm = fake_llm(notes, revised)
    result = review(draft, res, validate(draft, res), llm)
    assert result.verdict == "major"
    assert [n.kind for n in result.notes] == [NoteKind.OVERCLAIM, NoteKind.OTHER]
    assert "inflated" in llm.prompts[0] or "downgraded" in llm.prompts[0]  # report issues are passed in
    assert result.revised.summary.startswith("One approach")
    assert result.revised.evidence[0].strength is Strength.LIMITED


def test_accept_skips_revision(res, fake_llm):
    llm = fake_llm({"verdict": "accept", "notes": []})
    result = review(extractive_answer(Q, res), res, None, llm)
    assert result.revised is None and len(llm.prompts) == 1


def test_empty_revision_is_ignored(res, fake_llm):
    llm = fake_llm({"verdict": "minor", "notes": [{"kind": "overclaim", "comment": "x"}]}, {"summary": ""})
    assert review(draft_for(res), res, None, llm).revised is None


# --- pipeline -------------------------------------------------------------
def test_pipeline_offline(collection, retriever):
    p = Pipeline(collection, llm=None, scorer=LexicalScorer(), retriever=retriever)
    r = p.run(Q)
    assert not r.llm_used and r.retrieval_mode == "bm25"
    assert r.answer.mode == "extractive"
    assert r.report.ok
    assert r.highlights and r.highlights.scorer == "lexical-overlap"
    assert r.comparison is None
    assert {"retrieve", "generate", "validate", "review", "highlight"} <= set(r.timings)


def test_pipeline_with_llm(collection, retriever, fake_llm):
    res = retriever.retrieve(Q)
    a, b = res[0], res[1]
    draft = {"summary": f"Two approaches help [{a.label}] [{b.label}]. A fake source agrees [P99].",
             "findings": [{"paper": a.paper.key, "finding": a.passage.text, "citations": [a.label]},
                          {"paper": b.paper.key, "finding": b.passage.text, "citations": [a.label]}],
             "evidence": [{"claim": "Both help.", "citations": [a.label, b.label], "strength": "strong"}]}
    review_notes = {"verdict": "accept", "notes": []}
    closed_book = {"answer": "Retrieval helps.", "claims": ["Retrieval augmentation reduces hallucination in dialogue."]}
    # The reviewer accepts, but the validator's proven errors still force a revision; this fake
    # "revision" repeats the draft, so the same errors must still be caught and removed.
    llm = fake_llm(draft, review_notes, draft, closed_book)

    r = Pipeline(collection, llm=llm, scorer=LexicalScorer(), retriever=retriever).run(Q)
    assert r.llm_used
    # validator caught the invented and misattributed citations and the inflated rating
    assert r.report.invented == 1 and r.report.misattributed == 1
    assert r.answer.evidence[0].strength is Strength.MODERATE
    # invalid labels never reach the user
    assert "[P99]" not in r.answer.summary
    assert r.answer.findings[1].citations == []
    assert r.comparison and r.comparison.stats()["claims"] == 1
    assert len(llm.prompts) == 4
    assert r.review.verdict == "minor"


def test_pipeline_flags_can_skip_stages(collection, retriever):
    r = Pipeline(collection, llm=None, scorer=LexicalScorer(), retriever=retriever).run(
        Q, with_review=False, with_highlights=False)
    assert r.review is None and r.highlights is None and r.draft is None


def test_validator_errors_force_revision_when_reviewer_accepts(res, fake_llm):
    a = res[0]
    draft = Answer(question=Q, summary=f"Something [P99].",
                   findings=[Finding(paper_key=a.paper.key, finding=a.passage.text, citations=[a.label])])
    revised = {"summary": f"Fixed [{a.label}].", "findings": [{"paper": a.paper.key, "finding": a.passage.text,
                                                               "citations": [a.label]}]}
    llm = fake_llm({"verdict": "accept", "notes": []}, revised)
    result = review(draft, res, validate(draft, res), llm)
    assert result.verdict == "minor"
    assert any(n.kind is NoteKind.BAD_CITATION for n in result.notes)
    assert result.revised is not None and "P99" not in result.revised.summary
