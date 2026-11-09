import pytest

from citebench.generate import extractive_answer
from citebench.models import Answer, Finding, RetrievedPassage
from citebench.nli import (
    LexicalScorer,
    NLIScore,
    Support,
    best_support,
    classify,
    highlight_answer,
    highlight_text,
)

Q = "What approaches have researchers used to reduce hallucinations in language models?"


class KeywordNLI:
    """Deterministic stand-in for an NLI model: entails if all keywords of the hypothesis are in the premise."""

    name = "fake"
    supported_at, partial_at, contradicted_at = 0.6, 0.3, 0.5

    def score(self, pairs):
        out = []
        for premise, hyp in pairs:
            if "CONTRA" in hyp and "57.6%" in premise:
                out.append(NLIScore(0.05, 0.9, 0.05))
            elif all(w.lower() in premise.lower() for w in hyp.split()[:3]):
                out.append(NLIScore(0.9, 0.02, 0.08))
            else:
                out.append(NLIScore(0.1, 0.1, 0.8))
        return out


@pytest.fixture
def res(retriever):
    return retriever.retrieve(Q, top_k=12)


def label_of(res, key):
    return next(r.label for r in res if r.paper.key == key)


@pytest.fixture
def pinned(collection):
    """Specific passages with fixed labels, independent of retrieval ranking."""
    def get(key, idx, label):
        p = next(x for x in collection.passages if x.paper_key == key and x.index == idx)
        return RetrievedPassage(label=label, passage=p, paper=collection.papers[key], score=1.0)
    return [get("varshney2023", 1, "P1"), get("lin2021", 1, "P2"), get("dhuliawala2023", 1, "P3")]


def test_classify_thresholds():
    s = LexicalScorer()
    assert classify(NLIScore(0.9, 0, 0.1), s) is Support.SUPPORTED
    assert classify(NLIScore(0.4, 0, 0.6), s) is Support.PARTIAL
    assert classify(NLIScore(0.1, 0, 0.9), s) is Support.UNSUPPORTED
    assert classify(NLIScore(0.1, 0.8, 0.1), s) is Support.CONTRADICTED


def test_supported_sentence_points_at_source(pinned):
    res, lbl = pinned, "P1"
    out = highlight_text(f"Detection reached about 88% recall [{lbl}].", res, KeywordNLI())
    assert out[0].support is Support.SUPPORTED
    assert out[0].best.label == lbl
    assert "88% recall" in out[0].best.sentence
    assert out[0].text == "Detection reached about 88% recall."


def test_contradiction_is_surfaced(pinned):
    res, lbl = pinned, "P1"
    out = highlight_text(f"CONTRA mitigation never corrected anything [{lbl}].", res, KeywordNLI())
    assert out[0].support is Support.CONTRADICTED


def test_uncited_and_invalid_citations(res):
    out = highlight_text("No citation here. Bad citation [P99].", res, KeywordNLI())
    assert [s.support for s in out] == [Support.UNCITED, Support.UNSUPPORTED]


def test_finding_uses_default_citations(pinned):
    res, lbl = pinned, "P3"
    out = highlight_text("CoVe decreased hallucinations on list questions.", res, KeywordNLI(), default_citations=[lbl])
    assert out[0].citations == [lbl] and out[0].support is Support.SUPPORTED


def test_lexical_scorer_catches_negation_flip():
    s = LexicalScorer()
    [same] = s.score([("Retrieval reduced hallucination in dialogue.", "Retrieval reduced hallucination in dialogue.")])
    [flip] = s.score([("Retrieval reduced hallucination in dialogue.", "Retrieval did not reduce hallucination in dialogue.")])
    assert same.entailment == 1.0
    assert flip.contradiction > flip.entailment


def test_best_support_prefers_matching_sentence(pinned):
    r = pinned[1]
    ev = best_support("humans were truthful on 94% of questions", [r], LexicalScorer())
    assert ev and "94%" in ev.sentence


def test_highlight_answer_extractive_is_mostly_supported(res):
    ans = extractive_answer(Q, res)
    h = highlight_answer(ans, res, scorer=LexicalScorer())
    assert h.scorer == "lexical-overlap"
    finding_labels = [s.support for v in h.findings.values() for s in v]
    # extractive findings are copied from their passages, so they must all be supported
    assert finding_labels and all(lbl is Support.SUPPORTED for lbl in finding_labels)
    assert sum(h.counts().values()) == len(h.summary) + len(finding_labels)


def test_highlight_answer_flags_unsupported_finding(res):
    lbl = res[0].label
    ans = Answer(question=Q, summary="", findings=[
        Finding(paper_key=res[0].paper.key, finding="Quantum annealing solves protein folding instantly.",
                citations=[lbl])])
    h = highlight_answer(ans, res, scorer=LexicalScorer())
    assert h.findings[res[0].paper.key][0].support is Support.UNSUPPORTED


def test_changed_number_is_not_fully_supported(pinned):
    out = highlight_text("Detection reached about 99% recall [P1].", pinned, LexicalScorer())
    assert out[0].support is not Support.SUPPORTED


def test_negation_count_mismatch_with_existing_negation():
    s = LexicalScorer()
    premise = "Without retrieved knowledge, DoLa improved truthfulness on TruthfulQA."
    [flip] = s.score([(premise, "Without retrieved knowledge, DoLa did not improve truthfulness on TruthfulQA.")])
    assert flip.contradiction > flip.entailment


def test_off_topic_passages_are_not_judged(pinned):
    from citebench.nli import best_support
    class AlwaysContradicts:
        name = "contra"; supported_at, partial_at, contradicted_at = 0.6, 0.3, 0.5
        def score(self, pairs):
            return [NLIScore(0.0, 0.99, 0.01) for _ in pairs]
    # nothing in the pinned passages is about the C4 dataset -> no verdict at all
    assert best_support("The C4 dataset is used to pretrain language models.", pinned, AlwaysContradicts()) is None
    out = highlight_text("The C4 dataset is used to pretrain language models [P1].", pinned, AlwaysContradicts())
    assert out[0].support is Support.UNSUPPORTED
