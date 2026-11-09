import pytest

from citebench.generate import extractive_answer
from citebench.models import Answer, EvidenceClaim, Finding, Strength, Tension
from citebench.validate import IssueKind, drop_invalid_citations, support_overlap, validate

Q = "What approaches have researchers used to reduce hallucinations in language models?"


@pytest.fixture
def res(retriever):
    return retriever.retrieve(Q, top_k=12)


def by_paper(res, key):
    return next(r for r in res if r.paper.key == key)


def kinds(report):
    return [i.kind for i in report.issues]


def test_extractive_answer_is_clean(res):
    rep = validate(extractive_answer(Q, res), res)
    assert rep.ok
    assert rep.invented == rep.misattributed == 0
    assert rep.total > 0


def test_invented_label_is_flagged(res):
    ans = Answer(question=Q, summary="Something is true [P99].")
    rep = validate(ans, res)
    assert rep.invented == 1
    assert IssueKind.INVENTED in kinds(rep)
    assert not rep.ok


def test_misattributed_finding(res):
    a, b = res[0], next(r for r in res if r.paper.key != res[0].paper.key)
    ans = Answer(question=Q, summary="", findings=[
        Finding(paper_key=a.paper.key, finding=a.passage.text, citations=[b.label])])
    rep = validate(ans, res)
    assert rep.misattributed == 1
    assert rep.status_of(f"finding:{a.paper.key}", b.label) == "misattributed"


def test_unknown_paper(res):
    ans = Answer(question=Q, summary="", findings=[Finding(paper_key="nobody2099", finding="x", citations=["P1"])])
    assert IssueKind.UNKNOWN_PAPER in kinds(validate(ans, res))


def test_weak_support(res):
    r = res[0]
    ans = Answer(question=Q, summary="", findings=[
        Finding(paper_key=r.paper.key, finding="Bananas ripen faster beside apples in warm kitchens.",
                citations=[r.label])])
    rep = validate(ans, res)
    assert rep.weak == 1 and rep.ok  # weak is a warning, not an error
    assert rep.resolved == 1


def test_good_support_is_ok(res):
    r = by_paper(res, "dhuliawala2023")
    # A faithful paraphrase of the cited passage: drop a few words, keep the substance.
    words = r.passage.text.split()
    paraphrase = " ".join(w for i, w in enumerate(words) if i % 4 != 3)
    ans = Answer(question=Q, summary="", findings=[
        Finding(paper_key=r.paper.key, finding=paraphrase, citations=[r.label])])
    rep = validate(ans, res)
    assert rep.checks[0].status == "ok"


def test_uncited_finding_and_summary_sentence(res):
    ans = Answer(question=Q, summary="Retrieval and verification both reduce hallucination in large models.",
                 findings=[Finding(paper_key=res[0].paper.key, finding="It helped.", citations=[])])
    rep = validate(ans, res)
    assert kinds(rep).count(IssueKind.UNCITED) == 2


def test_strength_downgraded_when_too_few_papers(res):
    one = res[0]
    two = next(r for r in res if r.paper.key != one.paper.key)
    same_paper = [r for r in res if r.paper.key == one.paper.key]
    ev = [
        EvidenceClaim(claim="A", citations=[one.label], strength=Strength.STRONG),
        EvidenceClaim(claim="B", citations=[one.label, two.label], strength=Strength.STRONG),
        EvidenceClaim(claim="C", citations=[r.label for r in same_paper], strength=Strength.MODERATE),
        EvidenceClaim(claim="D", citations=[one.label, "P99"], strength=Strength.MODERATE),
    ]
    ans = Answer(question=Q, summary="", evidence=ev)
    rep = validate(ans, res)
    assert [e.strength for e in ev] == [Strength.LIMITED, Strength.MODERATE, Strength.LIMITED, Strength.LIMITED]
    assert all(e.downgraded for e in ev)
    assert kinds(rep).count(IssueKind.DOWNGRADED) == 4


def test_strength_not_downgraded_when_justified(res):
    labels = []
    seen = set()
    for r in res:
        if r.paper.key not in seen:
            seen.add(r.paper.key)
            labels.append(r.label)
        if len(labels) == 3:
            break
    ev = EvidenceClaim(claim="X", citations=labels, strength=Strength.STRONG)
    validate(Answer(question=Q, summary="", evidence=[ev]), res)
    assert ev.strength is Strength.STRONG and not ev.downgraded


def test_fix_strength_off_leaves_rating(res):
    ev = EvidenceClaim(claim="A", citations=[res[0].label], strength=Strength.STRONG)
    validate(Answer(question=Q, summary="", evidence=[ev]), res, fix_strength=False)
    assert ev.strength is Strength.STRONG


def test_tension_citations_checked(res):
    ans = Answer(question=Q, summary="", tensions=[Tension(point="t", citations=["P1", "P42"])])
    rep = validate(ans, res)
    assert rep.invented == 1 and rep.resolved == 1


def test_drop_invalid_citations(res):
    a, b = res[0], next(r for r in res if r.paper.key != res[0].paper.key)
    ans = Answer(question=Q, summary=f"Real claim [{a.label}] and fake [P77].", findings=[
        Finding(paper_key=a.paper.key, finding=a.passage.text, citations=[a.label, b.label])])
    rep = validate(ans, res)
    drop_invalid_citations(ans, rep)
    assert "[P77]" not in ans.summary and f"[{a.label}]" in ans.summary
    assert ans.summary.endswith("fake.")
    assert ans.findings[0].citations == [a.label]


def test_support_overlap_ignores_labels():
    assert support_overlap("retrieval reduces hallucination [P1]", "Retrieval reduces hallucination.") == 1.0
    assert support_overlap("", "anything") == 1.0


def test_report_summary_shape(res):
    rep = validate(Answer(question=Q, summary="X happens [P1]. Y happens [P50]."), res)
    s = rep.summary()
    assert s["total"] == 2 and s["invented"] == 1


def test_changed_number_is_flagged(collection):
    from citebench.models import RetrievedPassage
    p = next(x for x in collection.passages if x.paper_key == "lin2021" and x.index == 1)
    r = RetrievedPassage(label="P1", passage=p, paper=collection.papers["lin2021"], score=1.0)
    good = Answer(question=Q, summary="", findings=[Finding("lin2021", "The best model was truthful on 58% of questions.", ["P1"])])
    bad = Answer(question=Q, summary="Humans were truthful on 49% [P1].",
                 findings=[Finding("lin2021", "The best model was truthful on 85% of questions.", ["P1"])])
    assert IssueKind.NUMBER not in kinds(validate(good, [r]))
    assert kinds(validate(bad, [r])).count(IssueKind.NUMBER) == 2


def test_numbers_ignore_citation_labels():
    from citebench.validate import numbers_in
    assert numbers_in("Rose from 47.50% to 14.5% [P3] in 2023") == {"47.5", "14.5", "2023"}


def test_reference_years_are_not_numbers():
    from citebench.validate import numbers_in
    assert numbers_in("Gekhman et al. (2024) found 58% [P1]. DoLa (2023) helps. Lewis et al., 2020 agrees.") == {"58"}
    assert numbers_in("In 2023, accuracy reached 71%.") == {"2023", "71"}
