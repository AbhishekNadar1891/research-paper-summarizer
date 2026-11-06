import pytest

from citebench.compare import ClaimStatus, check_claims, compare, extract_claims
from citebench.generate import extractive_answer
from citebench.nli import LexicalScorer

Q = "What approaches have researchers used to reduce hallucinations in language models?"

CLOSED_BOOK = {
    "answer": "Researchers use retrieval, verification and decoding tricks. Some claims here are made up.",
    "claims": [
        # paraphrases of starter passages -> should be backed
        "Chain-of-Verification has the model draft a response, plan verification questions and answer them independently.",
        "On TruthfulQA the best model was truthful on 58% of questions while humans were truthful on 94%.",
        # plausible but not in the collection -> not in papers
        "Constitutional classifiers block jailbreak prompts in production chat systems.",
        # flips a passage -> contradicted (lexical scorer catches the negation)
        "Retrieval-augmented dialogue models did not reduce knowledge hallucination compared with standard large dialogue models.",
    ],
}


@pytest.fixture
def res(retriever):
    return retriever.retrieve(Q, top_k=12) + retriever.retrieve("TruthfulQA humans truthful 94%", top_k=2)


@pytest.fixture
def relabelled(res):
    # make labels unique after concatenating two retrievals
    for i, r in enumerate(res, start=1):
        r.label = f"P{i}"
    return res


def test_extract_claims_fallback():
    assert extract_claims("One fact. Two facts.", None) == ["One fact.", "Two facts."]
    assert extract_claims("ignored", ["  a ", "", "b"]) == ["a", "b"]


@pytest.fixture
def per_claim(retriever):
    return lambda claim: retriever.retrieve(claim, top_k=3)


def test_check_claims_statuses(relabelled, per_claim):
    checks = check_claims(CLOSED_BOOK["claims"], relabelled, LexicalScorer(), claim_retriever=per_claim)
    statuses = [c.status for c in checks]
    assert statuses[0] in (ClaimStatus.BACKED, ClaimStatus.PARTIAL)
    assert statuses[1] in (ClaimStatus.BACKED, ClaimStatus.PARTIAL)
    assert statuses[2] is ClaimStatus.NOT_IN_PAPERS
    assert statuses[3] is ClaimStatus.CONTRADICTED
    assert checks[1].evidence and "94%" in checks[1].evidence.sentence


def test_compare_end_to_end(relabelled, fake_llm, per_claim):
    grounded = extractive_answer(Q, relabelled)
    llm = fake_llm(CLOSED_BOOK)
    cmp = compare(Q, relabelled, llm, scorer=LexicalScorer(), grounded=grounded, claim_retriever=per_claim)
    assert "no documents" in llm.prompts[0].lower()
    assert "PASSAGES" not in llm.prompts[0]  # closed-book really is closed-book
    s = cmp.stats()
    assert s["claims"] == 4
    assert s["contradicted"] == 1 and s["not_in_papers"] >= 1
    assert 0 < cmp.backed_rate < 1
    # the grounded answer used papers that the memory-only answer never matched
    assert cmp.papers_unused_by_memory
    assert set(cmp.papers_unused_by_memory) <= {f.paper_key for f in grounded.findings}


def test_compare_needs_llm(relabelled):
    assert compare(Q, relabelled, llm=None) is None


def test_compare_with_sentence_fallback(relabelled, fake_llm):
    cmp = compare(Q, relabelled, fake_llm({"answer": "TruthfulQA has 817 questions in 38 categories. Cats can fly."}),
                  scorer=LexicalScorer())
    assert [c.claim for c in cmp.claims] == ["TruthfulQA has 817 questions in 38 categories.", "Cats can fly."]
    assert cmp.claims[1].status is ClaimStatus.NOT_IN_PAPERS


def test_without_claim_retriever_only_question_passages_count(retriever):
    res = retriever.retrieve("TruthfulQA benchmark", top_k=1)
    checks = check_claims(["Chain-of-Verification plans verification questions."], res, LexicalScorer())
    assert checks[0].status is ClaimStatus.NOT_IN_PAPERS
