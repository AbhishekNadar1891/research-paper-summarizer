from collections import Counter

import numpy as np
import pytest

from citebench.ingest import load_starter
from citebench.retrieval import (
    HybridRetriever,
    RetrievalConfig,
    expand_query,
    reciprocal_rank_fusion,
    select_diverse,
    stem,
    tokenize,
)


@pytest.fixture(scope="module")
def col():
    return load_starter()


@pytest.fixture(scope="module")
def bm25(col):
    return HybridRetriever(col, use_embeddings=False)


def test_stem_conflates_word_forms():
    assert stem("hallucination") == stem("hallucinations") == stem("hallucinate")
    assert stem("calibration") == stem("calibrate")
    assert tokenize("The model is hallucinating") == ["model", stem("hallucinating")]


def test_query_expansion_adds_weighted_synonyms():
    w = expand_query(tokenize("reduce hallucination"))
    assert w[stem("hallucination")] == 1.0
    assert w[stem("factuality")] < 1.0


def test_bm25_mode_without_encoder(bm25):
    assert bm25.mode == "bm25"


def test_black_box_detection_query_finds_selfcheckgpt(bm25):
    top = bm25.retrieve("detect hallucinations without access to model internals", top_k=3)
    assert top[0].paper.key == "manakul2023"


def test_citation_query_finds_alce(bm25):
    keys = [r.paper.key for r in bm25.retrieve("how accurate are LLM citations", top_k=3)]
    assert "gao2023" in keys


def test_broad_query_is_multi_document(bm25):
    res = bm25.retrieve("What approaches have researchers used to reduce hallucinations in language models?",
                        top_k=12, max_per_paper=2)
    papers = Counter(r.paper.key for r in res)
    assert len(papers) >= 8
    assert max(papers.values()) <= 2


def test_labels_are_sequential(bm25):
    res = bm25.retrieve("retrieval augmentation factuality")
    assert [r.label for r in res] == [f"P{i}" for i in range(1, len(res) + 1)]


def test_empty_query(bm25):
    assert bm25.retrieve("   ") == []


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert fused[1] == fused[2] > fused[3]
    assert 4 in fused


def test_select_diverse_prefers_new_papers():
    ranked = [(0, 9.0), (1, 8.0), (2, 7.0), (3, 6.0)]
    owner = {0: "a", 1: "a", 2: "b", 3: "c"}.get
    chosen = select_diverse(ranked, owner, top_k=3, max_per_paper=2)
    assert [i for i, _ in chosen] == [0, 2, 3]  # second "a" passage waits for other papers
    chosen = select_diverse(ranked, owner, top_k=4, max_per_paper=2)
    assert [i for i, _ in chosen] == [0, 1, 2, 3]


def test_hybrid_path_with_fake_encoder(col):
    vocab = ["consisten", "sampl", "retriev", "decod", "citat"]

    def fake_encode(texts):
        return np.array([[t.lower().count(v) + 0.01 for v in vocab] for t in texts], dtype=float)

    r = HybridRetriever(col, encoder=fake_encode, config=RetrievalConfig(top_k=5))
    assert r.mode == "hybrid"
    res = r.retrieve("sampling consistency to detect hallucination")
    assert len(res) == 5
    assert "manakul2023" in {x.paper.key for x in res}
