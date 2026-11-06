import pytest

from citebench.ingest import (
    Collection,
    _arxiv_id,
    chunk_text,
    clean_pdf_text,
    load_starter,
    split_sentences,
)
from citebench.models import Family, Paper


def test_split_keeps_decimals_and_abbreviations():
    text = "It fell from 47.5% to 14.5% on average. Lewis et al. (2020) disagree. e.g. this one stays."
    assert split_sentences(text) == [
        "It fell from 47.5% to 14.5% on average.",
        "Lewis et al. (2020) disagree. e.g. this one stays.",
    ]


def test_split_empty():
    assert split_sentences("   ") == []


def test_chunk_groups_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(7))
    chunks = chunk_text(text, sentences_per_chunk=3, min_chars=1)
    assert len(chunks) == 3
    assert chunks[0].startswith("Sentence number 0")
    assert chunks[-1] == "Sentence number 6 is here."


def test_chunk_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(5))
    chunks = chunk_text(text, sentences_per_chunk=3, overlap=1, min_chars=1)
    assert chunks[1].startswith("Sentence number 2")


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("A b. C d.", sentences_per_chunk=2, overlap=2)


def test_clean_pdf_text_rejoins_and_drops_references():
    body = "Retrieval aug-\nmentation reduces errors.\nIt works well. " * 20
    raw = body + "\nReferences\n[1] Someone. A paper. 2020."
    cleaned = clean_pdf_text(raw)
    assert "augmentation" in cleaned
    assert "[1] Someone" not in cleaned


def test_arxiv_id_parsing():
    assert _arxiv_id("https://arxiv.org/abs/2305.14251v2") == "2305.14251"
    assert _arxiv_id("2104.07567") == "2104.07567"
    with pytest.raises(ValueError):
        _arxiv_id("not-an-id")


def test_starter_collection_loads():
    col = load_starter()
    assert len(col) == 17
    assert len(col.passages) == 51
    for paper in col.papers.values():
        assert paper.url.startswith("https://arxiv.org/abs/")
    families = {p.family for p in col.papers.values()}
    assert Family.RETRIEVAL in families and Family.MEASUREMENT in families


def test_collection_add_remove_and_roundtrip(tmp_path):
    col = Collection()
    paper = Paper(key="x2024", title="X", authors="X et al.", year=2024,
                  text=("First claim is stated here. Second claim follows on. Third claim ends it. "
                        "A fourth, longer sentence starts the second passage of this paper."))
    added = col.add(paper)
    assert [p.uid for p in added] == ["x2024#0", "x2024#1"]
    with pytest.raises(ValueError):
        col.add(paper)

    path = tmp_path / "c.json"
    col.save(path)
    again = Collection.load(path)
    assert [p.text for p in again.passages] == [p.text for p in col.passages]

    col.remove("x2024")
    assert len(col) == 0 and not col.passages


def test_short_tail_is_folded_into_previous_chunk():
    text = "This is a reasonably long first sentence. And a second long sentence too. Then a third. Tiny."
    chunks = chunk_text(text, sentences_per_chunk=3, min_chars=40)
    assert len(chunks) == 1 and chunks[0].endswith("Tiny.")
