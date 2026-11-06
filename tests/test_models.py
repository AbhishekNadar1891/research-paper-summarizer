from citebench.models import Passage, Strength


def test_strength_from_paper_count():
    assert Strength.from_paper_count(0) is Strength.LIMITED
    assert Strength.from_paper_count(1) is Strength.LIMITED
    assert Strength.from_paper_count(2) is Strength.MODERATE
    assert Strength.from_paper_count(5) is Strength.STRONG


def test_strength_min_papers():
    assert Strength.STRONG.min_papers == 3
    assert Strength.MODERATE.min_papers == 2


def test_passage_uid():
    assert Passage(paper_key="lewis2020", index=2, text="x").uid == "lewis2020#2"
