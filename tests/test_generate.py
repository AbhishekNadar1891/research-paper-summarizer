import pytest

from citebench.generate import answer_from_json, build_prompt, extractive_answer, generate_grounded
from citebench.llm import LLMError, complete_json, parse_json
from citebench.models import Strength

Q = "What approaches have researchers used to reduce hallucinations in language models?"


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Here you go:\n```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json('Sure. {"a": 3} Hope that helps.') == {"a": 3}
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_complete_json_repairs_once(fake_llm):
    llm = fake_llm("not json", '{"ok": true}')
    assert complete_json(llm, "x") == {"ok": True}
    assert "not valid JSON" in llm.prompts[1]


def test_prompt_contains_labels_and_paper_keys(retriever):
    res = retriever.retrieve(Q, top_k=4)
    prompt = build_prompt(Q, res)
    for r in res:
        assert f"[{r.label}] paper={r.paper.key}" in prompt
    assert "Use ONLY the passages" in prompt


def test_generate_with_fake_llm(retriever, fake_llm):
    res = retriever.retrieve(Q, top_k=4)
    reply = {
        "summary": f"Verification helps [{res[0].label}].",
        "findings": [{"paper": res[0].paper.key, "finding": "It helped.", "citations": [res[0].label],
                      "study_type": "method"}],
        "evidence": [{"claim": "Fixes help.", "citations": ["P1", "P2"], "strength": "Moderate", "why": "two"}],
        "tensions": [{"point": "Trade-off.", "citations": "P2, P3"}],
        "gaps": ["Cost is not measured."],
    }
    ans = generate_grounded(Q, res, llm=fake_llm(reply))
    assert ans.mode == "llm"
    assert ans.findings[0].paper_key == res[0].paper.key
    assert ans.evidence[0].strength is Strength.MODERATE
    assert ans.tensions[0].citations == ["P2", "P3"]
    assert ans.gaps == ["Cost is not measured."]


def test_answer_from_json_is_forgiving():
    ans = answer_from_json("q", {"summary": "s", "findings": [{"paper": "x"}, "junk"],
                                 "evidence": [{"claim": "c", "strength": "huge", "citations": ["[P1]", "P1", "bad"]}]})
    assert ans.findings == []
    assert ans.evidence[0].strength is Strength.LIMITED
    assert ans.evidence[0].citations == ["P1"]
    assert answer_from_json("q", ["not", "a", "dict"]).summary == ""


def test_extractive_answer_without_llm(retriever):
    res = retriever.retrieve(Q)
    ans = generate_grounded(Q, res, llm=None)
    assert ans.mode == "extractive"
    keys = {r.paper.key for r in res}
    assert {f.paper_key for f in ans.findings} == keys
    labels = {r.label for r in res}
    for f in ans.findings:
        assert set(f.citations) <= labels
        own = {r.label for r in res if r.paper.key == f.paper_key}
        assert set(f.citations) <= own
    # strength follows paper count
    for e in ans.evidence:
        assert e.strength is Strength.from_paper_count(len(e.citations))


def test_extractive_finding_keeps_decimal_numbers(retriever):
    res = retriever.retrieve("hallucination rate from 47.5% to 14.5% GPT-3.5", top_k=3)
    ans = extractive_answer("hallucination rate GPT-3.5", res)
    f = next(f for f in ans.findings if f.paper_key == "varshney2023")
    assert "47.5%" in f.finding or "88%" in f.finding


def test_no_passages():
    ans = generate_grounded("anything", [], llm=None)
    assert ans.findings == [] and ans.gaps


def test_resolve_paper_key_variants(retriever):
    from citebench.generate import resolve_paper_key
    res = retriever.retrieve(Q, top_k=12)
    keys = {r.paper.key for r in res}
    assert "dhuliawala2023" in keys
    lbl = next(r.label for r in res if r.paper.key == "dhuliawala2023")
    assert resolve_paper_key("dhuliawala2023", [], res) == "dhuliawala2023"
    assert resolve_paper_key("Dhuliawala et al. (2023)", [], res) == "dhuliawala2023"
    assert resolve_paper_key("Chain-of-Verification Reduces Hallucination in Large Language Models", [], res) == "dhuliawala2023"
    assert resolve_paper_key(lbl, [], res) == "dhuliawala2023"
    assert resolve_paper_key("Nobody et al.", [lbl], res) == "Nobody et al."  # named but unknown: kept, validator flags it


def test_answer_from_json_attributes_unnamed_finding_by_citation(retriever):
    res = retriever.retrieve(Q, top_k=4)
    ans = answer_from_json(Q, {"summary": "s", "findings": [{"finding": "x", "citations": [res[0].label]}]}, res)
    assert ans.findings[0].paper_key == res[0].paper.key
