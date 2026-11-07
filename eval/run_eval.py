"""Offline evaluation of Citebench.  Run:  python eval/run_eval.py

1. Retrieval: recall / MRR of hand-labelled relevant papers, plus ablations
   (no synonym expansion, no per-paper diversity).
2. Citation checking: plant known errors in correct answers and measure how
   many the citation validator and the truth highlighter catch, and how often
   they raise false alarms on clean answers.

Writes eval/results.json and docs/EVALUATION.md. Uses BM25 retrieval and the
lexical NLI scorer, so it needs no API key and no model downloads.
Pass --nli-model to use the real NLI cross-encoder instead (needs sentence-transformers).
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from citebench import retrieval as R  # noqa: E402
from citebench.generate import extractive_answer  # noqa: E402
from citebench.ingest import load_starter  # noqa: E402
from citebench.models import Answer, EvidenceClaim, Finding, Strength  # noqa: E402
from citebench.nli import LexicalScorer, Support, highlight_text, load_scorer  # noqa: E402
from citebench.retrieval import HybridRetriever, RetrievalConfig  # noqa: E402
from citebench.validate import IssueKind, validate  # noqa: E402

QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text())["questions"]


# ---------------------------------------------------------------------------
# 1. Retrieval
# ---------------------------------------------------------------------------
def retrieval_metrics(retriever: HybridRetriever, ks=(3, 5, 12)) -> dict:
    per_q, mrr, hit1 = [], 0.0, 0
    recalls = {k: 0.0 for k in ks}
    for item in QUESTIONS:
        rel = set(item["relevant"])
        res = retriever.retrieve(item["q"], top_k=max(ks))
        ranked_papers = list(dict.fromkeys(r.paper.key for r in res))
        first = next((i for i, k in enumerate(ranked_papers, 1) if k in rel), None)
        mrr += 1 / first if first else 0
        hit1 += first == 1
        row = {"q": item["q"], "first_relevant_rank": first}
        for k in ks:
            got = len(rel & set(ranked_papers[:k])) / len(rel)
            recalls[k] += got
            row[f"recall@{k}"] = round(got, 3)
        row["distinct_papers@12"] = len(ranked_papers)
        per_q.append(row)
    n = len(QUESTIONS)
    return {
        **{f"recall@{k}": round(v / n, 3) for k, v in recalls.items()},
        "mrr": round(mrr / n, 3), "hit@1": round(hit1 / n, 3),
        "avg_distinct_papers@12": round(sum(r["distinct_papers@12"] for r in per_q) / n, 2),
        "per_question": per_q,
    }


def run_retrieval(col, embeddings: bool = False) -> dict:
    full = retrieval_metrics(HybridRetriever(col, use_embeddings=False))
    hybrid = None
    if embeddings:
        h = HybridRetriever(col, use_embeddings=True)
        hybrid = retrieval_metrics(h) if h.mode == "hybrid" else None

    saved = R.SYNONYMS
    R.SYNONYMS = []
    no_syn = retrieval_metrics(HybridRetriever(col, use_embeddings=False))
    R.SYNONYMS = saved

    no_div = HybridRetriever(col, use_embeddings=False,
                             config=RetrievalConfig(top_k=12, max_per_paper=99))
    no_div.retrieve = lambda q, top_k=None, max_per_paper=None, _r=no_div: _plain_topk(_r, q, top_k or 12)
    no_diversity = retrieval_metrics(no_div)
    strip = lambda d: {k: v for k, v in d.items() if k != "per_question"}  # noqa: E731
    out = {"bm25_full": full, "ablation_no_synonyms": strip(no_syn), "ablation_no_diversity": strip(no_diversity)}
    if hybrid:
        out["hybrid_embeddings"] = hybrid
    return out


def _plain_topk(r: HybridRetriever, q: str, k: int):
    from citebench.models import RetrievedPassage
    ranked = r.rank(q)[:k]
    ps = r.collection.passages
    return [RetrievedPassage(label=f"P{i}", passage=ps[j], paper=r.collection.paper_of(ps[j]), score=s)
            for i, (j, s) in enumerate(ranked, 1)]


# ---------------------------------------------------------------------------
# 2. Citation checking: planted errors
# ---------------------------------------------------------------------------
_VERBS = {
    "reduced": "reduce", "improved": "improve", "decreased": "decrease", "outperformed": "outperform",
    "reached": "reach", "raised": "raise", "corrected": "correct", "achieved": "achieve", "showed": "show",
    "reduces": "reduce", "improves": "improve", "detects": "detect", "shows": "show", "trains": "train",
    "combines": "combine", "breaks": "break", "obtains": "obtain", "shifts": "shift", "works": "work",
    "learn": "learn", "set": "set", "found": "find", "finds": "find", "fine-tune": "fine-tune",
}


def negate(text: str) -> str | None:
    for word in re.findall(r"[A-Za-z\-]+", text):
        base = _VERBS.get(word.lower())
        if base:
            return re.sub(rf"\b{re.escape(word)}\b", f"did not {base}", text, count=1)
    for aux in (" is ", " are ", " was ", " were ", " can "):
        if aux in text:
            return text.replace(aux, aux.rstrip() + " not ", 1)
    return None


def change_number(text: str) -> str | None:
    m = re.search(r"\d+(?:\.\d+)?", text)
    if not m:
        return None
    old = m.group(0)
    new = str(round(float(old) * 1.7 + 9, 1)).rstrip("0").rstrip(".")
    return text[: m.start()] + new + text[m.end():]


@dataclass
class Case:
    kind: str
    answer: Answer
    location: str          # where the planted error is
    expected: tuple        # IssueKinds that count as "caught" by the validator


def make_cases(question: str, res, rng_offset: int) -> list[Case]:
    clean = extractive_answer(question, res)
    clean.summary = ""  # judge findings and evidence only
    cases = [Case("clean", clean, "", ())]
    labels_by_paper = {}
    for r in res:
        labels_by_paper.setdefault(r.paper.key, []).append(r.label)
    others = list(res)
    for i, f in enumerate(clean.findings):
        loc = f"finding:{f.paper_key}"

        def variant(**changes):
            a = copy.deepcopy(clean)
            target = next(x for x in a.findings if x.paper_key == f.paper_key)
            for k, v in changes.items():
                setattr(target, k, v)
            return a

        cases.append(Case("invented_label", variant(citations=["P99"]), loc, (IssueKind.INVENTED,)))
        other = next((r for r in others[(i + rng_offset) % len(others):] + others if r.paper.key != f.paper_key), None)
        if other:
            cases.append(Case("wrong_paper_label", variant(citations=[other.label]), loc, (IssueKind.MISATTRIBUTED,)))
            unrelated = other.passage.text.split(". ")[0].rstrip(".") + "."
            cases.append(Case("unrelated_claim", variant(finding=unrelated), loc, (IssueKind.WEAK,)))
        neg = negate(f.finding)
        if neg:
            cases.append(Case("negated_claim", variant(finding=neg), loc, (IssueKind.WEAK,)))
        num = change_number(f.finding)
        if num:
            cases.append(Case("changed_number", variant(finding=num), loc, (IssueKind.WEAK, IssueKind.NUMBER)))
    # inflated evidence: one paper rated strong
    a = copy.deepcopy(clean)
    a.evidence = [EvidenceClaim(claim="One study shows this works everywhere.", citations=[res[0].label],
                                strength=Strength.STRONG)]
    cases.append(Case("inflated_strength", a, "evidence[0]", (IssueKind.DOWNGRADED,)))
    return cases


def validator_caught(case: Case, rep) -> bool:
    return any(i.location == case.location and i.kind in case.expected for i in rep.issues)


def highlighter_caught(case: Case, res, scorer) -> bool:
    if not case.location.startswith("finding:"):
        return False
    f = next(x for x in case.answer.findings if f"finding:{x.paper_key}" == case.location)
    sents = highlight_text(f.finding, res, scorer, default_citations=f.citations)
    return any(s.support is not Support.SUPPORTED for s in sents)


def run_citation_checks(col, scorer) -> dict:
    retriever = HybridRetriever(col, use_embeddings=False)
    tally: dict[str, dict[str, int]] = {}
    clean_findings = clean_val_fp = clean_hl_fp = clean_answers = clean_answer_fp = 0
    for qi, item in enumerate(QUESTIONS):
        res = retriever.retrieve(item["q"], top_k=12)
        for case in make_cases(item["q"], res, qi):
            rep = validate(copy.deepcopy(case.answer), res)
            if case.kind == "clean":
                clean_answers += 1
                clean_answer_fp += not rep.ok
                for f in case.answer.findings:
                    clean_findings += 1
                    loc = f"finding:{f.paper_key}"
                    clean_val_fp += any(i.location == loc for i in rep.issues)
                    sents = highlight_text(f.finding, res, scorer, default_citations=f.citations)
                    clean_hl_fp += any(s.support is not Support.SUPPORTED for s in sents)
                continue
            t = tally.setdefault(case.kind, {"cases": 0, "validator": 0, "highlighter": 0, "either": 0})
            v = validator_caught(case, rep)
            h = highlighter_caught(case, res, scorer)
            t["cases"] += 1
            t["validator"] += v
            t["highlighter"] += h
            t["either"] += v or h
    rates = {k: {**v, **{f"{d}_rate": round(v[d] / v["cases"], 3) for d in ("validator", "highlighter", "either")}}
             for k, v in tally.items()}
    total = sum(v["cases"] for v in tally.values())
    either = sum(v["either"] for v in tally.values())
    return {
        "planted_errors": rates,
        "overall_detection_rate": round(either / total, 3) if total else None,
        "total_planted_errors": total,
        "clean": {
            "answers": clean_answers, "answers_with_error_flags": clean_answer_fp,
            "findings": clean_findings,
            "validator_false_alarm_rate": round(clean_val_fp / clean_findings, 3),
            "highlighter_false_alarm_rate": round(clean_hl_fp / clean_findings, 3),
        },
        "scorer": scorer.name,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
LABELS = {
    "invented_label": "Made-up citation (cites P99)",
    "wrong_paper_label": "Citation to another paper's passage",
    "unrelated_claim": "Claim from a different paper, original citation kept",
    "negated_claim": "Claim negated (\"reduced\" -> \"did not reduce\")",
    "changed_number": "Number changed (e.g. 58% -> 107.6%)",
    "inflated_strength": "Single paper rated \"strong\" evidence",
}


def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def write_markdown(results: dict, path: Path) -> None:
    r, c = results["retrieval"], results["citation_checks"]
    full = r["bm25_full"]
    lines = [
        "# Evaluation", "",
        f"Generated by `python eval/run_eval.py` on {results['meta']['date']} "
        f"(Python {results['meta']['python']}, {results['meta']['runtime_s']} s).", "",
        "What was run: retrieval over the 17-paper starter collection "
        f"({'BM25 and hybrid BM25 + embeddings' if 'hybrid_embeddings' in r else 'BM25 only'}), the citation "
        f"validator, and the truth highlighter with the **{c['scorer']}** scorer. No LLM was used, "
        "so these numbers describe the retrieval and checking parts of the system.", "",
        "## 1. Retrieval", "",
        f"{len(QUESTIONS)} questions, each hand-labelled with the papers a good answer should draw on "
        "([eval/questions.json](../eval/questions.json)).", "",
        "| Setting | Recall@3 | Recall@5 | Recall@12 | MRR | Hit@1 | Distinct papers in top 12 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, key in (("Hybrid: BM25 + bge-small embeddings (RRF)", "hybrid_embeddings"),
                      ("BM25 + synonyms + diversity", "bm25_full"),
                      ("Without synonym expansion", "ablation_no_synonyms"),
                      ("Without per-paper diversity", "ablation_no_diversity")):
        m = r.get(key)
        if not m:
            continue
        lines.append(f"| {name} | {pct(m['recall@3'])} | {pct(m['recall@5'])} | {pct(m['recall@12'])} | "
                     f"{m['mrr']:.2f} | {pct(m['hit@1'])} | {m['avg_distinct_papers@12']} |")
    lines += ["", "<details><summary>Per question</summary>", "",
              "| Question | First relevant rank | Recall@5 | Recall@12 |", "|---|---|---|---|"]
    for q in full["per_question"]:
        lines.append(f"| {q['q']} | {q['first_relevant_rank'] or '-'} | {pct(q['recall@5'])} | {pct(q['recall@12'])} |")
    lines += ["", "</details>", "",
              "## 2. Citation checking with planted errors", "",
              "For every question, the offline answer (each finding copied from its cited passage) is taken as a "
              "correct answer. Known errors are then planted in one finding or evidence claim at a time, and we "
              "check whether the citation validator and the truth highlighter flag that exact spot.", "",
              "| Planted error | Cases | Validator | Highlighter | Either |", "|---|---|---|---|---|"]
    for kind, label in LABELS.items():
        v = c["planted_errors"].get(kind)
        if v:
            lines.append(f"| {label} | {v['cases']} | {pct(v['validator_rate'])} | {pct(v['highlighter_rate'])} | "
                         f"**{pct(v['either_rate'])}** |")
    cl = c["clean"]
    lines += ["",
              f"**Overall:** {c['overall_detection_rate'] * 100:.1f}% of {c['total_planted_errors']} planted errors "
              "were caught by at least one check.", "",
              f"**False alarms on clean answers:** {cl['answers_with_error_flags']} of {cl['answers']} clean answers "
              f"got an error-level flag; the validator flagged {pct(cl['validator_false_alarm_rate'])} and the "
              f"highlighter {pct(cl['highlighter_false_alarm_rate'])} of {cl['findings']} correct findings.", "",
              ]
    base_path = ROOT / "eval" / "results_baseline.json"
    if base_path.exists() and not results["meta"].get("tag"):
        base = json.loads(base_path.read_text())["citation_checks"]
        lines += ["### Before and after the first evaluation", "",
                  "The first run (saved in [eval/results_baseline.json](../eval/results_baseline.json)) caught "
                  f"{pct(base['overall_detection_rate'])} overall. It missed every changed number and a third of "
                  "negations, so two checks were added: the validator now flags numbers a claim states that its "
                  "cited passages do not contain, and the highlighter caps such sentences at \"partial\" and "
                  "compares the *count* of negation words instead of their presence.", "",
                  "| Planted error | Before (either check) | After |", "|---|---|---|"]
        for kind, label in LABELS.items():
            b, a = base["planted_errors"].get(kind), c["planted_errors"].get(kind)
            if b and a:
                lines.append(f"| {label} | {pct(b['either_rate'])} | {pct(a['either_rate'])} |")
        lines += ["", "Because these fixes were designed after seeing the planted errors, the \"after\" numbers "
                  "are optimistic: the synthetic negations always insert \"did not\", and real model errors are "
                  "more varied.", ""]
    lines += [
              "## Reading these numbers", "",
              "- The questions and relevance labels were written by the same people who built the collection, and "
              "the collection is small (17 papers, 51 passages). Treat the retrieval scores as a sanity check.",
              "- The clean findings are copied word for word from their passages, so 0% false alarms is an easy "
              "case; an LLM's paraphrases will trigger some warnings that a reader has to dismiss.",
              "- Synonym expansion slightly lowers recall in the top 5 but raises the share of questions whose top "
              "result is relevant, and it spreads results over more papers.",
              "- Planted errors are synthetic. Real LLM mistakes are subtler, especially paraphrases that drift in "
              "meaning, which word overlap cannot see.",
              "- Changed numbers and negations are the hardest cases for the lexical scorer. They are what the NLI "
              "cross-encoder is for; rerun with `--nli-model` where the model can be downloaded.",
              "- Not measured here: answer quality from an LLM, how often a real LLM's citations are wrong, and "
              "the with-vs-without-docs comparison. Those need an LLM and are a next step.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-model", action="store_true", help="use the NLI cross-encoder instead of the lexical scorer")
    ap.add_argument("--embeddings", action="store_true", help="also evaluate hybrid retrieval with the embedding model")
    ap.add_argument("--tag", default="", help="write eval/results_<tag>.json and docs/EVALUATION_<tag>.md instead")
    args = ap.parse_args()
    start = time.time()
    col = load_starter()
    scorer = load_scorer() if args.nli_model else LexicalScorer()
    if args.nli_model and scorer.name == "lexical-overlap":
        print("WARNING: NLI model could not be loaded; using the lexical scorer.")
    results = {"retrieval": run_retrieval(col, args.embeddings), "citation_checks": run_citation_checks(col, scorer)}
    results["meta"] = {"tag": args.tag,"date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                       "python": platform.python_version(), "runtime_s": round(time.time() - start, 2),
                       "papers": len(col), "passages": len(col.passages), "questions": len(QUESTIONS)}
    suffix = f"_{args.tag}" if args.tag else ""
    (ROOT / "eval" / f"results{suffix}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, ROOT / "docs" / f"EVALUATION{suffix}.md")
    r, c = results["retrieval"]["bm25_full"], results["citation_checks"]
    print(f"Retrieval  recall@5={r['recall@5']:.2f} recall@12={r['recall@12']:.2f} MRR={r['mrr']:.2f}")
    print(f"Citations  caught {c['overall_detection_rate']:.0%} of {c['total_planted_errors']} planted errors; "
          f"clean false alarms: validator {c['clean']['validator_false_alarm_rate']:.0%}, "
          f"highlighter {c['clean']['highlighter_false_alarm_rate']:.0%}")
    for k, v in c["planted_errors"].items():
        print(f"  {k:20s} n={v['cases']:3d}  validator={v['validator_rate']:.0%}  "
              f"highlighter={v['highlighter_rate']:.0%}  either={v['either_rate']:.0%}")


if __name__ == "__main__":
    main()
