"""Evaluate the full pipeline with a real LLM.  Run:  python eval/run_llm_eval.py

Uses the provider configured in .env (e.g. CITEBENCH_PROVIDER=ollama, CITEBENCH_MODEL=qwen2.5:7b).
For every question in eval/questions.json it runs retrieve -> generate -> validate -> Reviewer 2 ->
highlight -> with-vs-without-docs, and records:

  * how often the LLM's own citations are invalid (made up / wrong paper) or weak
  * how often it rates evidence stronger than its cited papers allow
  * how many answer sentences the cited passages support (truth highlighter)
  * what Reviewer 2 changes (draft vs revised)
  * how much of a memory-only answer the papers back
  * whether the answer covers the papers a good answer should use
  * failures and timings

Writes eval/results_llm_<model>.json and docs/EVALUATION_LLM.md.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from citebench.config import settings  # noqa: E402
from citebench.ingest import load_starter  # noqa: E402
from citebench.llm import get_llm  # noqa: E402
from citebench.nli import LexicalScorer, Support, highlight_answer, load_scorer  # noqa: E402
from citebench.pipeline import Pipeline  # noqa: E402
from citebench.validate import IssueKind  # noqa: E402

QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text())["questions"]


def support_counts(highlights) -> dict[str, int]:
    return highlights.counts() if highlights else {}


def cited_support_rate(counts: dict[str, int]) -> float | None:
    cited = sum(v for k, v in counts.items() if k != Support.UNCITED.value)
    return counts.get(Support.SUPPORTED.value, 0) / cited if cited else None


def issue_count(report, kind: IssueKind) -> int:
    return sum(i.kind is kind for i in report.issues) if report else 0


def evaluate_question(pipe: Pipeline, item: dict, with_compare: bool) -> dict:
    t0 = time.perf_counter()
    r = pipe.run(item["q"], with_review=True, with_highlights=True, with_comparison=with_compare)
    draft_rep = r.draft_report or r.report
    final_rep = r.report
    draft_hl = highlight_answer(r.draft, r.retrieved, pipe.scorer) if r.draft is not None else r.highlights
    relevant = set(item["relevant"])
    retrieved_papers = {x.paper.key for x in r.retrieved}
    answered_papers = {f.paper_key for f in r.answer.findings}
    row = {
        "q": item["q"],
        "seconds": round(time.perf_counter() - t0, 1),
        "timings": r.timings,
        "retrieved_papers": len(retrieved_papers),
        "relevant_retrieved": len(relevant & retrieved_papers) / len(relevant),
        "relevant_in_answer": len(relevant & answered_papers) / len(relevant),
        "findings": len(r.answer.findings),
        "evidence_claims": len(r.answer.evidence),
        "draft": {
            "citations": draft_rep.total, "invented": draft_rep.invented,
            "misattributed": draft_rep.misattributed, "weak": draft_rep.weak,
            "number_issues": issue_count(draft_rep, IssueKind.NUMBER),
            "uncited": issue_count(draft_rep, IssueKind.UNCITED),
            "unknown_paper": issue_count(draft_rep, IssueKind.UNKNOWN_PAPER),
            "strength_downgrades": issue_count(draft_rep, IssueKind.DOWNGRADED),
            "support": support_counts(draft_hl),
        },
        "final": {
            "citations": final_rep.total, "invented": final_rep.invented,
            "misattributed": final_rep.misattributed, "weak": final_rep.weak,
            "number_issues": issue_count(final_rep, IssueKind.NUMBER),
            "strength_downgrades": issue_count(final_rep, IssueKind.DOWNGRADED),
            "support": support_counts(r.highlights),
        },
        "review": {"verdict": r.review.verdict if r.review else None,
                   "notes": [n.kind.value for n in r.review.notes] if r.review else [],
                   "revised": bool(r.review and r.review.revised is not None)},
        "summary": r.answer.summary,
    }
    if r.comparison:
        row["memory_only"] = {**r.comparison.stats(),
                              "claims_detail": [{"claim": c.claim, "status": c.status.value,
                                                 "evidence": c.evidence.sentence if c.evidence else None}
                                                for c in r.comparison.claims]}
    return row


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]

    def stage(name):
        tot = sum(r[name]["citations"] for r in ok)
        bad = sum(r[name]["invented"] + r[name]["misattributed"] for r in ok)
        sup = {k: sum(r[name]["support"].get(k, 0) for r in ok) for k in [s.value for s in Support]}
        return {
            "citations": tot,
            "invalid_citation_rate": bad / tot if tot else None,
            "invented": sum(r[name]["invented"] for r in ok),
            "misattributed": sum(r[name]["misattributed"] for r in ok),
            "weak_rate": sum(r[name]["weak"] for r in ok) / tot if tot else None,
            "number_issues": sum(r[name]["number_issues"] for r in ok),
            "strength_downgrades": sum(r[name]["strength_downgrades"] for r in ok),
            "answers_with_invalid_citation": sum((r[name]["invented"] + r[name]["misattributed"]) > 0 for r in ok),
            "sentence_support": sup,
            "supported_share_of_cited": cited_support_rate(sup),
        }

    mem = [r["memory_only"] for r in ok if "memory_only" in r]
    out = {
        "questions": len(rows), "succeeded": len(ok), "failed": len(rows) - len(ok),
        "avg_seconds": mean([r["seconds"] for r in ok]),
        "relevant_retrieved": mean([r["relevant_retrieved"] for r in ok]),
        "relevant_in_answer": mean([r["relevant_in_answer"] for r in ok]),
        "draft": stage("draft"), "final": stage("final"),
        "reviewer": {
            "verdicts": {v: sum(r["review"]["verdict"] == v for r in ok) for v in ("accept", "minor", "major")},
            "revised": sum(r["review"]["revised"] for r in ok),
            "note_kinds": {k: sum(r["review"]["notes"].count(k) for r in ok)
                           for k in sorted({n for r in ok for n in r["review"]["notes"]})},
        },
    }
    if mem:
        claims = sum(m["claims"] for m in mem)
        out["memory_only"] = {
            "answers": len(mem), "claims": claims,
            **{k: sum(m[k] for m in mem) for k in ("backed", "partial", "not_in_papers", "contradicted")},
            "backed_or_partial_rate": (sum(m["backed"] + m["partial"] for m in mem) / claims) if claims else None,
        }
    return out


def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def write_markdown(res: dict, path: Path) -> None:
    a, m = res["aggregate"], res["meta"]
    d, f = a["draft"], a["final"]
    L = [f"# Evaluation with a real LLM: `{m['model']}`", "",
         f"Generated by `python eval/run_llm_eval.py` on {m['date']} · provider **{m['provider']}** · "
         f"retrieval **{m['retrieval_mode']}** · sentence scorer **{m['scorer']}** · "
         f"{a['succeeded']}/{a['questions']} questions answered · {a['avg_seconds']:.0f} s per question on average "
         f"({m['machine']}).", "",
         "## The LLM's own citations (draft, before any fixing)", "",
         "| Measure | Result |", "|---|---|",
         f"| Citations written | {d['citations']} |",
         f"| Invalid (made up or pointing at another paper) | {pct(d['invalid_citation_rate'])} "
         f"({d['invented']} made up, {d['misattributed']} wrong paper) |",
         f"| Answers with at least one invalid citation | {d['answers_with_invalid_citation']} of {a['succeeded']} |",
         f"| Weak support (little overlap with the cited passage) | {pct(d['weak_rate'])} |",
         f"| Numbers not found in the cited passage | {d['number_issues']} |",
         f"| Evidence ratings lowered (too few papers for the rating) | {d['strength_downgrades']} |",
         f"| Cited sentences the passages support (truth highlighter) | {pct(d['supported_share_of_cited'])} |",
         "", "## After Reviewer 2 and the validator", "",
         "| Measure | Draft | Final |", "|---|---|---|",
         f"| Invalid citations | {pct(d['invalid_citation_rate'])} | {pct(f['invalid_citation_rate'])} "
         "(then removed before display) |",
         f"| Weak support | {pct(d['weak_rate'])} | {pct(f['weak_rate'])} |",
         f"| Cited sentences supported | {pct(d['supported_share_of_cited'])} | {pct(f['supported_share_of_cited'])} |",
         "", f"Reviewer 2 verdicts: {a['reviewer']['verdicts']}; it rewrote {a['reviewer']['revised']} of "
         f"{a['succeeded']} answers. Notes raised: {a['reviewer']['note_kinds'] or 'none'}.", "",
         "Sentence labels (final): " + ", ".join(f"{k} {v}" for k, v in f["sentence_support"].items()), "",
         "## Coverage", "",
         f"- Relevant papers that retrieval found: {pct(a['relevant_retrieved'])}",
         f"- Relevant papers the final answer reports a finding for: {pct(a['relevant_in_answer'])}", ""]
    if "memory_only" in a:
        mo = a["memory_only"]
        L += ["## With docs vs. without docs", "",
              f"The model answered each question from memory; its {mo['claims']} claims were checked against the "
              "collection:", "",
              "| Backed | Partly backed | Not in these papers | Contradicted |", "|---|---|---|---|",
              f"| {mo['backed']} | {mo['partial']} | {mo['not_in_papers']} | {mo['contradicted']} |", "",
              f"{pct(mo['backed_or_partial_rate'])} of memory-only claims were backed at least partly by the papers. "
              "\"Not in these papers\" means unverifiable against this 17-paper collection, not false.", ""]
    L += ["## Per question", "", "| Question | Citations | Invalid | Weak | Supported | Revised | Seconds |",
          "|---|---|---|---|---|---|---|"]
    for r in res["questions"]:
        if "error" in r:
            L.append(f"| {r['q']} | failed: {r['error'][:80]} | | | | | |")
            continue
        dd = r["draft"]
        L.append(f"| {r['q']} | {dd['citations']} | {dd['invented'] + dd['misattributed']} | {dd['weak']} | "
                 f"{pct(cited_support_rate(dd['support']))} | {'yes' if r['review']['revised'] else 'no'} | {r['seconds']} |")
    L += ["", "## Caveats", "",
          "- 15 questions over a 17-paper collection of paraphrased passages: a small, author-built test.",
          "- \"Weak support\" and the highlighter are automatic judgements; spot-check them against the passages "
          "(per-question details, including every answer and memory-only claim, are in the JSON file).",
          "- Results depend on the model, its quantisation and the prompt; rerun after changing any of them.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--no-compare", action="store_true", help="skip the with-vs-without-docs step")
    ap.add_argument("--embeddings", action="store_true", help="hybrid retrieval (downloads the embedding model)")
    ap.add_argument("--nli-model", action="store_true", help="NLI cross-encoder for the highlighter")
    args = ap.parse_args()

    llm = get_llm()
    if llm is None:
        sys.exit("No LLM configured. Set CITEBENCH_PROVIDER (e.g. ollama) in .env; see .env.example.")
    scorer = load_scorer() if args.nli_model else LexicalScorer()
    pipe = Pipeline(load_starter(), llm=llm, scorer=scorer, use_embeddings=args.embeddings)
    model = getattr(llm, "model", type(llm).__name__)
    print(f"Model {model} via {settings.provider} · retrieval {pipe.retriever.mode} · scorer {scorer.name}")

    qs = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    rows = []
    for i, item in enumerate(qs, 1):
        print(f"[{i}/{len(qs)}] {item['q']}", flush=True)
        try:
            row = evaluate_question(pipe, item, not args.no_compare)
            d = row["draft"]
            print(f"      {row['seconds']}s · {d['citations']} citations, {d['invented'] + d['misattributed']} invalid, "
                  f"{d['weak']} weak · revised={row['review']['revised']}")
        except Exception as exc:  # noqa: BLE001 - record and continue
            row = {"q": item["q"], "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}
            print(f"      FAILED: {row['error'][:200]}")
        rows.append(row)

    res = {
        "meta": {"model": model, "provider": settings.provider, "retrieval_mode": pipe.retriever.mode,
                 "scorer": scorer.name, "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"},
        "aggregate": aggregate(rows),
        "questions": rows,
    }
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    (ROOT / "eval" / f"results_llm_{slug}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_markdown(res, ROOT / "docs" / "EVALUATION_LLM.md")
    a = res["aggregate"]
    print(f"\nDone: {a['succeeded']}/{a['questions']} answered. Draft invalid citations "
          f"{pct(a['draft']['invalid_citation_rate'])}, weak {pct(a['draft']['weak_rate'])}, "
          f"supported sentences {pct(a['draft']['supported_share_of_cited'])} -> "
          f"{pct(a['final']['supported_share_of_cited'])} after review.")
    print(f"Wrote eval/results_llm_{slug}.json and docs/EVALUATION_LLM.md")


if __name__ == "__main__":
    main()
