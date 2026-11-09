"""Compare the truth highlighter's labels with hand judgements.  Run:  python eval/score_audit.py

Reads eval/audit_qwen2-5-7b.json (hand labels) and the results file it refers to (passages and
automatic labels). Reports how well the automatic labels match, both as recorded in the run and
with the close-restatement rule (citebench.nli.is_close_restatement) applied afterwards.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from citebench.nli import is_close_restatement  # noqa: E402


def score(pairs):
    """pairs: (needs_flag, flagged). Returns counts and rates."""
    tp = sum(n and f for n, f in pairs)
    fn = sum(n and not f for n, f in pairs)
    fp = sum(f and not n for n, f in pairs)
    tn = sum(not n and not f for n, f in pairs)
    return {"sentences": len(pairs), "real_problems": tp + fn, "caught": tp, "missed": fn,
            "false_alarms": fp, "correct_sentences": fp + tn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "accuracy": (tp + tn) / len(pairs) if pairs else None}


def main(audit_path: str = "eval/audit_qwen2-5-7b.json") -> dict:
    audit = json.loads((ROOT / audit_path).read_text())
    results = json.loads((ROOT / audit["results_file"]).read_text())
    passages = {q["q"]: {p["label"]: p["text"] for p in q["passages"]} for q in results["questions"]}
    as_run, with_rule = [], []
    for it in audit["items"]:
        if it["hand_label"] == "excluded":
            continue
        needs_flag = it["hand_label"] != "supported"
        flagged = it["auto_label"] != "supported"
        as_run.append((needs_flag, flagged))
        src = " ".join(passages[it["question"]].get(c, "") for c in it["citations"])
        with_rule.append((needs_flag, flagged and not is_close_restatement(it["sentence"], src)))
    out = {"as_run": score(as_run), "with_restatement_rule": score(with_rule),
           "hand_supported_share": sum(not n for n, _ in as_run) / len(as_run)}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
