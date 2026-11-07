# Citebench: Research Paper Summarizer with Checked Citations

Ask a research question, such as *"What approaches have researchers used to reduce hallucinations in language models?"*. Citebench retrieves passages from many papers, writes an answer in which every claim cites a passage, checks those citations in code, and keeps **what each paper found** separate from **what the overall evidence supports**.

![Answer view: color-coded sentences, per-paper findings and retrieved passages](docs/screenshots/answer.png)

## Features

| | What it does | Where |
|---|---|---|
| **Multi-document retrieval** | Hybrid BM25 + dense embeddings merged with reciprocal rank fusion. It takes the best passage from each paper before any second passage, so answers draw on many papers. | `citebench/retrieval.py` |
| **Grounded answers** | The LLM sees only labelled passages (P1, P2, …) and returns JSON: a cited summary, per-paper findings, cross-paper evidence rated *strong / moderate / limited*, tensions and gaps. | `citebench/generate.py` |
| **Citation validator** | Flags made-up citations, findings that cite another paper's passage, findings about papers that weren't retrieved, weak support, uncited claims, numbers the cited passages don't contain, and evidence rated stronger than its distinct papers allow (lowered automatically). | `citebench/validate.py` |
| **Truth highlighter** | Scores each answer sentence against its cited passages with an NLI cross-encoder and labels it supported, partial, unsupported or contradicted. Hovering shows the exact source sentence. | `citebench/nli.py` |
| **With docs vs. without docs** | Asks the same question with no documents, splits that answer into atomic claims, and checks each against the whole collection: backed, partial, not in papers, or contradicted. | `citebench/compare.py` |
| **Reviewer 2** | A second pass critiques the draft for overclaiming, cherry-picking, misread numbers and inflated ratings, then a revise pass fixes it. | `citebench/review.py` |
| **Library** | Add papers from PDF or arXiv. A 17-paper starter set on LLM hallucination is included. | `citebench/ingest.py` |
| **Offline mode** | Without an API key, answers are extractive (each paper's best sentence) and every check still runs. | |

## How it works

```mermaid
flowchart LR
    Q([Question]) --> R[Hybrid retrieval<br/>BM25 + embeddings, RRF<br/>per-paper diversity]
    C[(Papers<br/>PDF / arXiv / JSON)] --> I[Ingest<br/>2–3 sentence passages] --> R
    R -->|P1…Pn| G[Grounded answer<br/>LLM, JSON]
    G --> V[Citation validator]
    V --> RV[Reviewer 2<br/>critique + revise]
    RV --> V2[Re-validate<br/>drop invalid citations]
    V2 --> H[Truth highlighter<br/>NLI per sentence]
    Q --> CB[Closed-book answer<br/>no documents]
    CB --> CC[Check each claim<br/>against the papers]
    R --> CC
    H --> UI[Streamlit app]
    CC --> UI
```

The design follows one rule: **the LLM is anchored to the documents but never trusted.** Instructions reduce hallucinated citations without removing them, so every citation is checked in code before the answer is shown.

## Results

From `python eval/run_eval.py`, which needs no API key or model downloads. Full report: [docs/EVALUATION.md](docs/EVALUATION.md).

**Retrieval** (15 hand-labelled questions, BM25 mode):

| Recall@5 | Recall@12 | MRR | Hit@1 | Distinct papers in top 12 |
|---|---|---|---|---|
| 75% | 87% | 0.94 | 93% | 11.1 (7.1 without per-paper diversity) |

**Citation checking** (705 errors planted in correct answers):

| Planted error | Caught |
|---|---|
| Made-up citation | 100% |
| Citation to another paper's passage | 100% |
| Claim from a different paper | 100% |
| Negated claim ("reduced" → "did not reduce") | 100% |
| Changed number | 98% |
| Single paper rated "strong" | 100% |
| **False alarms on correct findings** | **0%** |

The first evaluation run caught 86%. It missed every changed number and a third of negations, which led to the number check and a better negation check. Because those checks were designed after seeing the planted errors, treat the current numbers as optimistic. Real LLM mistakes are subtler, and LLM-mode answer quality hasn't been evaluated yet.

## Quick start

```bash
git clone https://github.com/AbhishekNadar1891/research-paper-summarizer
cd research-paper-summarizer
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # optional: add ANTHROPIC_API_KEY for LLM answers
streamlit run app.py
```

You can also paste an API key into the app's sidebar for one session. On the first run, the embedding model (`BAAI/bge-small-en-v1.5`) and NLI model (`cross-encoder/nli-deberta-v3-small`) download from Hugging Face. If they can't load, the app falls back to BM25-only search and a lexical scorer.

```bash
pytest                       # 73 tests, no network or API key needed
python eval/run_eval.py      # regenerates docs/EVALUATION.md
```

### Use it from Python

```python
from citebench.pipeline import Pipeline

result = Pipeline().run("Does retrieval augmentation improve factuality?")
print(result.answer.summary)
for f in result.answer.findings:
    print(f.paper_key, f.finding, f.citations)
print(result.report.summary())          # citation check
print(result.highlights.counts())       # supported / partial / unsupported ...
```

## Screenshots

| Evidence ratings and per-paper findings | Library |
|---|---|
| ![Evidence](docs/screenshots/evidence.png) | ![Library](docs/screenshots/library.png) |

## Design decisions

- **Small passages (2–3 sentences)** make citations precise. A citation to a whole page can't be checked.
- **Reciprocal rank fusion** merges BM25 and embedding rankings using rank positions only, so their different score scales never need tuning.
- **Per-paper diversity:** without it, the top 12 passages covered 7.1 papers on average instead of 11.1, and recall dropped from 87% to 77%. A summarizer that sees one paper five times writes a summary of one paper.
- **Evidence strength is counted, not asked for.** The model proposes a rating, and the validator lowers it to what the number of distinct cited papers allows (3+ strong, 2 moderate, 1 limited).
- **"Not in these papers" is not "false".** In the with-vs-without view, an unbacked claim may still be true; it just can't be checked against this collection, and the UI says so.
- **Pluggable models.** The LLM, the embedding encoder and the NLI scorer are injected, which is how the tests run on fakes with no network.

## Limitations

- The starter collection is small (17 papers, 51 passages), and its passages are paraphrased summaries, not the papers' own text.
- Retrieval questions and relevance labels were written by the project authors.
- The lexical fallback can't detect paraphrases that change meaning. The NLI model is better but still weak on numbers, which is why numbers are checked separately.
- LLM answer quality and the with-vs-without comparison have only been tested with a simulated LLM so far.
- Added papers live in the app session. Use `Collection.save()` to keep a collection on disk.

## Project layout

```
app.py                  Streamlit app
citebench/
  ingest.py             PDF / arXiv / JSON loading, passage chunking
  retrieval.py          BM25 + embeddings, RRF, per-paper diversity
  llm.py                Claude wrapper, tolerant JSON parsing
  generate.py           grounded answer + offline extractive fallback
  validate.py           citation validator
  nli.py                truth highlighter
  compare.py            with docs vs. without docs
  review.py             Reviewer 2
  pipeline.py           end-to-end pipeline
data/papers/starter.json
eval/                   questions, evaluation script, results
tests/                  pytest suite (fake LLM, no network)
docs/                   evaluation report, roadmap, screenshots
```
