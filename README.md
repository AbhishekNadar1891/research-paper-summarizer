# Citebench: Research Paper Summarizer

Ask a research question, such as *"What approaches have researchers used to reduce hallucinations in language models?"*. Citebench retrieves passages from many papers, summarizes what each paper found with citations, and separates that from what the overall evidence supports.

> 🚧 Work in progress. See [docs/ROADMAP.md](docs/ROADMAP.md) for what's built so far.

## What makes it different

- **Per-paper findings vs. overall evidence.** Each paper's result is reported on its own. Cross-paper conclusions get a strength rating based on how many independent papers back them.
- **Citation validation.** Every citation the LLM writes is checked in code. Citations that point to passages that were never retrieved, that are attributed to the wrong paper, or whose passage doesn't support the claim get flagged.
- **Truth highlighter.** An NLI model scores each sentence of the answer against its cited passage, and the sentence is colored by how well the passage supports it.
- **With docs vs. without docs.** The same question is answered from the model's memory alone and from the retrieved papers, side by side, with the claims the papers don't back up marked.
- **Reviewer 2.** A second pass critiques the summary for overclaiming before you see it.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
pytest
```

The app entry point arrives in a later commit.

## Project layout

```
citebench/        the library: ingest, retrieval, generation, validation
tests/            unit tests
data/papers/      paper collection (starter set included)
docs/             roadmap and design notes
```
