"""Load papers and split them into citable passages.

Sources:
  * a JSON collection (see data/papers/starter.json)
  * a local PDF (via pypdf)
  * an arXiv ID (via the `arxiv` package): abstract, or full PDF text
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DATA_DIR
from .models import Family, Paper, Passage, StudyType

STARTER_PATH = DATA_DIR / "papers" / "starter.json"

# A sentence ends at . ! or ? followed by whitespace and a capital, digit-free
# opener or bracket. This keeps decimals ("47.5%") and "et al. (2023)" intact.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"'])")
_ABBREVIATIONS = ("et al.", "e.g.", "i.e.", "Fig.", "Eq.", "vs.", "cf.", "Sec.")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking decimals or common abbreviations."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    protected = text
    for i, abbr in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbr, f"\x00{i}\x00")
    parts = _SENTENCE_END.split(protected)
    out = []
    for part in parts:
        for i, abbr in enumerate(_ABBREVIATIONS):
            part = part.replace(f"\x00{i}\x00", abbr)
        part = part.strip()
        if part:
            out.append(part)
    return out


def chunk_text(text: str, sentences_per_chunk: int = 3, overlap: int = 0, min_chars: int = 40) -> list[str]:
    """Group sentences into passages. Small passages make citations precise."""
    if sentences_per_chunk < 1:
        raise ValueError("sentences_per_chunk must be >= 1")
    if not 0 <= overlap < sentences_per_chunk:
        raise ValueError("overlap must be in [0, sentences_per_chunk)")
    sents = split_sentences(text)
    step = sentences_per_chunk - overlap
    chunks = []
    for start in range(0, len(sents), step):
        chunk = " ".join(sents[start:start + sentences_per_chunk])
        if len(chunk) >= min_chars or not chunks:
            chunks.append(chunk)
        elif overlap == 0:
            chunks[-1] = f"{chunks[-1]} {chunk}"  # fold a short tail into the previous passage
        if start + sentences_per_chunk >= len(sents):
            break
    return chunks


def clean_pdf_text(raw: str) -> str:
    """Undo common PDF extraction artefacts: hyphenated line breaks, stray newlines, references."""
    text = raw
    # Drop the bibliography first, while headings are still on their own line:
    # reference lists match queries well but are not evidence.
    m = None
    for m in re.finditer(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text):
        pass
    if m and m.start() > len(text) * 0.5:
        text = text[: m.start()]
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated words
    text = re.sub(r"\s*\n\s*", " ", text)                  # newlines -> spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_pdf_text(path: str | Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    return clean_pdf_text(raw)


def _arxiv_id(id_or_url: str) -> str:
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", id_or_url)
    if not m:
        raise ValueError(f"Not an arXiv ID or URL: {id_or_url!r}")
    return m.group(1)


def fetch_arxiv(id_or_url: str, full_text: bool = False, family: Family = Family.FRAMING,
                study_type: StudyType = StudyType.METHOD) -> Paper:
    """Fetch metadata (and the abstract, or optionally the full PDF text) from arXiv."""
    import arxiv

    aid = _arxiv_id(id_or_url)
    result = next(arxiv.Client().results(arxiv.Search(id_list=[aid])))
    first = result.authors[0].name.split()[-1] if result.authors else "Unknown"
    authors = f"{first} et al." if len(result.authors) > 2 else " & ".join(a.name.split()[-1] for a in result.authors)
    text = result.summary
    if full_text:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = result.download_pdf(dirpath=tmp)
            text = load_pdf_text(pdf_path)
    return Paper(
        key=f"{first.lower()}{result.published.year}_{aid.replace('.', '')}",
        title=result.title.strip(),
        authors=authors,
        year=result.published.year,
        url=f"https://arxiv.org/abs/{aid}",
        family=family,
        study_type=study_type,
        text=text,
    )


@dataclass
class Collection:
    """Papers plus the passages they were split into."""

    papers: dict[str, Paper] = field(default_factory=dict)
    passages: list[Passage] = field(default_factory=list)

    def add(self, paper: Paper, passages: list[str] | None = None, sentences_per_chunk: int = 3) -> list[Passage]:
        if paper.key in self.papers:
            raise ValueError(f"Duplicate paper key: {paper.key}")
        texts = passages if passages is not None else chunk_text(paper.text, sentences_per_chunk)
        if not texts:
            raise ValueError(f"No passages could be made for {paper.key!r}; is the text empty?")
        self.papers[paper.key] = paper
        new = [Passage(paper_key=paper.key, index=i, text=t) for i, t in enumerate(texts)]
        self.passages.extend(new)
        return new

    def remove(self, key: str) -> None:
        self.papers.pop(key, None)
        self.passages = [p for p in self.passages if p.paper_key != key]

    def paper_of(self, passage: Passage) -> Paper:
        return self.papers[passage.paper_key]

    def __len__(self) -> int:
        return len(self.papers)

    # --- persistence -----------------------------------------------------
    def to_json(self) -> dict:
        by_paper: dict[str, list[str]] = {}
        for p in self.passages:
            by_paper.setdefault(p.paper_key, []).append(p.text)
        papers = []
        for key, paper in self.papers.items():
            d = asdict(paper)
            d.pop("text")
            d["family"] = paper.family.value
            d["study_type"] = paper.study_type.value
            d["passages"] = by_paper.get(key, [])
            papers.append(d)
        return {"papers": papers}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = STARTER_PATH) -> "Collection":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        col = cls()
        for d in data["papers"]:
            paper = Paper(
                key=d["key"], title=d["title"], authors=d.get("authors", "Unknown"),
                year=d.get("year", "n.d."), url=d.get("url", ""),
                family=Family(d.get("family", "framing")),
                study_type=StudyType(d.get("study_type", "method")),
                text=d.get("text", ""),
            )
            col.add(paper, passages=d.get("passages"))
        return col


def load_starter() -> Collection:
    return Collection.load(STARTER_PATH)
