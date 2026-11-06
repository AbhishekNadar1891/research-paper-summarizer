"""Multi-document retrieval: hybrid BM25 + dense embeddings with per-paper diversity.

Why hybrid?  BM25 is precise on exact terms ("TruthfulQA", "DoLa"), while
embeddings match meaning ("reduce hallucination" ~ "improve factuality").
The two rankings are merged with reciprocal rank fusion (RRF), which only
uses rank positions, so the very different score scales never need tuning.

Why diversity?  A summarizer that sees five passages from one paper writes a
summary of one paper. We take the best passage from each paper first and
only then allow a second passage per paper.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .ingest import Collection
from .models import RetrievedPassage

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------
STOPWORDS = frozenset(
    """a an the of and or to in on for with by from as at is are was were be been being that this these
    those it its into than then their there they them what which who whom how why when where do does did
    can could should would will may might has have had not no but if so such using use used about over
    more most less other some any each per via also we our us i you your researchers paper papers""".split()
)

_SUFFIXES = ("ations", "ation", "ates", "ated", "ate", "ments", "ment", "ness", "ities", "ity", "ings", "ing",
             "ions", "ion", "ers", "er", "ies", "ed", "es", "ly", "s")

# Domain synonyms, applied to the query only. Weighted lower than real query terms.
_SYNONYM_WORDS = [
    "hallucination factuality truthfulness unsupported fabricated error",
    "reduce mitigate decrease improve correct lower fix",
    "detect flag identify uncertainty confidence calibration consistency",
    "citation cite attribution",
    "retrieval grounding index documents",
    "verification revision edit fact-check",
    "fine-tuning training preference",
]
SYNONYM_WEIGHT = 0.35


def stem(word: str) -> str:
    """A deliberately small suffix stripper: enough to match hallucinate/hallucination/hallucinations."""
    w = word.lower()
    for suf in _SUFFIXES:
        if len(w) - len(suf) >= 4 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 5 and w.endswith("e"):
        w = w[:-1]
    return w


SYNONYMS: list[set[str]] = [{stem(w) for w in line.split()} for line in _SYNONYM_WORDS]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def expand_query(tokens: Sequence[str]) -> dict[str, float]:
    """Query term -> weight. Original terms weigh 1, synonyms SYNONYM_WEIGHT."""
    weights: dict[str, float] = {}
    for t in tokens:
        weights[t] = weights.get(t, 0) + 1.0
    for t in list(weights):
        for group in SYNONYMS:
            if t in group:
                for s in group:
                    weights.setdefault(s, SYNONYM_WEIGHT)
    return weights


# --------------------------------------------------------------------------
# Individual retrievers
# --------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, collection: Collection):
        self.collection = collection
        docs = [tokenize(f"{p.text} {collection.paper_of(p).title}") for p in collection.passages]
        self._bm25 = BM25Okapi(docs) if docs else None

    def scores(self, query: str) -> np.ndarray:
        n = len(self.collection.passages)
        if not self._bm25 or not n:
            return np.zeros(n)
        total = np.zeros(n)
        for term, w in expand_query(tokenize(query)).items():
            total += w * self._bm25.get_scores([term])
        return total


Encoder = Callable[[list[str]], np.ndarray]


def load_sentence_transformer(model_name: str) -> Encoder | None:
    """Return an encoder for `model_name`, or None if it can't be loaded (offline, not installed)."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001 - any failure means "fall back to BM25"
        log.warning("Embedding model unavailable (%s); using BM25 only.", exc)
        return None

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))

    return encode


class EmbeddingRetriever:
    # bge models expect this instruction in front of queries (not passages).
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, collection: Collection, encoder: Encoder, query_prefix: str | None = None):
        self.collection = collection
        self.encode = encoder
        self.query_prefix = self.QUERY_PREFIX if query_prefix is None else query_prefix
        texts = [f"{collection.paper_of(p).title}. {p.text}" for p in collection.passages]
        self._matrix = _normalize(encoder(texts)) if texts else np.zeros((0, 1))

    def scores(self, query: str) -> np.ndarray:
        if not len(self._matrix):
            return np.zeros(0)
        q = _normalize(self.encode([self.query_prefix + query]))[0]
        return self._matrix @ q


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


# --------------------------------------------------------------------------
# Fusion + diversity
# --------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Merge ranked lists of passage indices: score(d) = sum over lists of 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


def select_diverse(ranked: Sequence[tuple[int, float]], paper_of: Callable[[int], str],
                   top_k: int, max_per_paper: int) -> list[tuple[int, float]]:
    """Pass 1: best passage of each paper, in rank order. Pass 2: fill up to max_per_paper each."""
    chosen: list[tuple[int, float]] = []
    count: dict[str, int] = {}
    for idx, score in ranked:
        key = paper_of(idx)
        if key not in count:
            count[key] = 1
            chosen.append((idx, score))
            if len(chosen) >= top_k:
                break
    picked = {i for i, _ in chosen}
    for idx, score in ranked:
        if len(chosen) >= top_k:
            break
        key = paper_of(idx)
        if idx in picked or count[key] >= max_per_paper:
            continue
        count[key] += 1
        chosen.append((idx, score))
        picked.add(idx)
    order = {idx: r for r, (idx, _) in enumerate(ranked)}
    return sorted(chosen, key=lambda x: order[x[0]])


@dataclass
class RetrievalConfig:
    top_k: int = settings.top_k
    max_per_paper: int = settings.max_per_paper
    candidate_pool: int = 40      # how deep each ranking goes into fusion
    min_relative_bm25: float = 0.15  # in BM25-only mode, drop passages far below the best match


class HybridRetriever:
    """BM25 + embeddings via RRF. Falls back to BM25 alone if no encoder is available."""

    def __init__(self, collection: Collection, encoder: Encoder | None = None,
                 use_embeddings: bool = True, config: RetrievalConfig | None = None):
        self.collection = collection
        self.config = config or RetrievalConfig()
        self.bm25 = BM25Retriever(collection)
        if use_embeddings and encoder is None:
            encoder = load_sentence_transformer(settings.embedding_model)
        self.dense = EmbeddingRetriever(collection, encoder) if (use_embeddings and encoder) else None

    @property
    def mode(self) -> str:
        return "hybrid" if self.dense else "bm25"

    def rank(self, query: str) -> list[tuple[int, float]]:
        cfg = self.config
        bm = self.bm25.scores(query)
        bm_order = [int(i) for i in np.argsort(-bm) if bm[i] > 0][: cfg.candidate_pool]
        if not self.dense:
            if not bm_order:
                return []
            floor = bm[bm_order[0]] * cfg.min_relative_bm25
            return [(i, float(bm[i])) for i in bm_order if bm[i] >= floor]
        dn = self.dense.scores(query)
        dn_order = [int(i) for i in np.argsort(-dn)][: cfg.candidate_pool]
        fused = reciprocal_rank_fusion([bm_order, dn_order])
        return sorted(fused.items(), key=lambda x: -x[1])

    def retrieve(self, query: str, top_k: int | None = None, max_per_paper: int | None = None) -> list[RetrievedPassage]:
        if not query.strip():
            return []
        passages = self.collection.passages
        chosen = select_diverse(
            self.rank(query),
            paper_of=lambda i: passages[i].paper_key,
            top_k=top_k or self.config.top_k,
            max_per_paper=max_per_paper or self.config.max_per_paper,
        )
        return [
            RetrievedPassage(label=f"P{n}", passage=passages[i], paper=self.collection.paper_of(passages[i]), score=s)
            for n, (i, s) in enumerate(chosen, start=1)
        ]
