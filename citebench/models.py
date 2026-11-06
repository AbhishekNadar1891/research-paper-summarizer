"""Core data types shared across the pipeline.

The pipeline is:  Paper -> Passage (ingest) -> RetrievedPassage (retrieval)
-> Answer (generation) -> ValidationReport (citation checks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Family(str, Enum):
    """What kind of approach a paper takes. Used to group cross-paper evidence."""

    RETRIEVAL = "retrieval"          # grounding generation in retrieved documents
    VERIFICATION = "verification"    # checking / revising a draft
    INFERENCE = "inference"          # decoding or activation changes at inference time
    DETECTION = "detection"          # uncertainty, self-consistency, calibration
    TRAINING = "training"            # fine-tuning, preference optimisation
    MEASUREMENT = "measurement"      # benchmarks and evaluation metrics
    FRAMING = "framing"              # theory and surveys


class StudyType(str, Enum):
    METHOD = "method"
    EVALUATION = "evaluation"
    ANALYSIS = "analysis"
    THEORY = "theory"
    SURVEY = "survey"


class Strength(str, Enum):
    """How well a cross-paper claim is supported, by number of independent papers."""

    STRONG = "strong"      # 3+ papers agree
    MODERATE = "moderate"  # 2 papers
    LIMITED = "limited"    # 1 paper

    @classmethod
    def from_paper_count(cls, n: int) -> "Strength":
        if n >= 3:
            return cls.STRONG
        if n == 2:
            return cls.MODERATE
        return cls.LIMITED

    @property
    def min_papers(self) -> int:
        return {"strong": 3, "moderate": 2, "limited": 1}[self.value]


@dataclass
class Paper:
    key: str
    title: str
    authors: str
    year: int | str
    url: str = ""
    family: Family = Family.FRAMING
    study_type: StudyType = StudyType.METHOD
    text: str = ""  # full text or abstract; split into passages at ingest time

    def citation_label(self) -> str:
        return f"{self.authors} ({self.year})"


@dataclass
class Passage:
    paper_key: str
    index: int  # position within the paper
    text: str

    @property
    def uid(self) -> str:
        return f"{self.paper_key}#{self.index}"


@dataclass
class RetrievedPassage:
    """A passage returned for one query, labelled P1, P2, ... for citation."""

    label: str
    passage: Passage
    paper: Paper
    score: float


@dataclass
class Finding:
    """What one paper found, citing only that paper's passages."""

    paper_key: str
    finding: str
    citations: list[str] = field(default_factory=list)
    study_type: StudyType | None = None


@dataclass
class EvidenceClaim:
    """A conclusion drawn across papers."""

    claim: str
    citations: list[str] = field(default_factory=list)
    strength: Strength = Strength.LIMITED
    why: str = ""
    downgraded: bool = False


@dataclass
class Tension:
    point: str
    citations: list[str] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    summary: str
    findings: list[Finding] = field(default_factory=list)
    evidence: list[EvidenceClaim] = field(default_factory=list)
    tensions: list[Tension] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    grounded: bool = True  # False for the "without documents" answer
