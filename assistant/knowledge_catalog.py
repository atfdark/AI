"""Curated offline knowledge dataset catalog and bundle planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    """Describes a recommended dataset for offline assistant intelligence."""

    dataset_id: str
    name: str
    layer: str
    summary: str
    urls: Tuple[str, ...]
    size_gb_min: float
    size_gb_max: float
    recommended_trust: float
    tags: Tuple[str, ...]
    requires_manual_access: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "layer": self.layer,
            "summary": self.summary,
            "urls": list(self.urls),
            "size_gb_min": self.size_gb_min,
            "size_gb_max": self.size_gb_max,
            "recommended_trust": self.recommended_trust,
            "tags": list(self.tags),
            "requires_manual_access": self.requires_manual_access,
        }


DATASET_SPECS: List[DatasetSpec] = [
    DatasetSpec(
        dataset_id="wikipedia",
        name="Wikipedia Dump",
        layer="core",
        summary="Broad universal encyclopedia text for science, history, medicine, technology, and biographies.",
        urls=("https://dumps.wikimedia.org/",),
        size_gb_min=20.0,
        size_gb_max=40.0,
        recommended_trust=0.58,
        tags=("encyclopedia", "general", "science", "history"),
    ),
    DatasetSpec(
        dataset_id="wikidata",
        name="Wikidata",
        layer="core",
        summary="Structured entity relationships for reasoning and factual grounding.",
        urls=("https://dumps.wikimedia.org/wikidatawiki/entities/",),
        size_gb_min=8.0,
        size_gb_max=20.0,
        recommended_trust=0.74,
        tags=("knowledge_graph", "entities", "relationships"),
    ),
    DatasetSpec(
        dataset_id="c4_openweb",
        name="C4 / OpenWebText Subset",
        layer="core",
        summary="Filtered large-scale web text for broad language coverage.",
        urls=("https://commoncrawl.org/", "https://huggingface.co/datasets/c4"),
        size_gb_min=20.0,
        size_gb_max=150.0,
        recommended_trust=0.45,
        tags=("web", "general", "language"),
    ),
    DatasetSpec(
        dataset_id="arxiv",
        name="arXiv Metadata + Papers",
        layer="academic",
        summary="Research-heavy content for mathematics, physics, CS, and statistics.",
        urls=("https://www.kaggle.com/datasets/Cornell-University/arxiv", "https://info.arxiv.org/help/bulk_data_s3.html"),
        size_gb_min=5.0,
        size_gb_max=120.0,
        recommended_trust=0.88,
        tags=("research", "papers", "math", "physics", "ai"),
    ),
    DatasetSpec(
        dataset_id="pubmed",
        name="PubMed Baseline / Subset",
        layer="medical",
        summary="Biomedical abstracts and metadata for health and diagnostics grounding.",
        urls=("https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/",),
        size_gb_min=10.0,
        size_gb_max=80.0,
        recommended_trust=0.9,
        tags=("medical", "biology", "clinical", "research"),
    ),
    DatasetSpec(
        dataset_id="codesearchnet",
        name="CodeSearchNet",
        layer="programming",
        summary="Code plus natural-language documentation across major languages.",
        urls=("https://github.com/github/CodeSearchNet",),
        size_gb_min=2.0,
        size_gb_max=12.0,
        recommended_trust=0.76,
        tags=("code", "software", "engineering"),
    ),
    DatasetSpec(
        dataset_id="stackoverflow",
        name="Stack Overflow Archive",
        layer="programming",
        summary="High-volume practical debugging and implementation Q&A.",
        urls=("https://archive.org/details/stackexchange",),
        size_gb_min=20.0,
        size_gb_max=80.0,
        recommended_trust=0.67,
        tags=("code", "qna", "debugging"),
    ),
    DatasetSpec(
        dataset_id="proofwiki_mathse",
        name="ProofWiki + Math StackExchange",
        layer="science_math",
        summary="Proofs, derivations, and theorem reasoning examples.",
        urls=("https://proofwiki.org/", "https://archive.org/details/stackexchange"),
        size_gb_min=2.0,
        size_gb_max=20.0,
        recommended_trust=0.82,
        tags=("math", "proofs", "reasoning"),
    ),
    DatasetSpec(
        dataset_id="openstax",
        name="OpenStax Textbooks",
        layer="science_math",
        summary="Structured textbook knowledge across science, economics, and statistics.",
        urls=("https://openstax.org/subjects",),
        size_gb_min=2.0,
        size_gb_max=15.0,
        recommended_trust=0.92,
        tags=("textbook", "education", "physics", "biology", "chemistry"),
    ),
    DatasetSpec(
        dataset_id="stanford_philosophy",
        name="Stanford Encyclopedia of Philosophy",
        layer="humanities",
        summary="Deep conceptual and reasoning-heavy philosophy references.",
        urls=("https://plato.stanford.edu/",),
        size_gb_min=1.0,
        size_gb_max=5.0,
        recommended_trust=0.9,
        tags=("philosophy", "reasoning", "humanities"),
    ),
    DatasetSpec(
        dataset_id="cia_factbook",
        name="CIA World Factbook",
        layer="history_geography",
        summary="Country-level demographics, economics, and geopolitics snapshots.",
        urls=("https://www.cia.gov/the-world-factbook/",),
        size_gb_min=0.1,
        size_gb_max=2.0,
        recommended_trust=0.83,
        tags=("geography", "history", "country_data"),
    ),
    DatasetSpec(
        dataset_id="mimic_iv",
        name="MIMIC-IV",
        layer="medical",
        summary="Clinical ICU and chart events for advanced medical analytics and assistant workflows.",
        urls=("https://physionet.org/content/mimiciv/",),
        size_gb_min=50.0,
        size_gb_max=250.0,
        recommended_trust=0.93,
        tags=("medical", "clinical", "icu", "timeseries"),
        requires_manual_access=True,
    ),
    DatasetSpec(
        dataset_id="umls",
        name="UMLS",
        layer="medical",
        summary="Medical concept graph linking diseases, drugs, and symptoms.",
        urls=("https://www.nlm.nih.gov/research/umls/",),
        size_gb_min=2.0,
        size_gb_max=15.0,
        recommended_trust=0.91,
        tags=("medical", "knowledge_graph", "terminology"),
        requires_manual_access=True,
    ),
    DatasetSpec(
        dataset_id="kaggle_curated",
        name="Kaggle Curated Bundles",
        layer="data_science",
        summary="Domain datasets for tabular, NLP, and applied modeling tasks.",
        urls=("https://www.kaggle.com/datasets",),
        size_gb_min=5.0,
        size_gb_max=80.0,
        recommended_trust=0.62,
        tags=("data_science", "tabular", "nlp", "vision"),
    ),
    DatasetSpec(
        dataset_id="openassistant",
        name="OpenAssistant Conversations",
        layer="conversation",
        summary="Human preference and dialogue examples for assistant tone and style adaptation.",
        urls=("https://huggingface.co/datasets/OpenAssistant/oasst1",),
        size_gb_min=2.0,
        size_gb_max=20.0,
        recommended_trust=0.7,
        tags=("conversation", "assistant", "dialogue"),
    ),
    DatasetSpec(
        dataset_id="gutenberg",
        name="Project Gutenberg",
        layer="books",
        summary="Long-form literary and historical books for narrative and context depth.",
        urls=("https://www.gutenberg.org/",),
        size_gb_min=10.0,
        size_gb_max=80.0,
        recommended_trust=0.66,
        tags=("books", "literature", "history"),
    ),
]

DATASET_INDEX: Dict[str, DatasetSpec] = {spec.dataset_id: spec for spec in DATASET_SPECS}

BUNDLE_PRESETS: Dict[str, List[str]] = {
    "starter": [
        "wikipedia",
        "openstax",
        "stackoverflow",
        "pubmed",
        "codesearchnet",
        "cia_factbook",
    ],
    "core_plus": [
        "wikipedia",
        "wikidata",
        "c4_openweb",
        "openstax",
        "stackoverflow",
        "codesearchnet",
        "cia_factbook",
    ],
    "medical_plus": [
        "pubmed",
        "mimic_iv",
        "umls",
        "wikipedia",
        "openstax",
    ],
    "research_plus": [
        "arxiv",
        "pubmed",
        "wikipedia",
        "wikidata",
        "openstax",
    ],
    "full": [spec.dataset_id for spec in DATASET_SPECS],
}


def get_dataset(dataset_id: str) -> DatasetSpec | None:
    """Return a dataset spec by id."""
    return DATASET_INDEX.get((dataset_id or "").strip().lower())


def list_datasets(bundle: str | None = None) -> List[DatasetSpec]:
    """List dataset specs, optionally restricted to a preset bundle."""
    if not bundle:
        return list(DATASET_SPECS)

    ids = BUNDLE_PRESETS.get(bundle.lower(), [])
    return [DATASET_INDEX[item] for item in ids if item in DATASET_INDEX]


def _unique(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        key = (item or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def resolve_dataset_ids(bundle: str | None = None, include_ids: Iterable[str] | None = None) -> List[str]:
    """Resolve ordered unique dataset ids from a bundle and optional extra ids."""
    ids: List[str] = []
    if bundle:
        ids.extend(BUNDLE_PRESETS.get(bundle.lower(), []))
    if include_ids:
        ids.extend(include_ids)
    resolved = [item for item in _unique(ids) if item in DATASET_INDEX]
    return resolved


def estimate_size_gb(dataset_ids: Iterable[str]) -> Dict[str, float]:
    """Return total estimated disk footprint range in GB."""
    min_size = 0.0
    max_size = 0.0
    for dataset_id in dataset_ids:
        spec = DATASET_INDEX.get(dataset_id)
        if not spec:
            continue
        min_size += float(spec.size_gb_min)
        max_size += float(spec.size_gb_max)
    return {"min_gb": round(min_size, 2), "max_gb": round(max_size, 2)}


def build_bundle_plan(bundle: str = "starter", include_ids: Iterable[str] | None = None) -> Dict[str, Any]:
    """Build a normalized plan payload for dataset selection and preparation."""
    dataset_ids = resolve_dataset_ids(bundle=bundle, include_ids=include_ids)
    selected = [DATASET_INDEX[item] for item in dataset_ids]
    size = estimate_size_gb(dataset_ids)
    return {
        "bundle": bundle,
        "dataset_count": len(selected),
        "estimated_size_gb": size,
        "datasets": [item.to_dict() for item in selected],
    }
