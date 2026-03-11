import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_chunks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

MIN_TOP_SCORE = 0.45
MIN_STRONG_HITS = 2
STRONG_HIT_SCORE = 0.40


@dataclass
class Evidence:
    score: float
    doc_id: str
    vendor: str
    version: str
    page_start: int
    page_end: int
    section_title: str
    chunk_id: str
    text: str
    preview: str


_qdrant_client: Optional[QdrantClient] = None
_embed_model: Optional[SentenceTransformer] = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def extract_entities(query: str) -> List[str]:
    patterns = [
        r"\b0x[0-9a-fA-F]+\b",
        r"\bXid\s*\d+\b",
        r"\bXID\s*\d+\b",
        r"\b\d{4}-\d{4}\b",
    ]
    found: List[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, query))
    return sorted(set(re.sub(r"\s+", "", x).lower() for x in found))


def normalize_for_match(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower())


def infer_vendor_filter(query: str) -> Optional[str]:
    q = query.lower()
    nvidia_terms = ["xid", "dcgm", "nvlink", "cuda", "gpu", "nvidia", "pcie"]
    dell_terms = ["idrac", "poweredge", "perc", "psa", "epsa", "lifecycle controller", "dell"]

    if any(term in q for term in nvidia_terms):
        return "NVIDIA"
    if any(term in q for term in dell_terms):
        return "Dell"
    return None


def build_filter(query: str) -> Filter:
    must_conditions = [
        FieldCondition(key="trust_level", match=MatchValue(value="authoritative")),
        FieldCondition(key="is_toc_like", match=MatchValue(value=False)),
    ]

    vendor = infer_vendor_filter(query)
    if vendor:
        must_conditions.append(FieldCondition(key="vendor", match=MatchValue(value=vendor)))

    return Filter(must=must_conditions)


def retrieve(query: str, limit: int = 8) -> List[Evidence]:
    client = get_qdrant_client()
    model = get_embed_model()

    qv = model.encode([query], normalize_embeddings=True)[0].tolist()
    flt = build_filter(query)

    res = client.query_points(
        collection_name=COLLECTION,
        query=qv,
        limit=limit,
        with_payload=True,
        query_filter=flt,
    )

    out: List[Evidence] = []
    for h in res.points:
        p: Dict[str, Any] = h.payload or {}
        out.append(
            Evidence(
                score=float(h.score),
                doc_id=str(p.get("doc_id", "")),
                vendor=str(p.get("vendor", "")),
                version=str(p.get("version", "")),
                page_start=int(p.get("page_start") or -1),
                page_end=int(p.get("page_end") or -1),
                section_title=str(p.get("section_title", "")),
                chunk_id=str(p.get("chunk_id", "")),
                text=str(p.get("text", "")),
                preview=str(p.get("text_preview", "")),
            )
        )
    return out


def gate(query: str, evidence: List[Evidence]) -> Tuple[bool, Optional[str]]:
    if not evidence:
        return True, "No relevant content retrieved from indexed documents."

    top = evidence[0]
    if top.score < MIN_TOP_SCORE:
        return True, f"Top score {top.score:.3f} < MIN_TOP_SCORE {MIN_TOP_SCORE:.2f}."

    strong = [e for e in evidence if e.score >= STRONG_HIT_SCORE]
    if len(strong) < MIN_STRONG_HITS:
        return True, f"Only {len(strong)} chunks >= {STRONG_HIT_SCORE:.2f}; need {MIN_STRONG_HITS}."

    entities = extract_entities(query)
    if entities:
        joined = normalize_for_match(" ".join([e.text for e in evidence[:6]]))
        missing = [ent for ent in entities if ent not in joined]
        if missing:
            return True, f"Retrieved related content, but docs do NOT contain the exact code(s): {missing}"

    return False, None