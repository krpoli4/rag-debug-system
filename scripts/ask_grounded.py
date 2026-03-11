import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_chunks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Conservative defaults (tune later)
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


def extract_entities(query: str) -> List[str]:
    patterns = [
        r"\b0x[0-9a-fA-F]+\b",   # 0x0000007B
        r"\bXid\s*\d+\b",        # Xid 79
        r"\bXID\s*\d+\b",        # XID 79
        r"\b\d{4}-\d{4}\b",      # 2000-8001
    ]
    found: List[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, query))

    # normalize: "Xid 79" -> "xid79", "0x0000007B" -> "0x0000007b"
    norm = [re.sub(r"\s+", "", x).lower() for x in found]
    return sorted(set(norm))


def normalize_for_match(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower())


def retrieve(client: QdrantClient, model: SentenceTransformer, query: str, limit: int = 8) -> List[Evidence]:
    qv = model.encode([query], normalize_embeddings=True)[0].tolist()

    flt = Filter(
        must=[
            FieldCondition(key="trust_level", match=MatchValue(value="authoritative")),
            FieldCondition(key="is_toc_like", match=MatchValue(value=False)),
        ]
    )

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
                text=str(p.get("text", "")),                 # ✅ full text (requires re-ingest)
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

    # ✅ Exact-code gating
    entities = extract_entities(query)
    if entities:
        joined = normalize_for_match(" ".join([e.text for e in evidence[:6]]))
        missing = [ent for ent in entities if ent not in joined]
        if missing:
            return True, f"Retrieved related content, but docs do NOT contain the exact code(s): {missing}"

    return False, None


def main():
    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    while True:
        q = input("\nAsk (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            break

        evidence = retrieve(client, model, q, limit=8)

        # debug print so you can *see* entity extraction working
        ents = extract_entities(q)
        if ents:
            print(f"\n[debug] extracted_entities={ents}")

        refuse, reason = gate(q, evidence)

        if refuse:
            print("\n❌ REFUSE (not enough evidence in provided docs)")
            print(f"Reason: {reason}")
            print("Tip: add the vendor doc that defines this exact code, or provide the official reference link.")
            continue

        print("\n✅ EVIDENCE PACK (for grounded answering)")
        for i, e in enumerate(evidence[:6], 1):
            print(f"\n[{i}] score={e.score:.4f}  {e.vendor}/{e.doc_id} v{e.version}")
            print(f"    pages={e.page_start}-{e.page_end}  section={e.section_title}")
            print(f"    chunk_id={e.chunk_id}")
            print(f"    preview={e.preview[:260].replace('\\n',' ')}...")

        print("\n(Next step: LLM answer will be generated ONLY from this evidence pack + citations.)")


if __name__ == "__main__":
    main()