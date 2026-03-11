import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_chunks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

MIN_TOP_SCORE = 0.45
MIN_STRONG_HITS = 2
STRONG_HIT_SCORE = 0.40

client_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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


def retrieve(client_qdrant: QdrantClient, model: SentenceTransformer, query: str, limit: int = 8) -> List[Evidence]:
    qv = model.encode([query], normalize_embeddings=True)[0].tolist()

    flt = Filter(
        must=[
            FieldCondition(key="trust_level", match=MatchValue(value="authoritative")),
            FieldCondition(key="is_toc_like", match=MatchValue(value=False)),
        ]
    )

    res = client_qdrant.query_points(
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


def build_evidence_block(evidence: List[Evidence]) -> str:
    blocks = []
    for i, e in enumerate(evidence[:6], 1):
        blocks.append(
            f"""[Evidence {i}]
doc_id: {e.doc_id}
vendor: {e.vendor}
version: {e.version}
pages: {e.page_start}-{e.page_end}
section: {e.section_title}
chunk_id: {e.chunk_id}
text:
{e.text}
"""
        )
    return "\n\n".join(blocks)


def answer_with_openai(question: str, evidence: List[Evidence]) -> str:
    evidence_text = build_evidence_block(evidence)

    system_prompt = """
You are a hardware debugging assistant.
Answer ONLY from the provided evidence.
Do not use outside knowledge.
If the evidence is insufficient, say:
"Insufficient evidence in the provided documents."
Always cite sources inline using:
[doc_id pages X-Y, chunk_id]
Keep the answer concise, technical, and audit-friendly.
"""

    response = client_openai.responses.create(
        model="gpt-5.4",
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Question:
{question}

Evidence:
{evidence_text}

Instructions:
- Use only the evidence above.
- Do not infer beyond the evidence.
- Include citations inline.
"""
            }
        ],
    )

    return response.output_text


def main():
    client_qdrant = QdrantClient(url=QDRANT_URL)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    while True:
        q = input("\nAsk (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            break

        evidence = retrieve(client_qdrant, embed_model, q, limit=8)
        refuse, reason = gate(q, evidence)

        if refuse:
            print("\n❌ REFUSE")
            print(f"Reason: {reason}")
            print("Tip: provide exact XID / error code / log line / server model.")
            continue

        print("\n✅ EVIDENCE FOUND — generating grounded answer...\n")
        answer = answer_with_openai(q, evidence)
        print(answer)


if __name__ == "__main__":
    main()