import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_chunks"

CHUNKS_PATH = Path("data/processed/chunks/chunks.jsonl")

# Embedding model: 384 dims
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def stable_int_id(s: str) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_payload(rec: Dict[str, Any]) -> Dict[str, Any]:
    text = (rec.get("text") or "").strip()
    preview = text[:700]  # dev-friendly preview

    return {
        "chunk_id": rec["chunk_id"],
        "chunk_type": rec.get("chunk_type", ""),
        "doc_id": rec["doc_id"],
        "title": rec.get("title", ""),
        "vendor": rec.get("vendor", ""),
        "product_line": rec.get("product_line", ""),
        "doc_type": rec.get("doc_type", ""),
        "version": rec.get("version", ""),
        "publish_date": rec.get("publish_date", ""),
        "trust_level": rec.get("trust_level", ""),
        "section_title": rec.get("section_title", ""),
        "page_start": rec.get("page_start", None),
        "page_end": rec.get("page_end", None),
        "contains_table": bool(rec.get("contains_table", False)),
        "char_len": len(text),

        # ✅ ADD THIS
        "text": text,

        "text_preview": preview,
        "is_toc_like": text.lower().startswith("contents") or "table of contents" in text.lower(),
    }


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PATH}")

    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    buffer_recs: List[Dict[str, Any]] = []
    buffer_texts: List[str] = []
    points: List[PointStruct] = []

    total = 0

    for rec in iter_jsonl(CHUNKS_PATH):
        text = (rec.get("text") or "").strip()
        if not text:
            continue

        buffer_recs.append(rec)
        buffer_texts.append(text)

        if len(buffer_texts) >= BATCH_SIZE:
            embeddings = model.encode(
                buffer_texts, normalize_embeddings=True, show_progress_bar=False
            )
            for r, emb in zip(buffer_recs, embeddings):
                pid = stable_int_id(r["chunk_id"])
                payload = build_payload(r)
                points.append(PointStruct(id=pid, vector=emb.tolist(), payload=payload))

            client.upsert(collection_name=COLLECTION, points=points)
            total += len(points)
            print(f"✅ Upserted batch: {len(points)}  (total={total})")

            buffer_recs, buffer_texts, points = [], [], []

    if buffer_texts:
        embeddings = model.encode(buffer_texts, normalize_embeddings=True, show_progress_bar=False)
        for r, emb in zip(buffer_recs, embeddings):
            pid = stable_int_id(r["chunk_id"])
            payload = build_payload(r)
            points.append(PointStruct(id=pid, vector=emb.tolist(), payload=payload))

        client.upsert(collection_name=COLLECTION, points=points)
        total += len(points)
        print(f"✅ Upserted final batch: {len(points)}  (total={total})")

    print(f"\n🎉 Done. Total points upserted: {total}")


if __name__ == "__main__":
    main()