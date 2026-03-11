from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter, FieldCondition, MatchValue

QDRANT_URL = "http://localhost:6333"
COLLECTION = "rag_chunks"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def main():
    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    while True:
        q = input("\nQuery (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            break

        qv = model.encode([q], normalize_embeddings=True)[0].tolist()

        flt = Filter(
            must=[
                FieldCondition(key="trust_level", match=MatchValue(value="authoritative")),
                FieldCondition(key="is_toc_like", match=MatchValue(value=False)),
            ]
        )

        # ✅ New API (Universal Query)
        res = client.query_points(
            collection_name=COLLECTION,
            query=qv,
            limit=8,
            with_payload=True,
            query_filter=flt,
        )

        hits = res.points if hasattr(res, "points") else res  # safety

        print("\nTop hits:")
        for i, h in enumerate(hits, 1):
            p = h.payload or {}
            print(f"\n[{i}] score={h.score:.4f}  {p.get('vendor')}  {p.get('doc_id')}")
            print(f"  pages={p.get('page_start')}-{p.get('page_end')}  section={p.get('section_title')}")
            print(f"  chunk_id={p.get('chunk_id')}")
            print(f"  preview={str(p.get('text_preview',''))[:260].replace('\\n',' ')}...")

if __name__ == "__main__":
    main()