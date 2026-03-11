from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION = "rag_chunks"
QDRANT_URL = "http://localhost:6333"

# Dimension must match the embedding model you choose.
# We'll use "BAAI/bge-small-en-v1.5" which outputs 384-dim embeddings.
VECTOR_SIZE = 384

def main():
    client = QdrantClient(url=QDRANT_URL)

    # Recreate to avoid schema drift during development
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    print(f"✅ Created collection '{COLLECTION}' with vector size={VECTOR_SIZE}")

if __name__ == "__main__":
    main()