# faiss_test.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "train neural network",
    "fit machine learning model",
    "buy pizza",
    "deep learning tutorial",
    "transformer attention mechanism",
    "speech diarization loss function"
]

embeddings = model.encode(
    documents,
    convert_to_numpy=True
)

embeddings = embeddings.astype(
    np.float32
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)

query = "how is training loss calculated"

query_embedding = model.encode(
    [query],
    convert_to_numpy=True
)

query_embedding = query_embedding.astype(
    np.float32
)

distances, indices = index.search(
    query_embedding,
    k=3
)

print("\nQuery:")
print(query)

print("\nTop Results:\n")

for rank, idx in enumerate(indices[0]):
    print(
        f"{rank+1}. {documents[idx]}"
    )
