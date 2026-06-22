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
]

embeddings = model.encode(documents)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(embeddings)

query = "how to train AI"

query_emb = model.encode(
    [query]
).astype(np.float32)

distances, indices = index.search(
    query_emb,
    2
)

for i in indices[0]:
    print(documents[i])
