# embedding_test.py

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "train neural network",
    "fit machine learning model",
    "buy a pizza"
]

embeddings = model.encode(sentences)

print(embeddings.shape)
