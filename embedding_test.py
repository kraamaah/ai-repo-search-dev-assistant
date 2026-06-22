# embedding_test.py

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "where is the diarization loss computed",
    "which function calculates training loss",
    "how to order pizza online"
]

embeddings = model.encode(sentences)

print(embeddings.shape)
