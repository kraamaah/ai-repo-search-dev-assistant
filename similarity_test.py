from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "where is the diarization loss computed",
    "which function calculates training loss",
    "how to order pizza online"
]

embeddings = model.encode(texts)

sim1 = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)

sim2 = cosine_similarity(
    [embeddings[0]],
    [embeddings[2]]
)

print("Loss vs Training Loss:")
print(sim1)

print()

print("Loss vs Pizza:")
print(sim2)
