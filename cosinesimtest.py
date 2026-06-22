from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "train neural network",
    "fit machine learning model",
    "buy pizza"
]

emb = model.encode(texts)

print(
    cosine_similarity(
        [emb[0]],
        [emb[1]]
    )
)

print(
    cosine_similarity(
        [emb[0]],
        [emb[2]]
    )
)

