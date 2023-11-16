from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

"""
model = SentenceTransformer("output/all-mpnet-base-v2-v1-scifact")
#model = SentenceTransformer("output/sentence-transformers/all-distilroberta-v1-v1-scifact")
#model = SentenceTransformer("output/distilbert-base-uncased-v1-scifact")

sentences = ["This is a happy person", "This is a very happy person", "Each sentence is converted"]

embeddings = model.encode(sentences)
print(embeddings)

query = embeddings[0]
for i in range(1, len(embeddings)):
    sentence = embeddings[i]
    cos_sim = dot(query, sentence) / (norm(query) * norm(sentence))
    print(cos_sim)"""
