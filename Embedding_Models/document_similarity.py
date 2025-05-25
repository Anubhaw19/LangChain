from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documnet = [
    "The cat sat on the mat.",
    "She loves to read books.",
    "Birds sing in the morning.",
    "He runs very fast."
]

query = "tell me about morning"
document_embedding = embedding.embed_documents(documnet)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], document_embedding)[0]
# print(scores)
index, score = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]
print(query)
print(documnet[index])
print('score: ', score)
