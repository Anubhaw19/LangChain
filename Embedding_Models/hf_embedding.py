from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
text = "AI is the future of technology"
documnet = [
    "The cat sat on the mat.",
    "She loves to read books.",
    "Birds sing in the morning.",
    "He runs very fast."
]
vector = embedding.embed_query(text)
print(str(vector))