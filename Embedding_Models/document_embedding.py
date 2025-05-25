from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions = 32)
documnet = [
    "The cat sat on the mat.",
    "She loves to read books.",
    "Birds sing in the morning.",
    "He runs very fast."
]
result = embedding.embed_documents(documnet)
print(str(result))