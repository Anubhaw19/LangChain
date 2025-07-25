from langchain_community.vectorstores import FAISS #vector store from facebook (Facebook AI Similarity Search)
# from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()
documents = [
    Document(page_content="Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data and make decisions."),
    Document(page_content="The global e-commerce market has seen a surge in mobile transactions, contributing to over 60% of total online purchases."),
    Document(page_content="Climate change is primarily driven by increased levels of greenhouse gases such as CO2, CH4, and N2O."),
    Document(page_content="LangChain simplifies building applications powered by language models by offering composable abstractions."),
    Document(page_content="LangChain simplifies building applications powered by large language models by offering composable abstractions."),
    Document(page_content="SQL joins allow you to combine rows from two or more tables based on a related column between them.")
]

vectorstore = FAISS.from_documents(
    documents = documents,
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
)

retriever = vectorstore.as_retriever(
    search_type = "mmr", # this enables MMR (Maximal Marginal Relevance)
    search_kwargs = {"k":3, "lambda_mult": 0.7} # k = top results, lambda_mult = relevance-diversity balance
)

query = "what is langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n---Result{i+1}---")
    print(doc.page_content)