# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.schema import Document

load_dotenv()

#create LangChain documents 

documents = [
    Document(
        page_content="Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data and make decisions.",
        metadata={"source": "AI_Overview.pdf", "topic": "AI", "page": 1, "author": "John Doe"}
    ),
    Document(
        page_content="The global e-commerce market has seen a surge in mobile transactions, contributing to over 60% of total online purchases.",
        metadata={"source": "Ecommerce_Trends_2024.docx", "topic": "E-commerce", "page": 2, "author": "Jane Smith"}
    ),
    Document(
        page_content="Climate change is primarily driven by increased levels of greenhouse gases such as CO2, CH4, and N2O.",
        metadata={"source": "Climate_Report.txt", "topic": "Environment", "page": 3, "author": "UNEP"}
    ),
    Document(
        page_content="LangChain simplifies building applications powered by language models by offering composable abstractions.",
        metadata={"source": "LangChain_Intro.md", "topic": "LangChain", "page": 1, "author": "LangChain Team"}
    ),
    Document(
        page_content="SQL joins allow you to combine rows from two or more tables based on a related column between them.",
        metadata={"source": "SQL_Guide.epub", "topic": "Databases", "page": 5, "author": "Data Academy"}
    )
]

# vector_store = Chroma(
#     embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#     persist_directory = 'chroma_db',
#     collection_name = 'sample'
# )

#in-memory vectorstore 
vectorstore = Chroma.from_documents(
    documents = documents,
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    collection_name = "my_collection"
)

#convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k":2})

query = "what is the langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n---Result{i+1}---")
    print(doc.page_content)