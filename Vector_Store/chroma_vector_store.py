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

vector_store = Chroma(
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    persist_directory = 'chroma_db',
    collection_name = 'sample'
)

#ADD DOCUMENT
# print('document add: ',vector_store.add_documents(documents))

#VIEW DOCUMENT
# print('document view: ',vector_store.get(include = ['embeddings', 'documents', 'metadatas']))

#DOCUMENT SIMILARITY SEARCH

# result = vector_store.similarity_search(
#     query = 'discuss about climate change',
#     k=3)

# for index,docs in enumerate(result):
#     print(index,": ",docs.page_content)

result = vector_store.similarity_search_with_score(
    query = 'discuss about climate change',
    k=2)

for index,docs in enumerate(result):
    print(index,": ",docs[0].page_content)
    print('similarity_score: ',docs[1])

# META-DATA FILTERING
vector_store.similarity_search_with_score(
    query = "",
    filter={"topic":"Environment"}
)

#UPDATE DOCUMENTS
# updated_document = Document(
#         page_content="the content about the topic has been changed",
#         metadata={"source": "Climate_Report.txt", "topic": "Environment", "page": 3, "author": "UNEP"}
#     )
# vector_store.update_document(document_id= 'f6dbac66-b944-4066-b773-f40caca370bd', document = updated_document)

#DELETE DOCUMENT
# vector_store.delete(ids = ['f6dbac66-b944-4066-b773-f40caca370bd'])