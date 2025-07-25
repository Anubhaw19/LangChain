from langchain_community.vectorstores import FAISS #vector store from facebook (Facebook AI Similarity Search)
# from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

documents = [
    Document(page_content="""The Grand Canyon is one of the most visited natural wonders in the world.
Photosynthesis is the process by which green plants convert sunlight into energy.
Millions of tourists travel to see it every year. The rocks date back millions of years.""", metadata={"source": "Doc1"}),

    Document(page_content="""In medieval Europe, castles were built primarily for defense.
The chlorophyll in plant cells captures sunlight during photosynthesis.
Knights wore armor made of metal. Siege weapons were often used to breach castle walls.""", metadata={"source": "Doc2"}),

    Document(page_content="""Basketball was invented by Dr. James Naismith in the late 19th century.
It was originally played with a soccer ball and peach baskets. NBA is now a global league.""", metadata={"source": "Doc3"}),

    Document(page_content="""The history of cinema began in the late 1800s. Silent films were the earliest form.
Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
Modern filmmaking involves complex CGI and sound design.""", metadata={"source": "Doc4"}),
]

vectorstore = FAISS.from_documents(
    documents = documents,
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
)

base_retriever = vectorstore.as_retriever(search_kwargs = {"k":5})
llm = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash-lite')
compressor = LLMChainExtractor.from_llm(llm)

#Create the contexual compression retriever 
comparession_retriever = ContextualCompressionRetriever(
    base_retriever = base_retriever,
    base_compressor = compressor
)

query = "what is photosynthesis?"
compressed_results = comparession_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"\n---Result{i+1}---")
    print(doc.page_content)