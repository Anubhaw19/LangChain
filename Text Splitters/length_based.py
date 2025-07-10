from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('../Document_Loader/Introduction_to_AI.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap =0,
    separator = ''
)

# text = '''Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence. These tasks include understanding natural language, recognizing patterns, solving problems, learning from data, and making decisions. AI has evolved from being a theoretical concept to a transformative force reshaping industries, economies, and everyday life.

# At the core of AI is the idea of building systems that can learn and adapt. Traditional software follows explicit instructions, but AI systems can analyze large datasets and improve their performance over time without being explicitly programmed for every scenario. This learning capability is powered by subfields like machine learning, deep learning, and neural networks.'''

# result = splitter.split_text(text)

result = splitter.split_documents(docs)
print(result[5])