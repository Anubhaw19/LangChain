from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Introduction_to_AI.pdf')
docs = loader.load()
print(len(docs))
print(docs[0])