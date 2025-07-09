from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = 'books',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.lazy_load()  #load vs lazy_load()
# print(docs[3].page_content)

for document in docs:
    print(document.metadata)