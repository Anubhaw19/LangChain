from langchain_community.document_loaders import CSVLoader
loader = CSVLoader(file_path = 'sample_employee_data.csv')

docs = loader.load()

print(len(docs))
print(docs[0])