from langchain.text_splitter import RecursiveCharacterTextSplitter

text = ''' Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence. These tasks include understanding natural language, recognizing patterns, solving problems, learning from data, and making decisions. AI has evolved from being a theoretical concept to a transformative force reshaping industries, economies, and everyday life.
'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)