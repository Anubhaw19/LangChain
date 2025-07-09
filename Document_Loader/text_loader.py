from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

loader = TextLoader('sample.txt')
docs = loader.load()

print(type(docs)) # <class 'list'>
print(len(docs))

print(type(docs[0])) #<class 'langchain_core.documents.base.Document'>

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-pro')

prompt = PromptTemplate(
    template = 'summarise this text in 2 lines text: {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'text':docs[0].page_content})

print(result)