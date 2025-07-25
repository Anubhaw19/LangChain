from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

url = 'https://www.timeanddate.com/weather/india/bengaluru/hourly'
loader  = WebBaseLoader(url)

docs = loader.load()
# print(len(docs))
# print(docs[0].page_content)


model = ChatGoogleGenerativeAI(model = 'gemini-2.5-pro')

prompt = PromptTemplate(
    template = 'answer the following question in one line:\n {question} \n from the following texr - \n {text}',
    input_variables = ['question','text']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'question':'what is the possibility of rain','text':docs[0].page_content})

print(result)