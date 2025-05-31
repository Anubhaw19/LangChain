from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} exprt'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket', 'topic':'Dusra'})
print(prompt)

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')
result = model.invoke(prompt)
print(result.content)