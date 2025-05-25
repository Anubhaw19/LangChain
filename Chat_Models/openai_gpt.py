from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4.1-nano')
result = model.invoke('write a 3 line poem on AI')
print(result.content)