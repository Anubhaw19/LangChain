from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')
result = model.invoke('write a 3 line poem on AI')
print(result.content)