from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template = 'write a short joke about {topic}',
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = 'explain this joke {text}',
    input_variables = ['text']
)

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)
print(chain.invoke({'topic': 'cricket'}))