from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt = PromptTemplate(
    template = 'genegrate a short tweet about {topic}',
    input_variables = ['topic']
)
prompt2 = PromptTemplate(
    template = 'generate a short linkedin post about {topic}',
    input_variables = ['topic']
)

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()

chain = RunnableParallel({
    'tweet': RunnableSequence(prompt, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = chain.invoke({'topic': 'AI'})

print(result['tweet'])
print(result['linkedin'])