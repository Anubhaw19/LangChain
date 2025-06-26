from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template = 'write a detailed report about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'summarise the following {text} in less than 50 words',
    input_variables = ['text']
)

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
parser = StrOutputParser()

def word_count(text):
    return len(text.split())

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split())>300, RunnableSequence(prompt2, model, parser)), #if report is greater than 500 words, summarise it
    RunnablePassthrough() #else show it 
    # (condition, runnable),
    # default
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({'topic':'origin of nuclear energy'})
print(result)