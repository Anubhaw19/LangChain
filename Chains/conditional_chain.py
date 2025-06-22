from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda # runnable branch for conditional chain, runnalble parallel for parallel chain
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description = 'Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object = Feedback )

prompt1 = PromptTemplate(
    template = 'calssify the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables = {'format_instruction' : parser2.get_format_instructions()}
)
prompt2 = PromptTemplate(
    template = 'write an appropriate one liner response for this positive product feedback \n {feedback}' ,
    input_variables = ['feedback']
)
prompt3 = PromptTemplate(
    template = 'write an empathetic and acknowledging one liner response for this negative product feedback\n {feedback}' ,
    input_variables = ['feedback']
)

calssifier_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x : x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x : "could not find sentiment")
)

chain = calssifier_chain | branch_chain
result = chain.invoke({'feedback':'this is a terrible phone'})

print(result)

chain.get_graph().print_ascii()