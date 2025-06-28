from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

class Mobile(BaseModel):
    brand: str = Field(description = "name of the mobile phone brand")
    model: str = Field(description = 'name of the mobile model')
    price : int = Field(description = "launch price of the mobile model in INR")
    os: str = Field(description = "Oerating system of the mobile model")
    launch_date : str = Field(description = "Lauch date of the model in India")

parser = PydanticOutputParser(pydantic_object = Mobile)

template = PromptTemplate(
    template = 'Give the specification of {model_name} mobile model, if it is not a correct model name,do not assuem it \n {format_instruction}',
    input_variables = ['model_name'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

# prompt = template.invoke({'model_name':'Googgle Pixel 6'})

# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

#invoke model using chain
chain = template | model | parser
result = chain.invoke({'model_name':'Samsung Galaxy F62'})
print(result)