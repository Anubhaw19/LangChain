from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

schema = [
    ResponseSchema(name = 'Brand', description = "Brand of the mobile model"),
    ResponseSchema(name = 'Model', description = "Model name of the mobile model"),
    ResponseSchema(name = 'Price', description = "Price of the mobile model in INR"),
    ResponseSchema(name = 'OS', description = "Operating system of the mobile model"),
    ResponseSchema(name = 'Launch Date', description = "Launch date of the mobile model in India"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give the specification of {model_name} mobile model and just give the data \n {format_instruction}',
    input_variables = ['model_name'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

# Prompt = template.invoke({'model_name':'iPhone 16 pro'})
# result = model.invoke(Prompt)
# final_result = parser.parse(result.content)
# print(final_result)

#invoke model using chain
chain = template | model | parser
result = chain.invoke({'model_name': 'iPhone 16 plus'})
print(result)