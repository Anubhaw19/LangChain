import pandas as pd
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

df = pd.read_csv('model_data.csv')

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

schema = [
    ResponseSchema(name = 'Brand', description = "Brand of the mobile model"),
    ResponseSchema(name = 'Model', description = "Model name of the mobile model"),
    ResponseSchema(name = 'Price', description = "Price of the mobile model in INR"),
    ResponseSchema(name = 'OS', description = "Operating system of the mobile model"),
    ResponseSchema(name = 'Launch Date', description = "Launch date of the mobile model in India"),
    # ResponseSchema(name = 'Source', description = "Source of data, if referring to any website then url of it"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give the specification of {brand_name} brand, {model_name} model. \n {format_instruction}',
    input_variables = ['brand_name','model_name'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
# result = chain.invoke({'brand_name': 'apple','model_name': 'iPhone 16 plus'})
# print(result)

LLM_Responses =[]

for index, row in df.iterrows():
    brand_name = {row['Device Make']}
    model_name = {row['Device Model']}
    print(f"Row {index}: brand_name={brand_name}, model_name={model_name}")
    result = chain.invoke({'brand_name': brand_name,'model_name': model_name})
    LLM_Responses.append(result)
    if index == 11:
        break
    time.sleep(4)

output_df = pd.DataFrame(LLM_Responses)
print(output_df.head())
output_df.to_csv('output.csv', index=False)