import pandas as pd
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.runnables import RunnableConfig

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
    template = 'Give the specification of the following {brand_name} brand, {model_name} models. \n {format_instruction}',
    input_variables = ['brand_name','model_name'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)
chain = template | model | parser

LLM_Responses =[]
batch_df = df.iloc[:3]

batch_size = 3 # define batch size
for row_number in range(0,df.shape[0],batch_size):
    if row_number>15: # TODO: remove before running to fetch all model data
        break
    batch_df = df.iloc[row_number : row_number + batch_size]

    batch_inputs = [
        {'brand_name': row['Device Make'], 'model_name': row['Device Model']}
        for _, row in batch_df.iterrows()
    ]
    print('processing row: ',row_number, ':',row_number+batch_size)
    results = chain.batch(batch_inputs, config=RunnableConfig(max_concurrency=5))

    for data in results:
        LLM_Responses.append(data)
    time.sleep(30) #waiting for 30 sec to avoid rate limit exceed (RPM) (for more details check here :https://ai.google.dev/gemini-api/docs/rate-limits#free-tier)
    
output_df = pd.DataFrame(LLM_Responses)
output_df.to_csv('output_batch.csv', index=False)

print(output_df.head())

