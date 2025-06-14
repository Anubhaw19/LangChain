from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
# from langchain.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me brand, model, price(INR), OS, Launch date of Samsung Galaxy S24 Ultra \n {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

# print(final_result)
# print(type(final_result))

#using chain to invoke the model
chain = template | model | parser 
result = chain.invoke({})
print(result)