import pandas as pd
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-pro')

class Weather(BaseModel):
    city: str = Field(description = "name of the city")
    weather: str = Field(description = 'current weather condition as per google weather app')
    temperature : int = Field(description = "current temperature of the city in celcius")
    temperature_min: int = Field(description = "minimum temperature of the city in celcius")
    temperature_max : int = Field(description = "maximun temperature of the city in celcius")
    rain_condition : Literal['no rain','light rain', 'moderate rain','heavy rain'] = Field(description = 'current rain condition of the city')
    observation_time : str = Field(description = "current time for the weather conditon")
    forecast : str = Field(description = "next 12 hours, hourly forecast for rain possibility (no rain,light rain, moderate rain,heavy rain)")
    
parser = PydanticOutputParser(pydantic_object = Weather)

template = PromptTemplate(
    template = 'Give the current weather condition of {city} city \n {format_instruction}',
    input_variables = ['city'],
    partial_variables = {'format_instruction' : parser.get_format_instructions()}
)

chain = template | model | parser

result  = chain.invoke({'city':'Ranchi, Jharkhand, India'})
print(result)

