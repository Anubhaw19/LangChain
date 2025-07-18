from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import numpy as np

load_dotenv()

# Sample DataFrame
df_day = pd.DataFrame({
    'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'impressions': [10000, 12000, 11000, 15000, 17000, 20000, 18000],
    'views': [1500, 1600, 1400, 2000, 2500, 3000, 2700]
})
df_day['VTR'] = (df_day['views'] / df_day['impressions'] * 100).round(2)
df_day_str = df_day.to_string(index=False)
# print("Campaign Performance by Day of Week:\n", df_day_str)

df_time = pd.DataFrame({
    'time_of_day': ['Morning', 'Afternoon', 'Evening', 'Night'],
    'impressions': [8000, 12000, 16000, 6000],
    'views': [1200, 1800, 2200, 700]
})
df_time['VTR'] = (df_time['views'] / df_time['impressions'] * 100).round(2)
df_time_str = df_time.to_string(index=False)
# print("\nCampaign Performance by Time of Day:\n", df_time_str)

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-pro')

prompt = PromptTemplate(
    template = 'write the temporal (day of week) commentary for my campign, based on this information: {data} \n do mention any possible reason behind any trend, keep the commentary short paragraph 2-3 points',
    input_variables = ['data']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'data':df_day_str})

print(result)

#result:
# Performance peaks significantly over the weekend, with Friday, Saturday, and Sunday delivering the highest impressions, views, and a strong View-Through Rate (VTR) of around 15%. This suggests the target audience has more leisure time and is more receptive to video content during their days off. Conversely, the campaign sees a dip in engagement efficiency mid-week, with Wednesday marking the lowest VTR (12.73%), likely as the audience is busier with work and daily routines, leading to lower attention spans.