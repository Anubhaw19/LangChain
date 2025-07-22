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

campaign_context = '''This is a campaign for McCain Frozen Foods that ran during the monsoon season on YouTube CTV, targeting a female audience aged 25–44.'''

prompt = PromptTemplate(
    template="""
You are a marketing analyst. Based on the following temporal campaign performance data (by time of day and day of week), write a concise merged commentary with 2–3 key insights:
{data}
Incorporate the campaign context below to explain any patterns or performance variations:
{context}
Keep the tone analytical and suitable for inclusion in a post-campaign insights report.
""",
    input_variables=['data', 'context']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'data':df_day_str+'\n'+df_time_str, 'context': campaign_context})

print(result)

#result:
### Campaign Temporal Analysis
# **Temporal Performance Analysis**

# Campaign performance showed distinct patterns by day and time, driven by the target audience's viewing habits on CTV during the monsoon season.

# 1.  **Weekend Viewing Delivers Peak Volume and Engagement:** Performance surged from Friday through Sunday, which saw the highest volume of impressions and views, coupled with a top-tier VTR of 15%. This indicates that our target audience (females 25-44) was most active and receptive during the weekend, aligning with increased leisure time for CTV viewing and planning convenient, "stay-at-home" meals with McCain products.

# 2.  **Daytime Delivers Efficiency, Evening Delivers Scale:** A clear trade-off between audience size and engagement emerged across the day. While evenings drove the highest reach (16,000 impressions), the VTR was highest in the morning and afternoon (15.00%). This suggests that the smaller, daytime audience was more attentive and less distracted, presenting an opportunity for highly efficient targeting. In contrast, the lower evening VTR (13.75%) likely reflects a more casual, co-viewing environment typical of primetime.