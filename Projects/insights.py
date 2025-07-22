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

campaign_context = ''' this is mccain frozen food brand, the campaign ran during mansoon on YouTube CTV, targetting female audience of age group between 25-44'''
prompt = PromptTemplate(
    template = 'write the short merged temporal commentary for the campaign, analysing time of day and day of week data: {data} \n do mention any possible reason behind any trend, keep the commentary short paragraph 2-3 points. Addition to this here is the campaign context: \n {context}',
    input_variables = ['data','context']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'data':df_day_str+'\n'+df_time_str, 'context': campaign_context})

print(result)

#result:
### Campaign Temporal Analysis

# **Day of the Week Performance:**

# The campaign saw a significant surge in viewership towards the weekend, with impressions and views peaking on Saturday. This trend is logical, as the target audience of females (25-44) likely has more leisure time for CTV viewing on weekends, especially during the monsoon season which encourages staying indoors. Interestingly, VTR was highest on Saturday, Sunday, and Monday (15.00%), suggesting that viewers are most receptive at the start of the week when planning meals and during their relaxed weekend family time, making them less likely to skip a relevant food ad.

# **Time of Day Performance:**

# Viewership volume peaked in the Evening, aligning with standard prime-time TV habits when the target audience is likely relaxing after their day. However, the highest audience engagement (VTR) occurred during the Morning and Afternoon (15.00%). This indicates that while reach is highest in the evening, the audience is more actively engaged and receptive to a McCain frozen food message earlier in the day, possibly while planning for lunch or dinner. The lower VTR in the evening and night, despite high impression volume, suggests viewers may be more focused on their chosen content and quicker to skip ads.


#short commentary
# The campaign's performance peaked over the weekend (Saturday/Sunday) and on Monday, which delivered the highest viewership and strongest VTR. This suggests the target audience of women is most receptive when planning meals for the week ahead or seeking convenient, cozy food options for family leisure time, a behavior amplified during the monsoon season. While evening viewing on CTV captured the largest audience volume, the ad's efficiency was significantly higher in the morning and afternoon, indicating a key window of opportunity when viewers are more actively engaged with food-related content and less likely to skip.