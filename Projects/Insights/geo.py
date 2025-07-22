from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import numpy as np

load_dotenv()

# Sample data for cities and regions in India
df_geo = pd.DataFrame({
    'city': ['Mumbai', 'Delhi', 'Bengaluru', 'Kolkata', 'Chennai', 'Hyderabad', 'Ahmedabad'],
    'region': ['Maharashtra', 'Delhi NCR', 'Karnataka', 'West Bengal', 'Tamil Nadu', 'Telangana', 'Gujarat'],
    'impressions': [25000, 22000, 20000, 18000, 16000, 15000, 14000],
    'views': [4000, 3500, 3000, 2800, 2400, 2300, 2100]
})

# Calculate VTR
df_geo['VTR'] = (df_geo['views'] / df_geo['impressions'] * 100).round(2)

# Convert to string
df_geo_str = df_geo.to_string(index=False)

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-pro')

campaign_context = '''This is a campaign for McCain Frozen Foods that ran during the monsoon season on YouTube CTV, targeting a female audience aged 25–44.'''

prompt = PromptTemplate(
    template="""
You are a marketing analyst. Based on the following geographical campaign performance data, write a short commentary (2–3 insightful points) highlighting key trends:
{data}
Also, consider the campaign context below while analyzing performance (e.g., city/region-based engagement variations):
{context}
Keep the tone concise, analytical, and suitable for inclusion in a post-campaign insights deck.
""",
    input_variables=['data', 'context']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'data':df_geo_str, 'context': campaign_context})

print(result)

#result:
### Campaign Performance: Geographical Insights

# *   **Top Metros Drive Highest Engagement:** The campaign resonated most strongly in Mumbai (16.00% VTR) and Delhi (15.91% VTR), which not only received the highest impressions but also converted them into views most efficiently. This suggests the message of convenient, at-home snacking during the monsoon season was particularly effective for the target audience in these fast-paced urban centers.

# *   **Cultural Affinity Boosts Kolkata:** Kolkata delivered a notably high VTR of 15.56%, outperforming larger markets like Bengaluru in terms of engagement. This strong performance is likely tied to the region's deep-rooted culture of enjoying hot, fried snacks during the monsoon, making the McCain value proposition highly relevant and appealing.

# *   **Consistent Performance in Southern & Western Markets:** Key markets like Bengaluru, Chennai, and Ahmedabad showed solid, consistent engagement with a 15.00% VTR. While this indicates a strong baseline performance, it also highlights an opportunity to explore more specific regional nuances in future creatives to potentially lift engagement to the levels seen in Mumbai and Delhi.