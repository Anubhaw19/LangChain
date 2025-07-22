from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import numpy as np

load_dotenv()

# Sample creative performance data
df_creative = pd.DataFrame({
    'creative_name': [
        'Bumper Ad – 6s', 
        'Recipe Showcase – 15s', 
        'Emotional Storytelling – 30s', 
        'Regional Language Voiceover – 15s',
        'Call-to-Action Focus – 10s'
    ],
    'impressions': [30000, 25000, 20000, 18000, 15000],
    'views': [5000, 7000, 6000, 5500, 4000],
    'clicks': [300, 450, 350, 380, 250]
})

# Calculate VTR and CTR
df_creative['VTR (%)'] = (df_creative['views'] / df_creative['impressions'] * 100).round(2)
df_creative['CTR (%)'] = (df_creative['clicks'] / df_creative['impressions'] * 100).round(2)

# Convert to string for prompt input (optional)
df_creative_str = df_creative.to_string(index=False)

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-pro')

campaign_context = '''This is a campaign for McCain Frozen Foods that ran during the monsoon season on YouTube CTV, targeting a female audience aged 25–44.'''

prompt = PromptTemplate(
    template="""
You are a marketing analyst. Based on the following creative-level performance data, write a concise commentary (2–3 key points in a paragraph) highlighting how different ad creatives performed:
{data}
Include any interesting trends, such as creative types driving higher engagement (VTR/CTR), and suggest possible reasons based on content style or audience behavior.
Use the campaign context below to support your reasoning:
{context}
Keep the tone analytical, succinct, and suitable for a post-campaign insights deck.
""",
    input_variables=['data', 'context']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'data':df_creative_str, 'context': campaign_context})

print(result)

#result:

### **Creative Performance Highlights**

# *   **Regional Language Drives Highest Engagement:** The "Regional Language Voiceover" creative was the top performer, delivering the highest View-Through Rate (30.56%) and Click-Through Rate (2.11%). This suggests that localized content created a strong personal connection with the target audience, cutting through generic advertising and prompting a higher level of interest and action.

# *   **Storytelling and Utility Outperform Direct Calls-to-Action:** Longer-form creatives focused on storytelling ("Emotional Storytelling") and practical value ("Recipe Showcase") achieved significantly higher engagement than shorter, more direct ads. The "Recipe Showcase" (28% VTR, 1.80% CTR) likely resonated by providing meal inspiration during the monsoon season, a time associated with enjoying hot snacks at home. This indicates our audience on CTV is more receptive to engaging narratives and useful content than to brief, transactional messages.

#Short:

# The campaign data reveals that creatives combining cultural relevance and practical utility drove the highest engagement. The **Regional Language Voiceover** creative was the top performer, delivering the highest VTR (30.56%) and CTR (2.11%), which suggests that localized content created a stronger connection with the target audience on YouTube CTV. Similarly, the **Recipe Showcase** performed strongly, indicating that our female audience is highly receptive to content that provides actionable meal solutions, especially during the monsoon season. While the longer **Emotional Storytelling** ad captured attention with a high VTR (30.00%), its slightly lower CTR suggests that while effective for brand-building, utility-focused creatives were more successful at driving immediate action in this campaign.