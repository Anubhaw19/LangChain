# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

st.header('Research Tool.')
# user_input = st.text_input('Enter your prompt')

paper_input = st.selectbox("Select Research Paper Name",["Select...", "Attention Is All you need","BERT: Pre-training of Deep Bidirectional Transformer", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input  = st.selectbox("Select Explanation Style",[ "Beginner-Friendly","Technical", "Code-Oriented", "Mathematical"]) 

length_input = st.selectbox("Select Explanation Length",["Short (1-2 paragraph)","Medium (3-5 paragraph)", "Long (detailed explanation)"])

#prompt tempalte
# template = PromptTemplate(
#     template = """
# Please summarise the research paper titled "{paper_input}" with the following specifications:
# Explanation  Style : {style_input}
# Explanation Length : {length_input}
# 1.Mathematical Details : 
#  - Include relevant mathematical equations if present in the paper.
#  - Explain the mathematical concepts using simple, ituitive code snippets where applicable.
# 2.Analogies:
#  - Use relatable analogies to simplify complex ideas.
# If certain information is not available instead of guessing. Ensure the summary is clear, accurate, and aligned with the provided style and length, also don't show the unnecessary text , just the content
# """,
# input_varibles = ['paper_input','style_input','length_input'],
# validate_template = False
# )

template = load_prompt('template.json')

#fill the place holders

prompt = template.invoke({
    'paper_input' : paper_input,
    'style_input' : style_input,
    'length_input' : length_input
})

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

 

if st.button('Summarise'):
    # result = model.invoke(user_input)
    result = model.invoke(prompt)
    st.write(result.content)