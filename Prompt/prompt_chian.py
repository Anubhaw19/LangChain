# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

st.header('Research Tool.')

paper_input = st.selectbox("Select Research Paper Name",["Select...", "Attention Is All you need","BERT: Pre-training of Deep Bidirectional Transformer", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input  = st.selectbox("Select Explanation Style",[ "Beginner-Friendly","Technical", "Code-Oriented", "Mathematical"]) 

length_input = st.selectbox("Select Explanation Length",["Short (1-2 paragraph)","Medium (3-5 paragraph)", "Long (detailed explanation)"])

template = load_prompt('template.json')

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

if st.button('Summarise'):
    chain = template | model
    result = chain.invoke({
    'paper_input' : paper_input,
    'style_input' : style_input,
    'length_input' : length_input
    })
    st.write(result.content)