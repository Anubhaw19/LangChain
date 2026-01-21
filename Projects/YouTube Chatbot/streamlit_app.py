import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

# ---------------- Utility ----------------

def extract_video_id(youtube_url):
    """
    Extracts video ID from YouTube URL or returns input if already ID
    """
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, youtube_url)
    if match:
        return match.group(1)
    return youtube_url.strip()  # fallback if user pasted ID

# ---------------- Cached Functions ----------------

@st.cache(show_spinner=False, allow_output_mutation=True)
def fetch_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
        transcript_text = " ".join([s.text for s in fetched_transcript.snippets])
        return transcript_text
    except TranscriptsDisabled:
        return None

@st.cache(show_spinner=False, allow_output_mutation=True)
def build_vectorstore(transcript_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.create_documents([transcript_text])

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )
    return vectorstore

def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.2)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, say "I don't know".

        Context:
        {context}

        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()
    chain = parallel_chain | prompt | llm | parser
    return chain

# ---------------- UI ----------------

st.set_page_config(page_title="YouTube RAG Chat", layout="wide")
st.title("🎥 YouTube Video Q&A")

youtube_url = st.text_input(
    "Paste YouTube URL or Video ID",
    placeholder="https://www.youtube.com/watch?v=Cbqtxys2qPM"
)

question = st.text_input(
    "Ask a question about the video",
    value="Can you summarize the video?"
)

run_button = st.button("Ask")

# ---------------- App Logic ----------------

if run_button:
    if not youtube_url.strip():
        st.warning("Please enter a YouTube URL or Video ID")
        st.stop()

    video_id = extract_video_id(youtube_url)

    with st.spinner("Fetching transcript..."):
        transcript_text = fetch_transcript(video_id)

    if transcript_text is None:
        st.error("No transcript available for this video.")
        st.stop()

    with st.spinner("Building vector database..."):
        vectorstore = build_vectorstore(transcript_text)

    with st.spinner("Generating answer..."):
        chain = build_chain(vectorstore)
        answer = chain.invoke(question)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Show full transcript"):
        st.write(transcript_text)
