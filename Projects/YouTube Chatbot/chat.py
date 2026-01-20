from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
load_dotenv()
# Step 1a - Indexing (Document Ingestion)-------------------
video_id = "Cbqtxys2qPM"

try:
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=['en'])

    # Extract plain text
    transcript_text = " ".join([snippet.text for snippet in fetched_transcript.snippets])

    # print(transcript_text)

except TranscriptsDisabled:
    print("No captions available for this video.")

# Step 1b - Indexing (Text Splitting)-------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.create_documents([transcript_text])
# print('chunk length: ',len(chunks))
# print(chunks[1])

# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)-------------------
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vector_store = FAISS.from_documents(chunks, embeddings)
vectorstore = FAISS.from_documents(
    documents = chunks,
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
)
# print(vectorstore.index_to_docstore_id)
# print(vectorstore.get_by_ids(['cc5f4cf4-102f-4bb5-8264-2e43f96040b7']))

# Step 2: Retrieval-----------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# print(retriever.invoke('What is the topic of discussion?'))

# Step 3 - Augmentation----------------------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', temperature=0.2)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
# question          = "will trump attack greenland? if yes then what was discussed"
# retrieved_docs    = retriever.invoke(question)
# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(context_text)
# final_prompt = prompt.invoke({"context": context_text, "question": question})

# Step 4 - Generation-------------
# answer = llm.invoke(final_prompt)
# print(answer.content)

# ------------------------BUILDING A CHAIN-----------------------------------------
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
# print(parallel_chain.invoke('Can you summarize the video'))


parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

result = main_chain.invoke('Can you summarize the video')

print(result)

'''
Result:
The video introduces "First Post Live," a new show aiming to provide news that gets straight to the point, covering topics from 
battlefields to negotiation tables. It discusses a message sent to Donald Trump, which Trump leaked on social media. The message states
the sender will use media engagements in Davos to highlight Trump's work in Syria, Gaza, and Ukraine, and is committed to finding a way
forward in Greenland. The transcript also mentions that Europeans are making a "last-ditch effort" to "reign in Donald Trump," though
indications suggest it won't work, and that Trump has posted two AI images on social media. There is a concern that Greenland is
"basically a sitting duck," with a hope that diplomacy can prevent a crisis.
'''