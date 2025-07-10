from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"), breakpoint_threshold_type = "standard_deviation",
    breakpoint_threshold_amount = 1
)
sample = '''
Technology:
Smart home technology is transforming everyday life by connecting devices like lights, thermostats, and security systems to the internet. These systems can be controlled remotely or even automated to adjust based on user habits, making homes more convenient and energy-efficient.
Travel:
The Alps stretch across countries like France, Switzerland, and Austria, offering year-round attractions. In winter, they’re a haven for skiing and snowboarding, while summer brings opportunities for hiking, biking, and exploring charming alpine villages.
Health:
Hydration plays a key role in maintaining overall health. Drinking enough water supports digestion, keeps energy levels stable, and improves focus. Even slight dehydration can impact mood and physical performance.

'''
docs = text_splitter.create_documents([sample])
print(len(docs))
for index,document in enumerate(docs):
    print('document:',index, end="\t")
    print(document.page_content)
# print(docs)



# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
# vector = embeddings.embed_query("hello, world!")