from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash')

prompt1 = PromptTemplate(
    template = 'Generate 3 line report about the {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate  3 questions answer from the following text\n {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided report and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables = ['notes','quiz'] 
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model |parser,
    'quiz': prompt2 | model | parser
    })

merge_chain = prompt3 | model | parser
chain = parallel_chain | merge_chain

text = 'Large Language Models (LLMs) are advanced AI systems trained on vast amounts of text to understand and generate human-like language. They can perform tasks like writing, summarizing, coding, and answering questions by predicting the next word in a sequence. While powerful and versatile, they may sometimes produce inaccurate information and are sensitive to how prompts are phrased. LLMs are widely used in chatbots, virtual assistants, content creation, and code generation tools.'

result = chain.invoke({'topic':'cricket','text':text})

print(result)

chain.get_graph().print_ascii()