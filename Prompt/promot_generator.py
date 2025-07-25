from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template = """
Please summarise the research paper titled "{paper_input}" with the following specifications:
Explanation  Style : {style_input}
Explanation Length : {length_input}
1.Mathematical Details : 
 - Include relevant mathematical equations if present in the paper.
 - Explain the mathematical concepts using simple, ituitive code snippets where applicable.
2.Analogies:
 - Use relatable analogies to simplify complex ideas.
If certain information is not available instead of guessing. Ensure the summary is clear, accurate, and aligned with the provided style and length, also don't show the unnecessary text , just the content
""",
input_varibles = ['paper_input','style_input','length_input'],
validate_template = False
)

template.save('template.json')