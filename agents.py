# Dependencies ↓
from langchain.agents import create_agent
from tools import web_search,scrape_url
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatMessagePromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatMistralAI(model="mistral-small-2506")

# 1st Agent
def build_search_agent():
    return create_agent(
        model = llm,
        tools = [web_search]
)


# 2nd Agent
def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url]
)

# writer chain

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])


writer_chain = writer_prompt | llm | StrOutputParser()
# writer_prompt → this is a component that takes the raw Output variables i.e. topic, research and formats them into a structured prompt object.
# llm → the formatted prompt goes through the LLM to create a raw text response.
# StrOutputParser → the response given by the llm goes through a parser that extracts just the clean string text, removing the metadata.


# critic_chain
critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()

















