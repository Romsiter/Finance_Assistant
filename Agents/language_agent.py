from crewai import Agent
import os
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from Agents.api_agent import stock_task
from Agents.retriever_agent import retrieval_task
from Agents.analysis_agent import analysis_task
load_dotenv()
import os

# Go one directory up to access query.txt
query_path = os.path.join(os.path.dirname(__file__), "..", "query.txt")

# Normalize path
query_path = os.path.abspath(query_path)

# Read the query
with open(query_path, "r", encoding="utf-8") as f:
    query = f.read().strip()
# Ensure API_KEY is not None before proceeding or add error handling
openai.api_key=os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model_name="gpt-4o")

language_agent = Agent(
    llm=llm,
    role="Financial Language Synthesis Expert",
    goal="Craft a final market briefing answer using stock_prices data obtained from stock_agent, relevant news data from retriever_agent and analysis data from analysis agent",
    backstory="An AI analyst skilled in summarizing financial context into human-readable briefs.",
    verbose=True
)

synthesis_task = Task(
    description=(
        f'''Get context from stock_prices given by stock_agent, relevant information based on query provided by retriever_agent and analysis data given by analysis_agent to answer the given user query "
        User Query: {query}

        "You must:\n"
        "- Summarize stock movement trends\n"
        "- Highlight any significant price changes\n"
        "- Mention closing prices if valid\n" 
        "- Mention earnings surprises if available\n"
        "- Reflect market sentiment\n"
        "- Keep the answer brief, but insightful'''
    ),
    expected_output="Final 3–5 sentence spoken-style answer to the query.",
    agent=language_agent,
    context=[stock_task,retrieval_task,analysis_task],
    output_file="final_answer.txt"
)

