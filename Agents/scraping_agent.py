import os
from crewai.tools import tool
import requests # Import the requests library
from dotenv import load_dotenv
import openai
load_dotenv()
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool
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
sources = {
      "CNBC": "https://www.cnbc.com/finance/"
  }
scraper_tool = ScrapeWebsiteTool()

scraper_agent = Agent(
    llm=llm,
    role="Financial News Collector",
    goal="Extract financial headlines from top market websites",
    backstory="An expert in extracting financial headlines from top market websites.",
    tools=[scraper_tool],
    verbose=True
)
news_task = Task(
    description=f"""Scrape the financial news for the last 2-3 days from the sources provided below.
    Sources:
    {sources}
    """,
    expected_output="Financial new from last 2-3 days from the provided sources. i.e. you have to give Headlines and brief story associated with the headline.",
    agent=scraper_agent,
    output_file="scraped_news.txt"
)
