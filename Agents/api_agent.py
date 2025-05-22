import os
from alpha_vantage.timeseries import TimeSeries
from crewai.tools import tool
import requests # Import the requests library
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
load_dotenv()
import os

# Go one directory up to access query.txt
query_path = os.path.join(os.path.dirname(__file__), "..", "query.txt")

# Normalize path
query_path = os.path.abspath(query_path)

# Read the query
with open(query_path, "r", encoding="utf-8") as f:
    query = f.read().strip()

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
# Ensure API_KEY is not None before proceeding or add error handling
openai.api_key=os.getenv('OPENAI_API_KEY')
@tool("get_stock_price")
def get_stock_price(symbol: str) -> str:
    """
    Fetches the daily stock price data for a given symbol from AlphaVantage.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL").

    Returns:
        str: A JSON string containing the daily stock data.
    """
    # Use the consistent environment variable name and API key
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}'
    r = requests.get(url)
    r.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    data = r.json()
    return str(data) # Return string representation of the dictionary data

llm = ChatOpenAI(model_name="gpt-4o")

stock_agent = Agent(
    llm = llm,
    role="Fetches stock price data and trends",
    goal="Polls the real-time & historical market data from AlphaVantage using the given tool",
    backstory="An expert in fetching stock price data and trends from AlphaVantage.", # Added backstory

    tools=[get_stock_price],
    verbose=True
)
stock_task = Task(
    description=f"""fetches the stock price data trends based on the user's query by extracting the suitable symbol and give that symbol to tool.
    Query:
    {query}
    
    """,
    agent=stock_agent,
    expected_output="Stock Prices based on query",
    output_file="stock_prices.json",
)