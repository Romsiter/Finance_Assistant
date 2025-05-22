# tools/analysis_tools.py

import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Type
from pydantic import BaseModel, Field
from crewai.tools.base_tool import BaseTool
from sentence_transformers import SentenceTransformer
from transformers import pipeline # Ensure pipeline is imported
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI

import json
from crewai import Agent, Task, Crew, Process
from Agents.api_agent import stock_task
from Agents.retriever_agent import retrieval_task
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

# ─── 1. Compute Returns from stock_prices.json ────────────────────────────────
class ReturnsInput(BaseModel):
    file_path: str = Field(..., description="Path to the structured stock_prices.json from Alpha Vantage")


# ─── Updated Compute Returns Tool ─────────────
class ComputeReturnsTool(BaseTool):
    name: str = "compute_returns"
    description: str = "Computes daily returns from Alpha Vantage-style structured JSON."
    args_schema: Type[BaseModel] = ReturnsInput

    def _run(self, file_path: str) -> Tuple[bool, str]:
        file_path="stock_prices.json"
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        with open(file_path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return False, "File is not valid JSON."

        if "Time Series (Daily)" not in data:
            return False, "No daily time series data found."

        ts_data = data["Time Series (Daily)"]

        try:
            records = [
                (datetime.strptime(date_str, "%Y-%m-%d"), float(metrics["4. close"]))
                for date_str, metrics in ts_data.items()
                if "4. close" in metrics
            ]
        except Exception as e:
            return False, f"Failed to parse time series: {e}"

        records.sort(reverse=True)

        if len(records) < 2:
            return True, "Not enough data points to compute returns."

        results = []
        for i in range(len(records) - 1):
            date, close = records[i]
            _, prev_close = records[i + 1]
            change = (close - prev_close) / prev_close * 100
            results.append(f"{date.date()}: return = {change:+.2f}%")

        return True, "\n".join(results)
# ─── 3. Sentiment Analysis Tool ─────────────────────────────────
class SentimentInput(BaseModel):
    headlines: List[str] = Field(
        ..., description="List of news headlines to analyze"
    )

class SentimentAnalysisTool(BaseTool):
    # Add type annotations for name and description
    name: str = "analyze_sentiment"
    description: str = "Analyze sentiment of headlines and give overall tilt"
    args_schema: Type[BaseModel] = SentimentInput

    # Remove the __init__ method that was causing the error

    def _run(self, headlines: List[str]) -> Tuple[bool, str]:
        # Initialize the pipeline here within the run method
        pipe = pipeline("sentiment-analysis")

        results = pipe(headlines)
        pos = sum(1 for r in results if r["label"] == "POSITIVE")
        neg = sum(1 for r in results if r["label"] == "NEGATIVE")
        neu = len(headlines) - pos - neg
        overall = (
            "positive" if pos > neg else "negative" if neg > pos else "neutral"
        )
        detail = "\n".join(
            f"{h} → {r['label']} ({r['score']:.2f})"
            for h, r in zip(headlines, results)
        )
        print(headlines)
        summary = f"Overall sentiment: **{overall}** (P:{pos} N:{neg} U:{neu})"
        return True, summary + "\n\n" + detail
    
    
    
    
analysis_agent = Agent(
llm=llm,
name="AnalysisAgent",
role="Perform quantitative returns and sentiment analysis",
goal="Produce a concise analytical brief given price data obtained from stock_agent and sentiment analysis on headlines obtained from retriever_agent",
backstory=(
    "A quantitative finance expert skilled in time-series analysis, "
    " NLP-based sentiment."
),
tools=[
    ComputeReturnsTool(),
    SentimentAnalysisTool(),
],
verbose=True,
)

analysis_task = Task(
    description="""
1. Compute daily returns from the stocks prices obtained from stock_agent.
2. Analyze sentiment of provided headlines obtained from retriever_agent.
""",
    expected_output="Returns % of daily returns computed from stock prices obtained from the stock_agent,  and detailed sentiment summary of sentimental analysis done on headlines given by retriever_agent",
    agent=analysis_agent,
    context=[stock_task,retrieval_task],
    output_file="analysis_brief.txt",
)