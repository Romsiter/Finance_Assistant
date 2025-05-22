# tools/retriever_tool.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from crewai.tools import tool
import os
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
import os
load_dotenv()

# Go one directory up to access query.txt
query_path = os.path.join(os.path.dirname(__file__), "..", "query.txt")

# Normalize path
query_path = os.path.abspath(query_path)

# Read the query
with open(query_path, "r", encoding="utf-8") as f:
    query = f.read().strip()
# Ensure API_KEY is not None before proceeding or add error handling
openai.api_key=os.getenv('OPENAI_API_KEY')
embedding_model_id = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_id,
            )   
source_texts = []

def build_index_from_files(news_file="scraped_news.txt"):
    global source_texts
    source_texts.clear()

    # Load and split scraped news
    if os.path.exists(news_file):
        with open(news_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.read().splitlines() if line.strip()]
            source_texts.extend(lines)

    if source_texts:   
        return FAISS.from_texts(source_texts, embeddings)
@tool("retriever_tool")
def retriever_tool():
    """
    Retrieves relevant financial news and stock price summaries for the user's query.
    Combines scraped_news.txt and stock_prices.json bullet data.
    """
    global query

    if not query:
        return "No query provided."
    
    vector_store=build_index_from_files()
    results=vector_store.similarity_search(query, k=5)
    print(results)

    if not results:
      return "No relevant information found"
    results_list = []
    for res in results:
        results_list.append(res.page_content)
    results = "\n\n".join(results_list)
    return results

retriever_agent = Agent(
    #llm=ChatOpenAI(model_name="gpt-4o"),
    name="RetrieverAgent",
    role="Financial Research Analyst",
    goal="Retrieving relevant financial news to the user query",
    backstory=(
        "A data-savvy agent that indexes local news and price logs, "
        "and uses semantic search to fetch contextually relevant insights."
    ),
    tools=[retriever_tool],
    verbose=True
)

retrieval_task = Task(
    description=(
        "Fetch the most relevant snippets from previous news based on the user's query."
    ),
    expected_output="Top-5 relevant news headlines.",
    agent=retriever_agent,
    output_file="retrieved_info.txt"
)

