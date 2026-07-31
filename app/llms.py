from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from .config import GOOGLE_API_KEY, WRITER_MODEL, REVIEWER_MODEL

search_tool = TavilySearch(max_results=3)  # fixed: was max_result

tools = [search_tool]

writer_llm = ChatGoogleGenerativeAI(
    model=WRITER_MODEL,
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY,
)
writer_llm_with_tools = writer_llm.bind_tools(tools)

reviewer_llm = ChatGroq(
    model=REVIEWER_MODEL,
    temperature=0.2,
)
