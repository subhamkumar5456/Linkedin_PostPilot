from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from .config import GOOGLE_API_KEY, WRITER_MODEL, REVIEWER_MODEL
from tracking.llm_tracker import TrackedLLM

search_tool = TavilySearch(max_results=3)

tools = [search_tool]

# Wrap each model with TrackedLLM — every .invoke() call is automatically
# timed, token-counted, and logged to RunMetrics. No boilerplate needed in nodes.
writer_llm = TrackedLLM(
    ChatGoogleGenerativeAI(
        model=WRITER_MODEL,
        google_api_key=GOOGLE_API_KEY,
    ),
    model_key="gemini-writer",
    node="writer",
)
# bind_tools() on a TrackedLLM returns another TrackedLLM, so the
# tool-bound variant is tracked too.
writer_llm_with_tools = writer_llm.bind_tools(tools)

reviewer_llm = TrackedLLM(
    ChatGroq(
        model=REVIEWER_MODEL,
        temperature=0.2,
    ),
    model_key="groq-reviewer",
    node="reviewer",
)
