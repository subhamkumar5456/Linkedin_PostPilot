import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))

WRITER_MODEL = os.getenv("WRITER_MODEL", "gemini-3.5-flash-lite")
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", "llama-3.3-70b-versatile")
