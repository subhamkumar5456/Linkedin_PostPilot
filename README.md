<div align="center">

# ✈️ LinkedIn PostPilot

### *Your AI-powered LinkedIn content team — in a terminal.*

An agentic pipeline that **researches the web → writes a post → reviews it → iterates** — autonomously, until the content is publish-ready.

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Graph-7C3AED?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/Gemini-AI%20Writer-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)
[![Groq](https://img.shields.io/badge/Groq-AI%20Reviewer-F97316?style=for-the-badge)](https://groq.com/)
[![Tavily](https://img.shields.io/badge/Tavily-Web%20Search-10B981?style=for-the-badge)](https://tavily.com/)
[![SQLite](https://img.shields.io/badge/SQLite-Run%20Metrics-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

</div>

---

## ✨ What It Does

You type a topic. PostPilot handles everything else:

| Step | Agent | What happens |
|------|-------|-------------|
| 🔍 **Research** | Tavily Search | Fetches up-to-date stats, trends, and context |
| ✍️ **Write** | Gemini (Writer) | Crafts a compelling LinkedIn post from the research |
| 🧐 **Review** | LLaMA 3.3 (Reviewer) | Scores the draft against 7 strict quality criteria |
| 🔄 **Refine** | Gemini (Writer) | Rewrites the post based on reviewer feedback |
| ✅ **Deliver** | — | Outputs a polished, publish-ready post |
| 📊 **Track** | SQLite | Logs latency, tokens, cost, and approval rate per run |

> No hashtags. No fluff. Just sharp, human-sounding LinkedIn content.

---

## 🧠 Agentic Pipeline

```
  ┌─────────────────────────────────────────────────────────┐
  │                   User inputs a topic                   │
  └───────────────────────────┬─────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  ✍️  Writer Node  │  ← Gemini
                    │  (builds prompt) │
                    └────────┬────────┘
                             │
               ┌─────────────┴──────────────┐
               │  Does it need web search?   │
               └─────────┬──────────────────┘
           Yes ◄──────────┘          └──────────► No
            │                                     │
            ▼                                     │
  ┌──────────────────┐                            │
  │ 🌐 Tavily Search  │                            │
  └────────┬─────────┘                            │
           │  (results added to messages)          │
           ▼                                      │
  ┌─────────────────┐                             │
  │  ✍️  Writer Node  │  ← generates post          │
  │  (reads results) │     from search data        │
  └────────┬─────────┘                            │
           │                                      │
           └─────────────────┬────────────────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │  📋 Extract Draft │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  🔍 Reviewer Node │  ← LLaMA 3.3 / Groq
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         ✅ APPROVED               ❌ REJECTED
              │                         │
              ▼                         ▼
       Final Post 🎉          Back to Writer Node
                              (up to MAX_ATTEMPTS)
```

---

## 🏆 Quality Criteria (Auto-Enforced)

The reviewer rejects any draft that fails even **one** of these:

| # | Criterion | Standard |
|---|-----------|----------|
| 1 | **Strong Hook** | First line must grab attention immediately |
| 2 | **Clear Takeaway** | One sharp, actionable insight |
| 3 | **Skimmable** | Short paragraphs — easy to skim |
| 4 | **Ideal Length** | 150–200 words, no padding |
| 5 | **Engaging Ending** | Closes with a question or CTA |
| 6 | **Human Tone** | Professional, not robotic or corporate |
| 7 | **No Hashtags** | Clean, algorithm-independent content |

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | State-machine agentic graph |
| **Writer LLM** | Google Gemini (`gemini-3.5-flash-lite`) | Post generation |
| **Reviewer LLM** | Groq / LLaMA 3.3 70B Versatile | Quality gating & feedback |
| **Web Search** | [Tavily](https://tavily.com) | Real-time research |
| **Metrics** | SQLite (`data/metrics.db`) | Latency, tokens, cost, approval |
| **Framework** | LangChain | LLM & tool integrations |
| **Config** | `python-dotenv` | Secure API key management |

---

## 📁 Project Structure

```
Linkedin_PostPilot/
│
├── run.py                   # 🚀 Entry point  →  python run.py
│
├── app/                     # Core pipeline (modular)
│   ├── __init__.py
│   ├── config.py            # Env vars, model names, MAX_ATTEMPTS
│   ├── state.py             # LangGraph State (TypedDict)
│   ├── prompts.py           # Writer & reviewer system prompts
│   ├── llms.py              # LLM + Tavily tool initialization
│   ├── nodes.py             # writer_node, reviewer_node, routers
│   ├── graph.py             # Graph construction & compilation
│   └── main.py              # run_cli() — CLI entry function
│
├── tracking/                # Run metrics & SQLite persistence
│   ├── db.py                # Schema, migrations, read helpers
│   └── metrics.py           # RunMetrics class, log_call(), get_usage()
│
├── data/
│   └── metrics.db           # SQLite DB (auto-created on first run)
│
├── requirements.txt
├── .env                     # 🔑 API keys (never committed)
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Clone

```bash
git clone https://github.com/subhamkumar5456/Linkedin_PostPilot.git
cd Linkedin_PostPilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional overrides (defaults shown)
WRITER_MODEL=gemini-3.5-flash-lite
REVIEWER_MODEL=llama-3.3-70b-versatile
MAX_ATTEMPTS=3
```

> 🔑 **Get your free keys:**
> - **Google Gemini** → [aistudio.google.com](https://aistudio.google.com/app/apikey)
> - **Groq** → [console.groq.com](https://console.groq.com/keys)
> - **Tavily** → [app.tavily.com](https://app.tavily.com)

### 5. Run

```bash
python run.py
```

---

## 🖥️ Example Session

```
=======================================================
Welcome to the LinkedIn Post Generator
=======================================================

This tool will draft a LinkedIn post for you, review it
itself, and iterate until it's publish-ready.
=======================================================

What topic do you want a LinkedIn post about?
> The rise of agentic AI in 2025

Starting generation...

 generated Post :
"Agentic AI isn't the future — it's already your coworker."
...

[Verdict:APPROVED]
[Feedback: Strong hook, clear takeaway, well-structured...]

Post has been ✅ approved

=======================================================
FINAL LINKEDIN POST
=======================================================
"Agentic AI isn't the future — it's already your coworker."
...
=======================================================
Total attempts: 1
Approved: True

=======================================================
RUN METRICS
=======================================================
run_id:            20260801_033300_481321
topic:             The rise of agentic AI in 2025
attempts_used:     1
approved:          True
one_shot_approval: True
total_latency_s:   3.565
total_tokens:      5025
num_llm_calls:     2

All metrics stored in SQLite at: data/metrics.db
```

---

## 📊 Run Metrics

Every run is automatically tracked in `data/metrics.db`. Query it any time:

```bash
# All runs summary
sqlite3 data/metrics.db "SELECT run_id, topic, attempts_used, approved, total_latency_s FROM runs ORDER BY created_at DESC;"

# Per-call breakdown for a specific run
sqlite3 data/metrics.db "SELECT node, attempt, latency_s, total_tokens, verdict FROM llm_calls WHERE run_id='<run_id>';"

# Approval rate across all runs
sqlite3 data/metrics.db "SELECT ROUND(AVG(approved)*100,1) || '%' AS approval_rate FROM runs;"
```

---

## 🌐 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | ✅ | — | Gemini writer LLM |
| `GROQ_API_KEY` | ✅ | — | LLaMA reviewer LLM |
| `TAVILY_API_KEY` | ✅ | — | Web search tool |
| `WRITER_MODEL` | ☑️ | `gemini-3.5-flash-lite` | Override writer model |
| `REVIEWER_MODEL` | ☑️ | `llama-3.3-70b-versatile` | Override reviewer model |
| `MAX_ATTEMPTS` | ☑️ | `3` | Max write→review iterations |

---

## 🔄 Agent State

```python
class State(TypedDict):
    topic           : str    # Your input topic
    messages        : list   # Full conversation history
    draft           : str    # Current post draft
    review_feedback : str    # Reviewer's last feedback
    is_approved     : bool   # Whether the post passed review
    attempt         : int    # Iteration counter (capped at MAX_ATTEMPTS)
```

---

## 🚧 Roadmap

- [ ] 🌐 Web UI (Streamlit or FastAPI frontend)
- [ ] 📅 Scheduled generation via cron
- [ ] 🎨 Multiple post styles: storytelling, listicle, thought leadership
- [ ] 🔗 LinkedIn API integration for direct publishing
- [ ] 📈 Engagement analytics for generated posts
- [ ] 🗂️ Post history and saved drafts

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [LangGraph](https://github.com/langchain-ai/langgraph) — for the agentic state-machine framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) — for state-of-the-art content generation
- [Groq](https://groq.com/) — for blazing-fast LLaMA inference
- [Tavily](https://tavily.com/) — for real-time web search

---

<div align="center">

Built with ❤️ by **Subham Kumar** &nbsp;|&nbsp;
[LinkedIn](https://www.linkedin.com/in/subhamkumar5456) &nbsp;|&nbsp;
[GitHub](https://github.com/subhamkumar5456)

</div>
