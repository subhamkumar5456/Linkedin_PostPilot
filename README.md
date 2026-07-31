<h1 align="center">
  🚀 LinkedIn PostPilot
</h1>

<p align="center">
  <b>An AI-powered agentic pipeline that autonomously writes, reviews, and refines LinkedIn posts — until they're publish-ready.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-Agentic%20Graph-7C3AED?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini-AI%20Writer-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-AI%20Reviewer-F97316?style=for-the-badge&logo=groq&logoColor=white" />
  <img src="https://img.shields.io/badge/Tavily-Web%20Search-10B981?style=for-the-badge" />
</p>

---

## ✨ What is LinkedIn PostPilot?

**LinkedIn PostPilot** is an **agentic AI system** built with [LangGraph](https://github.com/langchain-ai/langgraph) that acts as your personal LinkedIn content team. You give it a topic — it handles the rest:

- 🔍 **Researches** the web for fresh, current information using Tavily
- ✍️ **Writes** a compelling LinkedIn post using Google Gemini
- 🧐 **Reviews** the draft with a strict AI reviewer powered by Groq (LLaMA 3.3)
- 🔄 **Iterates** up to 3 times until the post passes quality checks
- ✅ **Delivers** a polished, publish-ready LinkedIn post

> No hashtags. No fluff. Just high-quality, human-sounding content.

---

## 🧠 How It Works — The Agentic Pipeline

```
 User Input (Topic)
        │
        ▼
  ┌──────────────┐       uses web search?
  │  🖊️  Writer   │──────────────────────────┐
  │   (Gemini)   │                          ▼
  └──────────────┘               ┌─────────────────────┐
        │                        │  🌐 Tavily Web Search │
        │ (draft ready)          └─────────────────────┘
        ▼                                   │
  ┌──────────────┐                          │
  │ Extract Draft│◄─────────────────────────┘
  └──────────────┘
        │
        ▼
  ┌──────────────┐
  │  🔍 Reviewer  │  (Groq / LLaMA 3.3 70B)
  │              │
  └──────┬───────┘
         │
    ┌────┴────┐
    │ VERDICT │
    └────┬────┘
         │
   ✅ APPROVED ──────────────────► Final Post 🎉
         │
   ❌ REJECTED ──────────────────► Back to Writer
         │                        (up to 3 attempts)
         └──────────────────────► MAX ATTEMPTS ► Final Post
```

The graph enforces **quality gates** — the post is only output when it meets all 7 content criteria or the max iteration limit is reached.

---

## 🏆 Content Quality Criteria (Auto-Enforced)

The AI reviewer checks every draft against these strict standards:

| # | Criterion | Description |
|---|-----------|-------------|
| 1 | **Strong Hook** | First line must grab attention immediately |
| 2 | **Clear Takeaway** | One valuable, actionable insight |
| 3 | **Skimmable Format** | Short paragraphs, easy to read |
| 4 | **Ideal Length** | 150–200 words — not too short, not too long |
| 5 | **Engaging Ending** | Closes with a question or call-to-action |
| 6 | **Human Tone** | Professional, but not robotic or corporate |
| 7 | **No Hashtags** | Clean, algorithm-independent content |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | Agentic graph state machine |
| **Writer LLM** | Google Gemini (`gemini-3.1-pro-preview`) | Content generation |
| **Reviewer LLM** | Groq / LLaMA 3.3 70B Versatile | Quality review & feedback |
| **Web Search** | [Tavily](https://tavily.com) | Real-time web research |
| **Framework** | LangChain | LLM & tool integrations |
| **Config** | `python-dotenv` | Secure API key management |

---

## 📁 Project Structure

```
linkedin-postpilot/
│
├── main.py              # 🧠 Core agentic pipeline (writer → reviewer → loop)
├── main_test.py         # 🧪 Tests
├── requirements.txt     # 📦 Python dependencies
├── .env                 # 🔑 API keys (not committed to Git)
├── .gitignore           # 🙈 Git ignore rules
└── README.md            # 📖 You are here!
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/linkedin-postpilot.git
cd linkedin-postpilot
```

### 2. Set Up a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

> 🔑 **Get your keys:**
> - **Google Gemini** → [aistudio.google.com](https://aistudio.google.com/app/apikey)
> - **Groq** → [console.groq.com](https://console.groq.com/keys)
> - **Tavily** → [tavily.com](https://tavily.com)

### 5. Run the App

```bash
python main.py
```

**Example session:**

```
═══════════════════════════════════════════════════════
Welcome to the LinkedIn Post Generator
═══════════════════════════════════════════════════════

This tool will draft a LinkedIn post for you, review it
itself, and iterate until it's publish-ready.

═══════════════════════════════════════════════════════

What topic do you want a LinkedIn post about?
> The rise of agentic AI in 2025

Starting generation...

  Generated Post:
  ...

[Verdict: APPROVED]

═══════════════════════════════════════════════════════
FINAL LINKEDIN POST
═══════════════════════════════════════════════════════
...
Total attempts: 2
Approved: True
```

---

## 🔄 Agent State Machine

The pipeline is powered by a **LangGraph `StateGraph`** with the following state:

```python
class State(TypedDict):
    topic           : str       # Your input topic
    messages        : list      # Conversation history
    draft           : str       # Current post draft
    review_feedback : str       # Reviewer's feedback
    is_approved     : bool      # Approval flag
    attempt         : int       # Iteration counter (max: 3)
```

---

## 🌐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | For Gemini writer LLM |
| `GROQ_API_KEY` | ✅ Yes | For LLaMA reviewer LLM |
| `TAVILY_API_KEY` | ✅ Yes | For web search tool |
| `MISTRAL_API_KEY` | Optional | Reserved for future use |

---

## 🚧 Roadmap

- [ ] 🌐 Add a web UI (Streamlit / FastAPI frontend)
- [ ] 📅 Schedule post generation on a cron
- [ ] 🎨 Support multiple post styles (storytelling, listicle, thought leadership)
- [ ] 🔗 LinkedIn API integration for direct posting
- [ ] 📊 Analytics: track engagement of generated posts
- [ ] 🗂️ Post history & saved drafts

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) & [LangGraph](https://github.com/langchain-ai/langgraph) for the agentic framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) for state-of-the-art content generation
- [Groq](https://groq.com/) for blazing-fast inference
- [Tavily](https://tavily.com/) for real-time web search

---

<p align="center">
  Built with ❤️ by <b>Subham Kumar</b> &nbsp;|&nbsp;
  <a href="https://www.linkedin.com/in/subhamkumar5456">LinkedIn</a>
</p>
