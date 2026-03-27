# 🔬 Agentic AI Research Assistant — Milestone 2

> Intelligent Research Topic Analysis using LangGraph, Groq LLM, RAG, and DuckDuckGo Web Search

## Overview

This is **Milestone 2** of the "Intelligent Research Topic Analysis and Agentic AI Research Assistant" project. It implements a fully agentic research pipeline using LangGraph with 6 nodes, conditional edges, and session-based conversational RAG.

**Key difference from Milestone 1:** Instead of traditional NLP (TF-IDF, LDA, spaCy), Milestone 2 uses an agentic AI approach with an LLM-powered research pipeline that can search the web, process content via RAG, and generate validated research reports.

## Architecture

```
User Question
      ↓
┌─────────────────┐
│  Rephrase Node  │ ← Chat history (conversational RAG)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Search Node    │ ← DuckDuckGo + BeautifulSoup scraping
└────────┬────────┘
         ↓
┌─────────────────┐
│  Retrieve Node  │ ← Chunk → Embed (MiniLM) → FAISS → Top-k
└────────┬────────┘
         ↓
┌─────────────────┐
│  Generate Node  │ ← Groq LLM → Structured report
└────────┬────────┘
         ↓
┌─────────────────┐     NEEDS_MORE_SEARCH
│  Validate Node  │ ────────────────────→ (back to Search, max 2×)
└────────┬────────┘
         ↓ VALID
┌─────────────────┐
│  Report Node    │ → Follow-ups + Topic expansion
└────────┬────────┘
         ↓
       END
```

## Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| LLM | Groq (llama-3.1-8b-instant) | Free tier |
| Web Search | DuckDuckGo (duckduckgo-search) | Free, no API key |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) | Free, runs locally |
| Vector Store | FAISS (faiss-cpu, in-memory) | Free |
| Agent Framework | LangGraph (LangChain) | Free |
| UI | Streamlit | Free |
| PDF Export | fpdf2 | Free |
| Web Scraping | BeautifulSoup4 + requests | Free |

**Total cost: $0** — All tools are free-tier or open-source.

## Project Structure

```
milestone2/
├── app.py                    # Main Streamlit UI (full app with all features)
├── config.py                 # Centralized settings (model names, chunk size, etc.)
├── requirements.txt          # All dependencies with pinned versions
├── README.md                 # This file
├── .streamlit/
│   └── config.toml           # Dark theme configuration
├── agents/
│   ├── __init__.py
│   └── research_agent.py     # LangGraph agent with 6 nodes + conditional edges
└── utils/
    ├── __init__.py
    ├── search.py              # DuckDuckGo web search + BeautifulSoup scraping
    ├── rag.py                 # Chunking, embedding, FAISS vector store, retrieval
    ├── llm.py                 # Groq LLM setup + 5 prompt templates
    └── export.py              # PDF and Markdown export
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- A free Groq API key ([get one here](https://console.groq.com/keys))

### Installation

1. **Clone/navigate to the project:**
   ```bash
   cd milestone2
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # or: .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

5. **Enter your Groq API key** in the sidebar and start researching!

## Features

### Core Features
- ✅ Open-ended research queries (any topic)
- ✅ DuckDuckGo web search + page scraping (no API key needed)
- ✅ RAG pipeline: chunk → embed → FAISS → retrieve
- ✅ LangGraph agent with 6 nodes, conditional edges, explicit state management
- ✅ Structured report: Title, Abstract, Key Findings, Sources (URLs), Conclusion
- ✅ Validation node to reduce hallucinations
- ✅ Error handling with user-friendly messages

### Extensions
- ✅ **Extension 1:** Session-based memory (conversational RAG — resolves pronouns)
- ✅ **Extension 2:** PDF and Markdown export (download buttons)
- ✅ **Extension 3:** Follow-up question generation (clickable)
- ✅ **Extension 4:** Topic expansion suggestions (clickable)

### UI Features
- ✅ Live 6-step progress tracker with status icons
- ✅ Premium dark theme with gradient accents
- ✅ Session history in sidebar
- ✅ Ready for Streamlit Community Cloud deployment

## Deployment (Streamlit Community Cloud)

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file
4. Deploy — users enter their own Groq API key in the sidebar

## Configuration

All settings are in `config.py`:

| Setting | Value | Description |
|---------|-------|-------------|
| GROQ_MODEL | llama-3.1-8b-instant | LLM model |
| GROQ_TEMPERATURE | 0.3 | Controls randomness |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | Local embedding model |
| MAX_SEARCH_RESULTS | 6 | DuckDuckGo results per query |
| CHUNK_SIZE | 500 | Characters per text chunk |
| CHUNK_OVERLAP | 50 | Overlap between chunks |
| TOP_K_RETRIEVAL | 4 | Chunks retrieved per query |
| MAX_SEARCH_ITERATIONS | 2 | Max validation retry loops |
| MAX_CHAT_HISTORY | 10 | Session memory limit |
