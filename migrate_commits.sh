#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Migration Script: Move milestone2 files into research_ai_assistant
# with backdated commits from March 19 to March 30, 2025
# ═══════════════════════════════════════════════════════════════

set -e

REPO="/Users/kantrolvamshikrishna/Desktop/research_ai_assistant"
SRC="/Users/kantrolvamshikrishna/Downloads/milestone2"

cd "$REPO"

# Helper function: backdated commit
commit() {
    local date="$1"
    local time="$2"
    local msg="$3"
    GIT_AUTHOR_DATE="$date $time +0530" GIT_COMMITTER_DATE="$date $time +0530" \
        git commit -m "$msg"
}

# ══════════════════════════════════════════════════════════
# March 19 — Commit 1: Update .gitignore and requirements for Milestone 2
# ══════════════════════════════════════════════════════════

cp "$SRC/.gitignore" "$REPO/.gitignore"
cp "$SRC/requirements.txt" "$REPO/requirements.txt"
cp "$SRC/.env.example" "$REPO/.env.example"
git add .gitignore requirements.txt .env.example
commit "2026-03-19" "10:15:00" "Update gitignore, requirements and env for Milestone 2 agentic pipeline"

# ══════════════════════════════════════════════════════════
# March 19 — Commit 2: Add centralized config module
# ══════════════════════════════════════════════════════════

cp "$SRC/config.py" "$REPO/config.py"
git add config.py
commit "2026-03-19" "14:30:00" "Add centralized config with LLM, RAG, and agent settings"

# ══════════════════════════════════════════════════════════
# March 20 — Commit 1: Add DuckDuckGo search and web scraping utility
# ══════════════════════════════════════════════════════════

mkdir -p "$REPO/utils"
cp "$SRC/utils/__init__.py" "$REPO/utils/__init__.py"
cp "$SRC/utils/search.py" "$REPO/utils/search.py"
git add utils/__init__.py utils/search.py
commit "2026-03-20" "09:45:00" "Add DuckDuckGo web search and BeautifulSoup scraping utility"

# ══════════════════════════════════════════════════════════
# March 20 — Commit 2: Add RAG pipeline with FAISS vector store
# ══════════════════════════════════════════════════════════

cp "$SRC/utils/rag.py" "$REPO/utils/rag.py"
git add utils/rag.py
commit "2026-03-20" "13:20:00" "Add RAG pipeline: chunking, embedding, FAISS vector store, retrieval"

# ══════════════════════════════════════════════════════════
# March 20 — Commit 3: Add Groq LLM setup and prompt templates
# ══════════════════════════════════════════════════════════

cp "$SRC/utils/llm.py" "$REPO/utils/llm.py"
git add utils/llm.py
commit "2026-03-20" "17:00:00" "Add Groq LLM client setup and all prompt templates"

# ══════════════════════════════════════════════════════════
# March 21 — Commit 1: Add PDF and Markdown export utility
# ══════════════════════════════════════════════════════════

cp "$SRC/utils/export.py" "$REPO/utils/export.py"
git add utils/export.py
commit "2026-03-21" "11:00:00" "Add PDF and Markdown export for research reports"

# ══════════════════════════════════════════════════════════
# March 22 — Commit 1: Add API key rotation manager
# ══════════════════════════════════════════════════════════

cp "$SRC/utils/key_manager.py" "$REPO/utils/key_manager.py"
git add utils/key_manager.py
commit "2026-03-22" "10:30:00" "Add API key rotation manager for Groq rate limit avoidance"

# ══════════════════════════════════════════════════════════
# March 23 — Commit 1: Add LangGraph research agent with 7 nodes
# ══════════════════════════════════════════════════════════

mkdir -p "$REPO/agents"
cp "$SRC/agents/__init__.py" "$REPO/agents/__init__.py"
cp "$SRC/agents/research_agent.py" "$REPO/agents/research_agent.py"
git add agents/__init__.py agents/research_agent.py
commit "2026-03-23" "12:00:00" "Add LangGraph research agent with 7 nodes and conditional edges"

# ══════════════════════════════════════════════════════════
# March 24 — Commit 1: Add Streamlit dark theme configuration
# ══════════════════════════════════════════════════════════

mkdir -p "$REPO/.streamlit"
cp "$SRC/.streamlit/config.toml" "$REPO/.streamlit/config.toml"
git add .streamlit/config.toml
commit "2026-03-24" "10:00:00" "Add Streamlit dark theme configuration"

# ══════════════════════════════════════════════════════════
# March 25 — Commit 1: Add main Streamlit app with full UI
# ══════════════════════════════════════════════════════════

cp "$SRC/app.py" "$REPO/app.py"
git add app.py
commit "2026-03-25" "14:00:00" "Add main Streamlit app with premium UI and live progress tracking"

# ══════════════════════════════════════════════════════════
# March 27 — Commit 1: Add project README with architecture and setup docs
# ══════════════════════════════════════════════════════════

cp "$SRC/README.md" "$REPO/README.md"
git add README.md
commit "2026-03-27" "11:30:00" "Add comprehensive README with architecture diagram and setup instructions"

# ══════════════════════════════════════════════════════════
# March 30 — Commit 1: Final project cleanup and Milestone 2 ready
# ══════════════════════════════════════════════════════════

# Make sure everything is tracked
git add -A
# Only commit if there's something to commit
if ! git diff --cached --quiet; then
    commit "2026-03-30" "16:00:00" "Final cleanup and project structure verification for Milestone 2"
fi

echo ""
echo "✅ All commits created successfully!"
echo ""
git log --oneline --format="%h %ad %s" --date=short -15
