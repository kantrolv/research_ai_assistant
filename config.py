"""
config.py — Centralized configuration for the Agentic AI Research Assistant

All tunable parameters live here so nothing is hardcoded across modules.
Change these values to experiment with different settings.
"""

# ══════════════════════════════════════════════════════════
# GROQ LLM SETTINGS
# ══════════════════════════════════════════════════════════
GROQ_MODEL = "llama-3.1-8b-instant"      # Free-tier, fast inference
GROQ_TEMPERATURE = 0.3                     # Lower = more deterministic
GROQ_MAX_TOKENS = 4096                     # Max output tokens per call

# ══════════════════════════════════════════════════════════
# EMBEDDING SETTINGS
# ══════════════════════════════════════════════════════════
EMBEDDING_MODEL = "all-MiniLM-L6-v2"      # Runs locally, no API key needed

# ══════════════════════════════════════════════════════════
# SEARCH SETTINGS
# ══════════════════════════════════════════════════════════
MAX_SEARCH_RESULTS = 6                     # DuckDuckGo results per query
MAX_CONTENT_LENGTH = 3000                  # Max chars to scrape per page

# ══════════════════════════════════════════════════════════
# RAG SETTINGS (token-optimized)
# ══════════════════════════════════════════════════════════
CHUNK_SIZE = 350                           # Characters per text chunk (was 500)
CHUNK_OVERLAP = 70                         # Overlap between chunks (was 50)
TOP_K_RETRIEVAL = 5                        # Number of chunks to retrieve (was 4)

# ══════════════════════════════════════════════════════════
# AGENT SETTINGS
# ══════════════════════════════════════════════════════════
MAX_SEARCH_ITERATIONS = 2                  # Max search→validate loops
MAX_CHAT_HISTORY = 10                      # Max Q&A pairs in session memory

# ══════════════════════════════════════════════════════════
# REPORT STRUCTURE
# ══════════════════════════════════════════════════════════
REPORT_SECTIONS = ["Title", "Abstract", "Key Findings", "Sources", "Conclusion"]
