"""
Summarization Tool — Node 2
Summarizes each search result snippet using the existing TextRank summarizer
from Milestone-1. Falls back to a simple truncation if summarization fails.

Improvements (Milestone-2 Fix):
  - Snippets are cleaned before summarization to remove marketing text,
    URLs, login prompts, and other non-research noise.
"""

import re
import logging
from utils.summarizer import generate_summary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fix 3 — Snippet cleaning: strip junk text before summarization
# ---------------------------------------------------------------------------

# Phrases that appear in junk snippets (case-insensitive removal)
JUNK_PHRASES = [
    r"login",
    r"sign\s*up",
    r"sign\s*in",
    r"create\s+account",
    r"researchgate",
    r"access\s+\d+\+?\s*million",
    r"join\s+for\s+free",
    r"request\s+full[\s-]*text",
    r"download\s+full[\s-]*text",
    r"read\s+full[\s-]*text",
    r"discover\s+the\s+world",
    r"publication\s+pages?",
    r"cookie\s+policy",
    r"privacy\s+policy",
    r"terms\s+of\s+service",
    r"accept\s+cookies",
    r"we\s+use\s+cookies",
    r"subscribe",
    r"buy\s+now",
    r"limited\s+time",
]

# Compile a single regex for efficiency
_junk_re = re.compile("|".join(JUNK_PHRASES), flags=re.IGNORECASE)


def clean_snippet(text: str) -> str:
    """
    Remove URLs, marketing phrases, and other non-research noise from a
    search snippet so downstream summarization gets cleaner input.
    """
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove junk phrases
    text = _junk_re.sub("", text)
    
    # Remove all non-alphanumeric characters except basic punctuation
    text = re.sub(r"[^a-zA-Z0-9., ]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def summarize_search_results(search_results: list[dict]) -> list[dict]:
    """
    Takes a list of search results and produces a short summary for each.

    Args:
        search_results: List of dicts with keys title, link, snippet.

    Returns:
        List of dicts with an added 'summary' key.
    """
    summarized = []

    for result in search_results:
        raw_snippet = result.get("snippet", "")
        title = result.get("title", "Untitled")
        link = result.get("link", "")

        # --- Apply snippet cleaning before summarization ---
        snippet = clean_snippet(raw_snippet)

        try:
            # Use the existing Milestone-1 summarizer (TextRank)
            if len(snippet.split()) > 15:
                summary = generate_summary(snippet, sentences_count=2)
            else:
                # Snippet is already short enough — use as-is
                summary = snippet

            # Fallback if summarizer returns empty
            if not summary or summary.strip() == "":
                summary = snippet[:300]

        except Exception as e:
            logger.warning(f"Summarization failed for '{title}': {e}")
            summary = snippet[:300] if snippet else "No summary available."

        summarized.append({
            "title": title,
            "link": link,
            "snippet": snippet,       # cleaned version
            "summary": summary
        })

    logger.info(f"Summarized {len(summarized)} results.")
    return summarized
