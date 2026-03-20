"""
utils/search.py — DuckDuckGo web search + BeautifulSoup scraping

Provides free web search (no API key required) and page content extraction.
Used by the search_node in the LangGraph agent.

Functions:
    - web_search(query)       → Search DuckDuckGo, return top results
    - scrape_page(url)        → Fetch and extract clean text from a URL
    - search_and_scrape(query)→ Full pipeline: search → scrape → return enriched results
"""

from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from config import MAX_SEARCH_RESULTS, MAX_CONTENT_LENGTH


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """
    Search DuckDuckGo and return top results.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of dicts with keys: title, url, snippet
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        print(f"[Search Error] {e}")
        return []


def scrape_page(url: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """
    Fetch a web page and extract its main text content.
    Strips scripts, styles, nav, footer, and header elements.

    Args:
        url: The URL to scrape
        max_length: Maximum characters to return (truncates after this)

    Returns:
        Clean text content from the page, or empty string on failure
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Extract and clean text
        text = soup.get_text(separator=" ", strip=True)

        # Collapse multiple whitespace
        text = " ".join(text.split())

        return text[:max_length]
    except Exception as e:
        print(f"[Scrape Error] {url}: {e}")
        return ""


def search_and_scrape(query: str) -> list[dict]:
    """
    Full pipeline: search DuckDuckGo → scrape each result page → return enriched list.

    Args:
        query: The search query

    Returns:
        List of dicts with keys: title, url, snippet, content
        Only includes results where scraping was successful.
    """
    results = web_search(query)
    enriched = []

    for r in results:
        content = scrape_page(r["url"])
        if content and len(content.strip()) > 50:
            enriched.append({
                "title": r["title"],
                "url": r["url"],
                "snippet": r["snippet"],
                "content": content,
            })

    return enriched
