"""
Search Tool — Node 1
Uses DuckDuckGo (via the `ddgs` package) to search for research papers.
Returns top 5 results with title, link, and snippet.

Improvements (Milestone-2 Fix):
  - Uses `ddgs` package (successor to duckduckgo-search)
  - Prioritizes academic sources (arxiv, semanticscholar, openreview)
  - Banned-domain filter to remove login pages, forums, directories
  - Fetches extra results to compensate for filtered-out junk
"""

from ddgs import DDGS
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fix 2 — Banned domains / URL patterns that return junk, not papers
# ---------------------------------------------------------------------------
BANNED_PATTERNS = [
    "fantbox",
    "news",
    "login",
    "api",
    "documentation",
    "semantic scholar",
    "random blog",
    "researchgate.net",
    "signup",
    "sign-up",
    "sign_up",
    "forum",
    "zhihu.com",
    "quora.com",
    "reddit.com",
    "medium.com",
    "slideshare.net",
    "youtube.com",
    "facebook.com",
    "twitter.com",
    "linkedin.com",
    "pinterest.com",
    "instagram.com",
    "wikipedia.org",
    "powerdrill.ai",
    "amazon.com",
    "ebay.com",
    "blog.",
    "marketing",
]

def build_search_query(query):
    return f"{query} explanation research paper site:arxiv.org OR site:britannica.com OR site:edu"

def is_valid_source(url: str) -> bool:
    """Return False if the URL matches a banned pattern."""
    url_lower = url.lower()
    return not any(pattern in url_lower for pattern in BANNED_PATTERNS)

def search_research_papers(query: str, max_results: int = 5) -> list[dict]:
    """
    Search DuckDuckGo for research papers related to the query.

    Strategy:
      1. Primary search — targets academic sources (arxiv, semanticscholar)
      2. If too few results, fallback to a broader research-oriented search
      3. All results are filtered through `is_valid_source`

    Args:
        query: The user's research question.
        max_results: Maximum number of valid results to return (default 5).

    Returns:
        List of dicts with keys: title, link, snippet
    """
    results: list[dict] = []
    seen_links: set[str] = set()

    # Fetch slightly more than needed so we have enough after filtering
    fetch_count = max_results * 2

    # -----------------------------------------------------------------------
    # Fix 1 — Academic-targeted query
    # -----------------------------------------------------------------------
    academic_query = build_search_query(query)

    try:
        raw_results = DDGS().text(academic_query, max_results=fetch_count)

        for item in raw_results:
            link = item.get("href", "")
            if link in seen_links:
                continue
            if not is_valid_source(link):
                continue
            seen_links.add(link)
            results.append({
                "title": item.get("title", "Untitled"),
                "link": link,
                "snippet": item.get("body", "No description available.")
            })
            if len(results) >= max_results:
                break

        logger.info(
            f"Academic search returned {len(results)} valid results "
            f"for: {query}"
        )

    except Exception as e:
        logger.warning(f"Academic search failed: {e}")

    # -----------------------------------------------------------------------
    # Fallback — broader search if academic query didn't yield enough
    # -----------------------------------------------------------------------
    if len(results) < max_results:
        fallback_query = f"{query} research paper abstract findings"
        try:
            raw_fallback = DDGS().text(fallback_query, max_results=fetch_count)

            for item in raw_fallback:
                link = item.get("href", "")
                if link in seen_links:
                    continue
                if not is_valid_source(link):
                    continue
                seen_links.add(link)
                results.append({
                    "title": item.get("title", "Untitled"),
                    "link": link,
                    "snippet": item.get("body", "No description available.")
                })
                if len(results) >= max_results:
                    break

            logger.info(
                f"Fallback search brought total to {len(results)} results"
            )

        except Exception as e:
            logger.warning(f"Fallback search also failed: {e}")

    # -----------------------------------------------------------------------
    # Edge case — no results at all
    # -----------------------------------------------------------------------
    if not results:
        logger.error(f"No valid results found for: {query}")
        results.append({
            "title": "Search Unavailable",
            "link": "",
            "snippet": (
                "Could not retrieve relevant research results. "
                "Please try rephrasing your query."
            )
        })

    return results
