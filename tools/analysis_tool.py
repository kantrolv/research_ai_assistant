"""
Analysis Tool — Node 3
Uses the trained ML models (TF-IDF, LDA, KMeans) from Milestone-1
to analyze summarized research content and predict topic + cluster.

Improvements (Milestone-2 Fix):
  - Enhanced keyword extraction with a custom junk-word blocklist so
    generic words like 'million', 'access', 'science' are suppressed.
  - Bigram-aware keyword extraction to surface meaningful research phrases
    like 'graph neural', 'node embedding', 'message passing'.
"""

import pickle
import logging
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from utils.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading (cached at module level so they're only loaded once)
# ---------------------------------------------------------------------------
_vectorizer = None
_lda = None
_kmeans = None
_models_loaded = False

# ---------------------------------------------------------------------------
# Fix 4 — A dedicated bigram TF-IDF vectorizer for keyword extraction
#          (the original vectorizer is still used for topic/cluster prediction)
# ---------------------------------------------------------------------------
_kw_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
)

# Extra blocklist: generic words the built-in stop_words won't catch
EXTRA_STOP_WORDS = {
    "million", "billion", "access", "science", "research", "paper",
    "study", "studies", "journal", "article", "page", "pages",
    "publication", "publications", "available", "download", "free",
    "review", "result", "results", "method", "methods", "approach",
    "based", "using", "used", "new", "also", "one", "two", "three",
    "first", "second", "like", "use", "way", "may", "many",
    "show", "shown", "propose", "proposed", "present", "presents",
    "work", "works", "recent", "recently", "however", "therefore",
    "provide", "provides", "provided", "including", "include",
    "important", "significant", "different", "general", "specific",
    "various", "several", "total", "number", "high", "low",
    "large", "small", "data", "set",
}


def _load_models():
    """Load the trained ML models from disk."""
    global _vectorizer, _lda, _kmeans, _models_loaded

    if _models_loaded:
        return

    try:
        with open("models/tfidf_vectorizer.pkl", "rb") as f:
            _vectorizer = pickle.load(f)
        with open("models/lda_model.pkl", "rb") as f:
            _lda = pickle.load(f)
        with open("models/kmeans_model.pkl", "rb") as f:
            _kmeans = pickle.load(f)
        _models_loaded = True
        logger.info("ML models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")
        _models_loaded = False


def _extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """
    Extract meaningful research keywords/phrases using a dedicated
    bigram-aware TF-IDF vectorizer.

    Fix 4 — Replaces the old approach that only used the trained
    unigram vectorizer, which surfaced generic words.
    """
    try:
        X = _kw_vectorizer.fit_transform([text])
        feature_names = _kw_vectorizer.get_feature_names_out()
        sorted_items = sorted(
            zip(X.tocoo().col, X.tocoo().data),
            key=lambda x: x[1],
            reverse=True,
        )

        keywords: list[str] = []
        for idx, score in sorted_items:
            phrase = feature_names[idx].lower().strip()
            # Skip if too short or is a stop/junk word
            if len(phrase) <= 2:
                continue
            tokens = phrase.split()
            if any(t in ENGLISH_STOP_WORDS or t in EXTRA_STOP_WORDS for t in tokens):
                continue
            keywords.append(phrase)
            if len(keywords) >= top_n:
                break

        return keywords

    except Exception as e:
        logger.warning(f"Keyword extraction failed, falling back: {e}")
        # Fallback: use the trained vectorizer (same as before)
        if _vectorizer is not None:
            X = _vectorizer.transform([text])
            feature_names = _vectorizer.get_feature_names_out()
            sorted_items = sorted(
                zip(X.tocoo().col, X.tocoo().data),
                key=lambda x: x[1],
                reverse=True,
            )
            keywords = []
            for idx, score in sorted_items:
                word = feature_names[idx].lower()
                if word not in ENGLISH_STOP_WORDS and word not in EXTRA_STOP_WORDS and len(word) > 2:
                    keywords.append(word)
                if len(keywords) >= top_n:
                    break
            return keywords
        return []


def analyze_research_content(summarized_results: list[dict]) -> dict:
    """
    Analyze the combined research content using the trained ML pipeline.

    Args:
        summarized_results: List of dicts, each with a 'summary' key.

    Returns:
        Dict with predicted_topic, topic_distribution, predicted_cluster,
        keywords, and per-item analysis.
    """
    _load_models()

    if not _models_loaded:
        return {
            "error": "ML models could not be loaded.",
            "predicted_topic": None,
            "topic_distribution": [],
            "predicted_cluster": None,
            "keywords": [],
            "per_item_analysis": []
        }

    # Combine all summaries into one text block for global analysis
    combined_text = " ".join([r.get("summary", "") for r in summarized_results])
    clean_text = preprocess_text(combined_text)

    try:
        # TF-IDF transform (original trained vectorizer — for topic/cluster)
        X = _vectorizer.transform([clean_text])

        # LDA topic prediction
        topic_dist = _lda.transform(X)[0]
        predicted_topic = int(np.argmax(topic_dist))

        # KMeans cluster prediction
        predicted_cluster = int(_kmeans.predict(X)[0])

        # ---------------------------------------------------------------
        # Fix 4 — Extract keywords with the improved extractor
        # ---------------------------------------------------------------
        keywords = _extract_keywords(clean_text, top_n=8)

        # Per-item analysis (lightweight — predict topic for each snippet)
        per_item = []
        for result in summarized_results:
            item_text = preprocess_text(result.get("summary", ""))
            if item_text.strip():
                Xi = _vectorizer.transform([item_text])
                item_topic = int(np.argmax(_lda.transform(Xi)[0]))
                item_cluster = int(_kmeans.predict(Xi)[0])
            else:
                item_topic = predicted_topic
                item_cluster = predicted_cluster

            per_item.append({
                "title": result.get("title", ""),
                "predicted_topic": item_topic,
                "predicted_cluster": item_cluster
            })

        return {
            "predicted_topic": predicted_topic,
            "topic_distribution": topic_dist.tolist(),
            "predicted_cluster": predicted_cluster,
            "keywords": keywords,
            "per_item_analysis": per_item,
            "error": None
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            "error": str(e),
            "predicted_topic": None,
            "topic_distribution": [],
            "predicted_cluster": None,
            "keywords": [],
            "per_item_analysis": []
        }
