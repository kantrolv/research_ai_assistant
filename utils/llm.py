"""
utils/llm.py — Groq LLM setup + all prompt templates

Provides:
    - get_llm(api_key)              → Initialize ChatGroq with llama-3.1-8b-instant
    - REPHRASE_PROMPT               → Rephrase question using chat history (compact)
    - RESEARCH_REPORT_PROMPT        → Generate structured report
    - VALIDATION_PROMPT             → Fact-check report (compact, single-word response)
    - FOLLOWUP_AND_EXPAND_PROMPT    → Combined: 3 follow-ups + 3 expanded queries (1 LLM call)

Token-optimized: All prompts are kept concise to stay under Groq free-tier limits.
"""

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from config import GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS


def get_llm(api_key: str) -> ChatGroq:
    """
    Initialize the Groq LLM client.

    Args:
        api_key: User's Groq API key

    Returns:
        ChatGroq instance configured with llama-3.1-8b-instant
    """
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
    )


# ══════════════════════════════════════════════════════════
# PROMPT 1: REPHRASE (Conversational RAG) — COMPACT
# ══════════════════════════════════════════════════════════

REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the question to be self-contained using the chat history. "
     "Resolve pronouns (it, they, this). If already clear, return as-is. "
    "IMPORTANT: Do NOT correct spelling or change any words. "
    "Do NOT replace technical terms or product names. "
    "Only resolve references from chat history. "
    "Return ONLY the rephrased question."),
    ("human",
     "History:\n{chat_history}\n\nQuestion: {question}\n\nRephrased:"),
])


# ══════════════════════════════════════════════════════════
# PROMPT 2: RESEARCH REPORT GENERATION
# ══════════════════════════════════════════════════════════

RESEARCH_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert research analyst. Using ONLY the provided source material, 
generate a comprehensive, well-structured research report. 

RULES:
- Use ONLY information from the provided sources. Do NOT hallucinate or add external information.
- If sources are insufficient, say so honestly rather than making things up.
- Include specific details, data points, and facts from the sources.
- Cite sources where relevant.

You MUST format your response in this EXACT structure:

## Title
[A clear, descriptive title for the research topic]

## Abstract
[2-3 sentence overview summarizing the key findings and scope of the research]

## Key Findings
- [Finding 1 with specific details and data from sources]
- [Finding 2 with specific details and data from sources]
- [Finding 3 with specific details and data from sources]
- [Finding 4 if available]
- [Finding 5 if available]

## Sources
- [Source 1 title](URL1)
- [Source 2 title](URL2)
- [Source 3 title](URL3)

## Conclusion
[2-3 sentence conclusion summarizing the overall state of research on this topic and potential future directions]"""),
    ("human",
     "Research Question: {question}\n\n"
     "Retrieved Source Material:\n{context}\n\n"
     "Generate the structured research report:"),
])


# ══════════════════════════════════════════════════════════
# PROMPT 3: VALIDATION (Fact-checking) — COMPACT
# ══════════════════════════════════════════════════════════

VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a fact-checker. Check if the report claims are supported by "
     "the source snippets. Respond with ONE word: VALID or NEEDS_MORE_SEARCH. "
     "If issues exist, write: ISSUES: [one sentence]."),
    ("human",
     "Report:\n{report}\n\nSource snippets:\n{context}\n\nVerdict:"),
])


# ══════════════════════════════════════════════════════════
# PROMPT 4: FOLLOW-UP + TOPIC EXPANSION (MERGED — 1 LLM call)
# ══════════════════════════════════════════════════════════

FOLLOWUP_AND_EXPAND_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Based on the research report and question below, generate:\n\n"
     "FOLLOW-UP QUESTIONS (3 questions the user might ask next):\n"
     "1. ...\n2. ...\n3. ...\n\n"
     "EXPANDED TOPICS (3 related search queries for deeper research):\n"
     "1. ...\n2. ...\n3. ...\n\n"
     "Return EXACTLY this format, nothing else."),
    ("human",
     "Question: {question}\n\nReport summary: {report_summary}\n\nGenerate:"),
])


# ══════════════════════════════════════════════════════════
# PROMPT 5: QUERY CORRECTION (Spell-check)
# ══════════════════════════════════════════════════════════

QUERY_CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search query spell-checker for technical queries. "
     "Fix ONLY obvious typos. Do NOT change meaning or intent.\n"
     "RULES:\n"
     "- For tech-looking words, prefer tech terms (langchai→langchain, NOT lingchi)\n"
     "- If unsure, return the query AS-IS\n"
     "- Do NOT add extra words or expand the query\n"
     "Return ONLY the corrected query, nothing else."),
    ("human", "Query: {question}\n\nCorrected:"),
])


# ══════════════════════════════════════════════════════════
# PROMPT 6: QUERY CLARIFICATION (Typo/Ambiguity Detection)
# ══════════════════════════════════════════════════════════

CLARIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a query clarification assistant. Analyze the user's research question and determine:
1. Does it contain likely typos or misspellings?
2. Is it ambiguous (could mean multiple things)?
3. What did the user most likely mean?

Respond in this EXACT format:
NEEDS_CLARIFICATION: true OR false
CORRECTED_QUERY: [the corrected/clarified version of the query]
CONFIDENCE: high OR low
REASON: [one short sentence explaining why correction was needed, or "Query is clear" if no correction needed]

RULES:
- If the query is clear and correctly spelled, set NEEDS_CLARIFICATION: false and return the original query as CORRECTED_QUERY
- If you detect a typo, suggest the most likely intended word
- For technical terms (programming, AI, frameworks), prefer tech interpretations over general words
  Examples: "langchai" → "langchain" (NOT "lingchi"), "faiss" → "faiss" (it's correct), "pytorch" → "pytorch"
- For ambiguous queries, prefer the most common/popular interpretation
- NEVER change the meaning or intent of the query
- NEVER add extra words or expand the query — only fix typos"""),
    ("human", "User query: {question}\n\nAnalysis:"),
])
