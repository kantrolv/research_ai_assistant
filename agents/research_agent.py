"""
agents/research_agent.py — LangGraph-based Agentic Research Assistant

This is the core of Milestone 2. It implements a LangGraph StateGraph with
7 nodes and conditional edges for iterative search refinement + early exit.

Graph Structure:
    [rephrase] → [search] → [retrieve] → (conditional)
                    ↑                          |
                    |              (has docs) → [generate] → [validate] → (conditional)
                    |                                                          |
                    └──── (if NEEDS_MORE_SEARCH, max 2 times) ────────────────┘
                    |                                                          |
                    |                                            (if VALID) → [finalize] → END
                    |
                    └── (no docs) → [early_exit] → END

Nodes:
    1. rephrase_node   — Conversational RAG + spell-check correction
    2. search_node     — DuckDuckGo search with retry (broader query fallback)
    3. retrieve_node   — Chunk → Embed (all-MiniLM-L6-v2) → FAISS → top-k retrieval
    4. generate_node   — Groq LLM generates structured report from retrieved chunks
    5. validate_node   — LLM fact-checks report claims against source chunks
    6. finalize_node   — Generate follow-up questions + expanded topic queries
    7. early_exit_node — Clean error message when search/retrieve returns nothing

API Key Rotation:
    LLM-calling nodes rotate through multiple Groq API keys (round-robin)
    to avoid hitting the 6000 TPM free-tier rate limit.

State (ResearchState):
    Inputs:     question, chat_history, api_keys, key_counter
    Processing: rephrased_question, search_results, chunks, retrieved_docs, search_iterations
    Outputs:    report, validation, follow_up_questions, expanded_queries, sources
    Tracking:   status, error, should_stop
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from utils.search import search_and_scrape
from utils.rag import chunk_documents, build_vectorstore, retrieve_relevant_chunks
from utils.llm import (
    get_llm,
    REPHRASE_PROMPT,
    QUERY_CORRECTION_PROMPT,
    CLARIFY_PROMPT,
    RESEARCH_REPORT_PROMPT,
    VALIDATION_PROMPT,
    FOLLOWUP_AND_EXPAND_PROMPT,
)
from config import MAX_SEARCH_ITERATIONS


# ══════════════════════════════════════════════════════════
# 1. STATE DEFINITION
# ══════════════════════════════════════════════════════════

class ResearchState(TypedDict):
    """
    The state that flows through every node in the LangGraph.
    Each node reads what it needs and returns updated fields.
    """
    # ── Inputs ──
    question: str                       # User's original question
    chat_history: list[dict]            # [{"question": ..., "answer": ...}, ...]
    api_keys: list[str]                 # List of Groq API keys for rotation
    key_counter: int                    # Tracks which key to use next

    # ── Clarification ──
    needs_clarification: bool           # Whether the query needs user confirmation
    corrected_query: str                # The corrected/clarified version of the query
    clarification_reason: str           # Why correction was suggested
    confidence: str                     # "high" or "low" confidence in correction
    user_confirmed: bool                # Set by app.py when user clicks confirm

    # ── Processing ──
    rephrased_question: str             # Self-contained version of the question
    search_results: list[dict]          # [{title, url, snippet, content}, ...]
    chunks: list                        # LangChain Document objects (chunked)
    retrieved_docs: list                # Top-k relevant chunks from FAISS
    search_iterations: int              # Counter to prevent infinite search loops
    should_stop: bool                   # Flag for early exit when no results

    # ── Outputs ──
    report: str                         # The structured research report (markdown)
    validation: str                     # VALID / NEEDS_MORE_SEARCH / ISSUES
    follow_up_questions: list[str]      # 3 suggested follow-up questions
    expanded_queries: list[str]         # 3 expanded topic search queries
    sources: list[dict]                 # [{title, url}, ...] for citation

    # ── Tracking ──
    status: str                         # Current step name (used by UI progress)
    error: str                          # Error message if anything fails


# ══════════════════════════════════════════════════════════
# HELPER: Rotated LLM access
# ══════════════════════════════════════════════════════════

def get_rotated_llm(state: dict):
    """
    Get an LLM instance using the next API key in rotation.

    Returns:
        (llm, new_counter) — the LLM instance and the incremented counter
    """
    keys = state["api_keys"]
    counter = state.get("key_counter", 0)
    key = keys[counter % len(keys)]
    llm = get_llm(key)
    return llm, counter + 1


# ══════════════════════════════════════════════════════════
# 2. NODE FUNCTIONS
# ══════════════════════════════════════════════════════════

def clarify_node(state: ResearchState) -> dict:
    """
    Smart clarification: detect typos/ambiguity and suggest corrections.
    Only flags queries that need clarification — clear queries pass through.
    """
    try:
        question = state["question"]

        llm, new_counter = get_rotated_llm(state)
        chain = CLARIFY_PROMPT | llm
        result = chain.invoke({"question": question})

        response = result.content.strip()

        # Parse the structured response
        needs_clarification = False
        corrected_query = question
        confidence = "high"
        reason = "Query is clear"

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("NEEDS_CLARIFICATION:"):
                needs_clarification = "true" in line.lower()
            elif line.startswith("CORRECTED_QUERY:"):
                corrected_query = line.replace("CORRECTED_QUERY:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                confidence = "low" if "low" in line.lower() else "high"
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        return {
            "needs_clarification": needs_clarification,
            "corrected_query": corrected_query,
            "clarification_reason": reason,
            "confidence": confidence,
            "status": "clarify_done",
            "key_counter": new_counter,
        }
    except Exception as e:
        # If clarification fails, just proceed with original query
        return {
            "needs_clarification": False,
            "corrected_query": state["question"],
            "clarification_reason": "",
            "confidence": "high",
            "status": "clarify_done",
            "key_counter": state.get("key_counter", 0),
        }


def rephrase_node(state: ResearchState) -> dict:
    """
    Node 1: Conversational RAG + Spell-Check.

    Two-step process:
    1. Use corrected_query (from clarify) as base, or fall back to original question
    2. If chat_history exists, rephrase for context (resolve pronouns)
    3. Always run query correction to fix typos
    """
    try:
        # Use corrected query if available (from clarify step), otherwise original
        question = state.get("corrected_query", state["question"])
        chat_history = state.get("chat_history", [])

        # Step 1: Rephrase with chat context (only if history exists)
        if chat_history:
            history_text = "\n".join(
                f"Q: {h['question']}\nA: {h['answer'][:200]}..."
                for h in chat_history[-5:]
            )
            llm, new_counter = get_rotated_llm(state)
            chain = REPHRASE_PROMPT | llm
            result = chain.invoke({
                "chat_history": history_text,
                "question": question,
            })
            question = result.content.strip()
            state = {**state, "key_counter": new_counter}

        # Step 2: Spell-check / query correction (always runs)
        llm, new_counter = get_rotated_llm(state)
        correction_chain = QUERY_CORRECTION_PROMPT | llm
        corrected = correction_chain.invoke({"question": question})
        corrected_text = corrected.content.strip()

        # Only use correction if it looks valid
        if corrected_text and len(corrected_text) < len(question) * 3:
            # Safety: check that correction shares words with original
            original_words = set(question.lower().split())
            corrected_words = set(corrected_text.lower().split())
            if original_words & corrected_words:
                question = corrected_text
            # else: correction went off-track, keep original

        return {
            "rephrased_question": question,
            "key_counter": new_counter,
            "status": "rephrase_done",
        }
    except Exception as e:
        # Graceful fallback: use original question if rephrasing/correction fails
        return {
            "rephrased_question": state.get("corrected_query", state["question"]),
            "status": "rephrase_done",
            "error": f"Rephrase warning: {str(e)}. Using original question.",
        }


def search_node(state: ResearchState) -> dict:
    """
    Node 2: Web Search — Search DuckDuckGo and scrape result pages.

    Includes retry logic:
    1. Try full query
    2. If no results → try first 4 words only
    3. If still nothing → try "what is {first 4 words}"
    """
    try:
        query = state.get("rephrased_question", state["question"])
        results = search_and_scrape(query)

        # Retry 1: Try a simplified/broader query (first 4 words)
        if not results:
            simplified = " ".join(query.split()[:4])
            results = search_and_scrape(simplified)

        # Retry 2: Try with "what is" prefix
        if not results:
            simplified = " ".join(query.split()[:4])
            results = search_and_scrape(f"what is {simplified}")

        # Retry 3: Try the original question (before rephrase/correction)
        if not results:
            original = state.get("question", "")
            if original and original != query:
                results = search_and_scrape(original)

        if not results:
            return {
                "search_results": [],
                "status": "search_done",
                "error": f"No results found for '{query}'. Try rephrasing your question.",
            }

        # Extract source metadata for citations
        sources = [{"title": r["title"], "url": r["url"]} for r in results]

        return {
            "search_results": results,
            "sources": sources,
            "search_iterations": state.get("search_iterations", 0) + 1,
            "status": "search_done",
        }
    except Exception as e:
        return {
            "search_results": [],
            "status": "search_done",
            "error": f"Search failed: {str(e)}",
        }


def retrieve_node(state: ResearchState) -> dict:
    """
    Node 3: RAG Pipeline — Chunk → Embed → FAISS → Retrieve top-k.

    Sets should_stop=True if no content is available, triggering early exit.
    """
    try:
        search_results = state.get("search_results", [])
        if not search_results:
            return {
                "retrieved_docs": [],
                "status": "retrieve_done",
                "should_stop": True,
                "error": "No content to process — search returned empty results.",
            }

        # Step 1: Chunk the scraped content (preserves metadata)
        chunks = chunk_documents(search_results)

        if not chunks:
            return {
                "retrieved_docs": [],
                "status": "retrieve_done",
                "should_stop": True,
                "error": "Could not extract meaningful content from sources.",
            }

        # Step 2: Build FAISS vector store (embed + index)
        vectorstore = build_vectorstore(chunks)

        # Step 3: Retrieve top-k relevant chunks
        query = state.get("rephrased_question", state["question"])
        relevant = retrieve_relevant_chunks(vectorstore, query)

        return {
            "chunks": chunks,
            "retrieved_docs": relevant,
            "should_stop": False,
            "status": "retrieve_done",
        }
    except Exception as e:
        return {
            "retrieved_docs": [],
            "should_stop": True,
            "status": "retrieve_done",
            "error": f"Retrieval failed: {str(e)}",
        }


def early_exit_node(state: ResearchState) -> dict:
    """
    Early Exit Node — Returns a clean error when search/retrieve found nothing.

    No LLM call — just a static error message with helpful tips.
    """
    question = state.get("rephrased_question", state["question"])
    return {
        "report": "",
        "validation": "",
        "follow_up_questions": [],
        "expanded_queries": [],
        "status": "complete",
        "error": (
            f"Could not find enough information about '{question}'. "
            f"Please try:\n"
            f"- Check your spelling\n"
            f"- Use more specific terms\n"
            f"- Try a different question"
        ),
    }


def generate_node(state: ResearchState) -> dict:
    """
    Node 4: Report Generation — Feed retrieved chunks to Groq LLM.

    Token-optimized: Each chunk is trimmed to 200 chars to reduce token usage.
    """
    try:
        retrieved = state.get("retrieved_docs", [])
        if not retrieved:
            return {
                "report": "## Error\n\nInsufficient data to generate a research report. "
                          "The search did not return enough relevant content.",
                "status": "generate_done",
            }

        # Format context with source attribution — TRIMMED to 200 chars per chunk
        context_parts = []
        for i, doc in enumerate(retrieved):
            source_title = doc.metadata.get("title", "Unknown Source")
            source_url = doc.metadata.get("url", "")
            trimmed_content = doc.page_content[:500]
            context_parts.append(
                f"[Source {i+1}: {source_title}]({source_url})\n{trimmed_content}\n"
            )
        context = "\n---\n".join(context_parts)

        llm, new_counter = get_rotated_llm(state)
        chain = RESEARCH_REPORT_PROMPT | llm
        result = chain.invoke({
            "question": state.get("rephrased_question", state["question"]),
            "context": context,
        })

        return {
            "report": result.content.strip(),
            "key_counter": new_counter,
            "status": "generate_done",
        }
    except Exception as e:
        return {
            "report": f"## Error\n\nReport generation failed: {str(e)}",
            "status": "generate_done",
            "error": f"Generation error: {str(e)}",
        }


def validate_node(state: ResearchState) -> dict:
    """
    Node 5: Validation — LLM fact-checks the report against source chunks.

    Token-optimized:
    - Only sends first 500 chars of report
    - Only uses top 3 chunks, each trimmed to 100 chars
    """
    try:
        report = state.get("report", "")
        retrieved = state.get("retrieved_docs", [])

        # Skip validation if there's nothing to validate
        if not report or not retrieved:
            return {"validation": "VALID", "status": "validate_done"}

        # Token-optimized: short snippets from top 3 chunks only
        context = "\n".join(
            f"- {doc.page_content[:100]}"
            for doc in retrieved[:3]
        )

        llm, new_counter = get_rotated_llm(state)
        chain = VALIDATION_PROMPT | llm
        result = chain.invoke({
            "report": report[:500],       # Only first 500 chars of report
            "context": context,
        })

        return {
            "validation": result.content.strip(),
            "key_counter": new_counter,
            "status": "validate_done",
        }
    except Exception as e:
        # If validation fails, proceed anyway with VALID status
        return {
            "validation": "VALID",
            "status": "validate_done",
            "error": f"Validation warning: {str(e)}",
        }


def finalize_node(state: ResearchState) -> dict:
    """
    Node 6: Finalize — Generate follow-up questions AND expanded topics.

    Token-optimized: Uses a SINGLE merged LLM call (was 2 separate calls).
    Only sends report[:300] as summary, not the full report.
    """
    try:
        llm, new_counter = get_rotated_llm(state)
        report = state.get("report", "")
        question = state.get("rephrased_question", state["question"])

        # Single merged LLM call for both follow-ups AND expansion
        chain = FOLLOWUP_AND_EXPAND_PROMPT | llm
        result = chain.invoke({
            "question": question,
            "report_summary": report[:300],   # Only first 300 chars
        })

        # Parse the combined response
        follow_ups, expanded = _parse_followup_and_expand(result.content)

        return {
            "follow_up_questions": follow_ups[:3],
            "expanded_queries": expanded[:3],
            "key_counter": new_counter,
            "status": "complete",
        }
    except Exception as e:
        # Non-critical: follow-ups are nice-to-have
        return {
            "follow_up_questions": [],
            "expanded_queries": [],
            "status": "complete",
            "error": f"Follow-up generation warning: {str(e)}",
        }


def _parse_followup_and_expand(text: str) -> tuple[list[str], list[str]]:
    """
    Parse the merged LLM response that contains both follow-up questions
    and expanded topics, separated by headers.

    Returns:
        (follow_up_questions, expanded_queries)
    """
    follow_ups = []
    expanded = []
    current_section = None

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        upper = line.upper()
        if "FOLLOW" in upper and ("QUESTION" in upper or "UP" in upper):
            current_section = "followup"
            continue
        elif "EXPAND" in upper and ("TOPIC" in upper or "QUER" in upper):
            current_section = "expanded"
            continue

        # Clean the line (remove numbering)
        cleaned = line.lstrip("0123456789.-) ").strip()
        if not cleaned:
            continue

        if current_section == "followup":
            follow_ups.append(cleaned)
        elif current_section == "expanded":
            expanded.append(cleaned)
        elif not current_section and len(follow_ups) < 3:
            # If no header found yet, assume follow-ups first
            follow_ups.append(cleaned)

    # If parsing failed to separate, split evenly
    if not expanded and len(follow_ups) >= 6:
        expanded = follow_ups[3:6]
        follow_ups = follow_ups[:3]

    return follow_ups, expanded


def _parse_numbered_list(text: str) -> list[str]:
    """
    Parse LLM output that contains numbered items (1. xxx, 2. xxx, 3. xxx).
    Strips numbering, bullets, and whitespace. Returns clean list of strings.
    """
    items = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = line.lstrip("0123456789.-) ").strip()
        if cleaned:
            items.append(cleaned)
    return items


# ══════════════════════════════════════════════════════════
# 3. CONDITIONAL EDGES
# ══════════════════════════════════════════════════════════

def should_continue_after_retrieve(state: ResearchState) -> str:
    """
    Conditional edge after retrieve_node.

    Decision logic:
        - If should_stop is True OR no retrieved docs → early_exit
        - Otherwise → continue to generate
    """
    if state.get("should_stop", False) or not state.get("retrieved_docs", []):
        return "early_exit"
    return "generate"


def should_retry_search(state: ResearchState) -> str:
    """
    Conditional edge after validate_node.

    Decision logic:
        - If validation says "NEEDS_MORE_SEARCH" AND iterations < MAX (2) → loop back to search
        - Otherwise → proceed to final report
    """
    validation = state.get("validation", "VALID")
    iterations = state.get("search_iterations", 0)

    if "NEEDS_MORE_SEARCH" in validation and iterations < MAX_SEARCH_ITERATIONS:
        return "search"     # Loop back for more data
    else:
        return "finalize"   # Proceed to final output


# ══════════════════════════════════════════════════════════
# 4. BUILD THE GRAPH
# ══════════════════════════════════════════════════════════

def build_research_graph():
    """
    Construct the LangGraph StateGraph with 7 nodes and conditional edges.

    Flow:
        rephrase → search → retrieve → (conditional)
                                            |
                              (has docs) → generate → validate → (conditional)
                                                                      |
                                                        (VALID) → finalize → END
                                                        (NEEDS_MORE) → search (loop, max 2×)
                              (no docs) → early_exit → END
    """
    graph = StateGraph(ResearchState)

    # Add all 7 nodes
    graph.add_node("rephrase", rephrase_node)
    graph.add_node("search", search_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("early_exit", early_exit_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize", finalize_node)

    # Linear edges
    graph.set_entry_point("rephrase")
    graph.add_edge("rephrase", "search")
    graph.add_edge("search", "retrieve")

    # Conditional edge: retrieve → generate (has docs) OR early_exit (no docs)
    graph.add_conditional_edges(
        "retrieve",
        should_continue_after_retrieve,
        {
            "generate": "generate",
            "early_exit": "early_exit",
        },
    )

    graph.add_edge("generate", "validate")

    # Conditional edge: validate → search (retry) OR finalize (done)
    graph.add_conditional_edges(
        "validate",
        should_retry_search,
        {
            "search": "search",
            "finalize": "finalize",
        },
    )

    # Terminal edges
    graph.add_edge("early_exit", END)
    graph.add_edge("finalize", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════
# 5. RUN THE AGENT
# ══════════════════════════════════════════════════════════

def run_research_agent(
    question: str,
    api_keys: list[str],
    chat_history: list[dict] | None = None,
    status_callback=None,
) -> dict:
    """
    Execute the full research pipeline.

    Args:
        question:        The user's research question
        api_keys:        List of Groq API keys (rotated across LLM calls)
        chat_history:    Previous Q&A pairs for conversational RAG
        status_callback: Optional callable(node_name, status) for progress tracking

    Returns:
        Final state dict with: report, sources, follow_up_questions,
        expanded_queries, validation, error, etc.
    """
    agent = build_research_graph()

    # Initialize the state with all fields
    initial_state = {
        "question": question,
        "api_keys": api_keys,
        "key_counter": 0,
        "chat_history": chat_history or [],
        "needs_clarification": False,
        "corrected_query": question,
        "clarification_reason": "",
        "confidence": "high",
        "user_confirmed": False,
        "rephrased_question": "",
        "search_results": [],
        "chunks": [],
        "retrieved_docs": [],
        "search_iterations": 0,
        "should_stop": False,
        "report": "",
        "validation": "",
        "follow_up_questions": [],
        "expanded_queries": [],
        "sources": [],
        "status": "starting",
        "error": "",
    }

    # Stream the graph execution to get real-time progress updates
    final_state = initial_state.copy()
    for step_output in agent.stream(initial_state):
        # step_output is a dict like {"node_name": {updated_fields}}
        for node_name, node_updates in step_output.items():
            # Merge updates into our tracked state
            final_state.update(node_updates)

            # Notify the UI about progress
            status = node_updates.get("status", node_name)
            if status_callback:
                status_callback(node_name, status)

    return final_state


def run_clarification(question: str, api_keys: list[str]) -> dict:
    """
    Run ONLY the clarification step (fast, single LLM call).
    This runs OUTSIDE the main graph — used by app.py for Phase 1.

    Returns:
        dict with needs_clarification, corrected_query, clarification_reason,
        confidence, status, key_counter
    """
    state = {
        "question": question,
        "api_keys": api_keys,
        "key_counter": 0,
        "needs_clarification": False,
        "corrected_query": question,
        "clarification_reason": "",
        "confidence": "high",
        "status": "starting",
    }

    result = clarify_node(state)
    return result
