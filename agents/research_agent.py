"""
Research Agent — Fully Autonomous LangGraph Workflow
=====================================================
Implements the complete agentic research pipeline:

  User Query
    → 🧠 Planner (intent + query expansion)
    → 🔍 Search (web retrieval via DuckDuckGo)
    → 🛡️ Validation (clean + filter noisy sources)
    → 🧠 Reasoning (RAG: aggregate findings from context)
    → 📊 Analysis (ML: TF-IDF, LDA, KMeans)
    → 📝 Report (structured markdown generation)
    → 📤 Export (save .md to disk)

Features:
  ✅ Open-ended research query support
  ✅ Web search with academic filtering
  ✅ RAG-based reasoning with hallucination-reduction prompts
  ✅ Explicit LangGraph state management
  ✅ Conditional edges for graceful error recovery
  ✅ Structured report: Title, Abstract, Key Findings, Sources, Conclusion
  ✅ Follow-up question generation (Extension)
  ✅ Markdown export (Extension)
  ✅ Session-based memory via state passthrough (Extension)
"""

from __future__ import annotations

import logging
import os
import re
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END

from tools.search_tool import search_research_papers
from tools.summarize_tool import summarize_search_results
from tools.analysis_tool import analyze_research_content
from utils.explanation_generator import generate_explanation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed State — Explicit state management across all nodes
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """Full state that flows through the LangGraph."""
    query: str
    expanded_queries: List[str]
    search_results: List[Dict[str, str]]
    summarized_results: List[Dict[str, str]]
    explanation: str
    findings: str
    analysis: Dict[str, Any]
    report: str
    follow_up_questions: List[str]
    export_path: str
    error: Optional[str]
    current_step: str
    # Session memory: accumulated across multiple invocations
    session_history: List[Dict[str, str]]


# ---------------------------------------------------------------------------
# Helper: LLM text generation (reusing core model)
# ---------------------------------------------------------------------------

def _llm_generate(prompt: str, max_length: int = 300) -> str:
    """Generate text using the shared flan-t5-large model."""
    try:
        from core.model import generate_text
        return generate_text(prompt, max_length=max_length)
    except Exception as e:
        logger.warning(f"LLM generation failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# NODE 1: Planner — Intent detection + query expansion
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    """Analyze query intent and generate expanded search queries."""
    query = state["query"]
    logger.info(f"[Planner Node] Planning for: {query}")

    try:
        expanded = [
            query,
            f"{query} research paper",
        ]

        return {
            "expanded_queries": expanded,
            "current_step": "planning_complete",
            "error": None,
        }
    except Exception as e:
        logger.error(f"Planner failed: {e}")
        return {
            "expanded_queries": [query],
            "error": None,   # non-fatal
            "current_step": "planning_complete",
        }


# ---------------------------------------------------------------------------
# NODE 2: Search — Web retrieval via DuckDuckGo
# ---------------------------------------------------------------------------

def search_node(state: AgentState) -> dict:
    """Search the web for research papers using expanded queries."""
    queries = state.get("expanded_queries", [state["query"]])
    logger.info(f"[Search Node] Running {len(queries)} queries")

    try:
        all_results: list[dict] = []
        seen_links: set[str] = set()

        for q in queries[:2]:  # limit to 2 queries for speed
            results = search_research_papers(q, max_results=3)
            for r in results:
                link = r.get("link", "")
                if link not in seen_links:
                    seen_links.add(link)
                    all_results.append(r)

        if not all_results:
            return {
                "search_results": [],
                "error": "No search results found. Try rephrasing your query.",
                "current_step": "search_complete",
            }

        # Deduplicate and limit
        return {
            "search_results": all_results[:5],
            "error": None,
            "current_step": "search_complete",
        }
    except Exception as e:
        logger.error(f"Search node failed: {e}")
        return {
            "search_results": [],
            "error": f"Search failed: {str(e)}",
            "current_step": "search_complete",
        }


# ---------------------------------------------------------------------------
# NODE 3: Validation + Summarization — Clean, filter, and summarize
# ---------------------------------------------------------------------------

def validation_node(state: AgentState) -> dict:
    """Validate sources and generate per-source summaries."""
    search_results = state.get("search_results", [])
    logger.info(f"[Validation Node] Processing {len(search_results)} results")

    if not search_results:
        return {
            "summarized_results": [],
            "error": state.get("error") or "No results to validate.",
            "current_step": "validation_complete",
        }

    try:
        # Use the existing summarization tool (includes cleaning + filtering)
        summarized = summarize_search_results(search_results)

        # Extra filter: remove items with very short or empty summaries
        valid = [s for s in summarized if len(s.get("summary", "").strip()) > 20]

        if not valid:
            return {
                "summarized_results": [],
                "error": "All sources were filtered out as low-quality.",
                "current_step": "validation_complete",
            }

        return {
            "summarized_results": valid,
            "error": None,
            "current_step": "validation_complete",
        }
    except Exception as e:
        logger.error(f"Validation node failed: {e}")
        return {
            "summarized_results": [],
            "error": f"Validation/summarization failed: {str(e)}",
            "current_step": "validation_complete",
        }


# ---------------------------------------------------------------------------
# NODE 4: Explanation — Clean conceptual explanation
# ---------------------------------------------------------------------------

def explanation_node(state: AgentState) -> dict:
    """Generate a clean, standalone explanation of the topic."""
    query = state.get("query", "")
    logger.info("[Explanation Node] Generating explanation")

    try:
        explanation = generate_explanation(query)
        return {
            "explanation": explanation,
            "current_step": "explanation_complete",
        }
    except Exception as e:
        logger.error(f"Explanation node failed: {e}")
        return {
            "explanation": f"Could not generate explanation: {str(e)}",
            "current_step": "explanation_complete",
        }


# ---------------------------------------------------------------------------
# NODE 5: Reasoning — RAG across aggregated sources (hallucination-reduced)
# ---------------------------------------------------------------------------

def reasoning_node(state: AgentState) -> dict:
    """Aggregate information across sources and reason to produce findings."""
    query = state.get("query", "")
    sources = state.get("summarized_results", [])
    logger.info(f"[Reasoning Node] Reasoning across {len(sources)} sources")

    try:
        # Build aggregated context
        context_parts = []
        for i, s in enumerate(sources[:5], 1):
            context_parts.append(
                f"Source {i} ({s.get('title', 'Unknown')}): {s.get('summary', '')}"
            )
        context = "\n".join(context_parts)

        # Hallucination-reduction RAG prompt
        findings_prompt = (
            f"You are an expert researcher. Answer the question using ONLY "
            f"the provided context below. Do NOT add any information that is "
            f"not present in the context.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Provide your answer as a numbered list of key findings:"
        )
        findings = _llm_generate(findings_prompt, max_length=500)

        if not findings.strip():
            # Fallback: create findings from summaries directly
            findings = "\n".join(
                [f"• {s.get('summary', '')}" for s in sources[:4]]
            )

        return {
            "findings": findings,
            "error": None,
            "current_step": "reasoning_complete",
        }
    except Exception as e:
        logger.error(f"Reasoning node failed: {e}")
        return {
            "findings": "",
            "error": f"Reasoning failed: {str(e)}",
            "current_step": "reasoning_complete",
        }


# ---------------------------------------------------------------------------
# NODE 6: Analysis — ML structural analysis (TF-IDF, LDA, KMeans)
# ---------------------------------------------------------------------------

def analysis_node(state: AgentState) -> dict:
    """Run ML models on validated research content."""
    summarized = state.get("summarized_results", [])
    logger.info(f"[Analysis Node] Analyzing {len(summarized)} items")

    if not summarized:
        return {
            "analysis": {"error": "No data to analyze."},
            "current_step": "analysis_complete",
        }

    try:
        analysis = analyze_research_content(summarized)
        return {
            "analysis": analysis,
            "current_step": "analysis_complete",
        }
    except Exception as e:
        # Non-fatal: report continues without ML analysis
        logger.warning(f"Analysis node failed (non-fatal): {e}")
        return {
            "analysis": {"error": str(e)},
            "current_step": "analysis_complete",
        }


# ---------------------------------------------------------------------------
# NODE 7: Report — Structured markdown report generation
# ---------------------------------------------------------------------------

def report_node(state: AgentState) -> dict:
    """Generate the full structured research report with all sections."""
    query = state.get("query", "")
    explanation = state.get("explanation", "")
    findings = state.get("findings", "")
    summarized = state.get("summarized_results", [])
    analysis = state.get("analysis", {})

    logger.info("[Report Node] Assembling structured report")

    try:
        lines = []

        # ── Title ──
        lines.append(f"# Research Report: {query.title()}")
        lines.append("")

        # ── Abstract ──
        lines.append("## Abstract")
        if explanation:
            # Use first 3 sentences of explanation as abstract
            sentences = [s.strip() for s in explanation.split(".") if s.strip()]
            abstract = ". ".join(sentences[:3]) + "."
            lines.append(abstract)
        elif summarized:
            snippets = [r.get("summary", "") for r in summarized[:3]]
            lines.append(" ".join(snippets))
        else:
            lines.append("No abstract could be generated.")
        lines.append("")

        # ── Key Findings ──
        lines.append("## Key Findings")
        if findings:
            lines.append(findings)
        elif summarized:
            for i, item in enumerate(summarized, 1):
                lines.append(
                    f"**Finding {i}** ({item.get('title', 'Untitled')}): "
                    f"{item.get('summary', 'N/A')}"
                )
        else:
            lines.append("No findings available.")
        lines.append("")

        # ── Analysis Results ──
        lines.append("## Analysis Results")
        if analysis and not analysis.get("error"):
            topic = analysis.get("predicted_topic")
            cluster = analysis.get("predicted_cluster")
            keywords = analysis.get("keywords", [])
            lines.append(f"- **Predicted Topic:** Topic {topic}")
            lines.append(f"- **Predicted Cluster:** Cluster {cluster}")
            if keywords:
                lines.append(f"- **Key Terms:** {', '.join(keywords)}")

            per_item = analysis.get("per_item_analysis", [])
            if per_item:
                lines.append("")
                lines.append("### Per-Source Analysis")
                for item in per_item:
                    lines.append(
                        f"  - *{item['title']}*: Topic {item['predicted_topic']}, "
                        f"Cluster {item['predicted_cluster']}"
                    )
        else:
            error = analysis.get("error", "Analysis unavailable.")
            lines.append(f"- Analysis could not be completed: {error}")
        lines.append("")

        # ── Sources (URLs) ──
        lines.append("## Sources")
        if summarized:
            for item in summarized:
                link = item.get("link", "")
                title = item.get("title", "Untitled")
                if link:
                    lines.append(f"- [{title}]({link})")
                else:
                    lines.append(f"- {title}")
        else:
            lines.append("- No sources available.")
        lines.append("")

        # ── Conclusion ──
        lines.append("## Conclusion")
        if summarized:
            # Generate LLM conclusion
            conc_prompt = (
                f"Write a brief concluding paragraph for a research report on "
                f"'{query}' based on these findings:\n{findings[:500]}"
            )
            conclusion = _llm_generate(conc_prompt, max_length=200)
            if not conclusion.strip():
                kw = analysis.get("keywords", [])
                kw_str = ", ".join(kw[:4]) if kw else "the queried topics"
                conclusion = (
                    f"Based on analysis of {len(summarized)} research sources, "
                    f"the key themes revolve around **{kw_str}**. "
                    f"Further investigation is recommended for deeper insights."
                )
            lines.append(conclusion)
        else:
            lines.append("Insufficient data to generate a conclusion.")
        lines.append("")

        report = "\n".join(lines)

        # ── Follow-up Questions (Extension) ──
        followup_prompt = (
            f"Based on the research topic '{query}', suggest 3 follow-up "
            f"research questions that would deepen understanding. "
            f"Return each question on a new line."
        )
        followup_text = _llm_generate(followup_prompt, max_length=200)
        followup_questions = [
            q.strip().lstrip("0123456789.-) ") 
            for q in followup_text.split("\n") 
            if q.strip() and len(q.strip()) > 10
        ][:3]

        # ── Markdown Export (Extension) ──
        export_path = ""
        try:
            os.makedirs("exports", exist_ok=True)
            safe_name = re.sub(r'[^a-zA-Z0-9_\-\s]', '', query).strip().replace(" ", "_")
            export_path = f"exports/{safe_name[:50]}_report.md"
            with open(export_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report exported to {export_path}")
        except Exception as ex:
            logger.warning(f"Export failed: {ex}")

        return {
            "report": report,
            "follow_up_questions": followup_questions,
            "export_path": export_path,
            "error": None,
            "current_step": "report_complete",
        }

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {
            "report": "",
            "follow_up_questions": [],
            "export_path": "",
            "error": f"Report generation failed: {str(e)}",
            "current_step": "report_complete",
        }


# ---------------------------------------------------------------------------
# Conditional Edge Logic (multi-step control flow)
# ---------------------------------------------------------------------------

def route_after_search(state: AgentState) -> str:
    """If search failed, short-circuit to END."""
    if state.get("error"):
        return "END"
    return "validate"


def route_after_validation(state: AgentState) -> str:
    """If no valid sources, still generate explanation but skip reasoning."""
    if state.get("error") or not state.get("summarized_results"):
        return "explain"
    return "explain"


def route_after_reasoning(state: AgentState) -> str:
    """Always proceed to analysis (non-fatal errors handled inside)."""
    return "analyze"


# ---------------------------------------------------------------------------
# Build the LangGraph Workflow
# ---------------------------------------------------------------------------

def build_research_agent():
    """
    Construct the LangGraph StateGraph:

        planner → search →(conditional)→ validate →(conditional)→ explain
                                                                     ↓
                               END ←── report ←── analyze ←── reason
    """
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("plan", planner_node)
    workflow.add_node("search", search_node)
    workflow.add_node("validate", validation_node)
    workflow.add_node("explain", explanation_node)
    workflow.add_node("reason", reasoning_node)
    workflow.add_node("analyze", analysis_node)
    workflow.add_node("report", report_node)

    # Entry point
    workflow.set_entry_point("plan")

    # Edges
    workflow.add_edge("plan", "search")

    # Conditional: if search fails → END
    workflow.add_conditional_edges(
        "search",
        route_after_search,
        {"validate": "validate", "END": END},
    )

    # After validation → always go to explain
    workflow.add_edge("validate", "explain")

    # Linear flow: explain → reason → analyze → report → END
    workflow.add_edge("explain", "reason")
    workflow.add_edge("reason", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_research_agent(query: str, session_history: list = None) -> dict:
    """
    High-level entry: run the full autonomous research workflow.

    Args:
        query: The user's open-ended research question.
        session_history: Optional list of previous queries for session memory.

    Returns:
        The final AgentState dict with report, analysis, sources, etc.
    """
    if not query or not query.strip():
        return {
            "query": query,
            "search_results": [],
            "summarized_results": [],
            "analysis": {},
            "report": "",
            "explanation": "",
            "findings": "",
            "follow_up_questions": [],
            "export_path": "",
            "error": "Please enter a valid research query.",
            "current_step": "error",
            "session_history": session_history or [],
        }

    agent = build_research_agent()

    initial_state: AgentState = {
        "query": query.strip(),
        "expanded_queries": [],
        "search_results": [],
        "summarized_results": [],
        "explanation": "",
        "findings": "",
        "analysis": {},
        "report": "",
        "follow_up_questions": [],
        "export_path": "",
        "error": None,
        "current_step": "initialized",
        "session_history": session_history or [],
    }

    try:
        final_state = agent.invoke(initial_state)
        # Update session history
        history = final_state.get("session_history", [])
        history.append({
            "query": query.strip(),
            "report_preview": (final_state.get("report", ""))[:200],
        })
        final_state["session_history"] = history
        return final_state
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {
            "query": query,
            "search_results": [],
            "summarized_results": [],
            "analysis": {},
            "report": "",
            "explanation": "",
            "findings": "",
            "follow_up_questions": [],
            "export_path": "",
            "error": f"Agent workflow failed: {str(e)}",
            "current_step": "error",
            "session_history": session_history or [],
        }
