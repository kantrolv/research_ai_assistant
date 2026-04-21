"""
app.py — Agentic AI Research Assistant (Milestone 2)
Main Streamlit application with full UI.

Features:
    - Open-ended research queries on any topic
    - LangGraph agent with 6 nodes + live progress tracking
    - API key rotation (up to 3 Groq keys) to avoid rate limits
    - Auto-loads keys from .env file (zero user input needed)
    - Conversational RAG (session-based memory)
    - Structured report: Title, Abstract, Key Findings, Sources, Conclusion
    - PDF & Markdown export
    - Follow-up question generation (clickable)
    - Topic expansion suggestions (clickable)
    - Validation details display
    - Ready for Streamlit Community Cloud deployment
"""

import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Ensure local packages resolve first on hosted environments.
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

try:
    from agents.research_agent import run_research_agent
except Exception:
    import importlib
    run_research_agent = importlib.import_module("agents.research_agent").run_research_agent

from utils.export import export_pdf, export_markdown
from config import MAX_CHAT_HISTORY


# ══════════════════════════════════════════════════════════
# API KEY LOADING (from .env or Streamlit secrets)
# ══════════════════════════════════════════════════════════

def get_api_keys() -> list[str]:
    """
    Load Groq API keys from .env file first, then Streamlit secrets as fallback.
    Supports up to 3 keys for rotation.
    """
    load_dotenv(override=True)  # Force reload .env on every call
    keys = []
    for i in range(1, 4):
        key = os.getenv(f"GROQ_API_KEY_{i}", "")
        # Only check st.secrets if .env didn't have the key
        # (avoids "No secrets found" warnings on local dev)
        if not key:
            try:
                if hasattr(st, "secrets") and len(st.secrets) > 0:
                    key = st.secrets.get(f"GROQ_API_KEY_{i}", "")
            except Exception:
                pass
        if key and key.strip():
            keys.append(key.strip())
    return keys


# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════
# CUSTOM CSS — Premium dark theme with gradient accents
# ══════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #00d2ff 0%, #7b2ff7 50%, #ff6ec7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #8892b0;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* ── Progress Steps ── */
    .step-container {
        display: flex;
        justify-content: space-between;
        padding: 1rem 0;
        margin: 1rem 0;
        gap: 0.5rem;
    }
    .step-item {
        text-align: center;
        flex: 1;
        padding: 0.7rem 0.4rem;
        border-radius: 10px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .step-active {
        background: rgba(123, 47, 247, 0.2);
        border: 1px solid #7b2ff7;
        box-shadow: 0 0 20px rgba(123, 47, 247, 0.15);
    }
    .step-done {
        background: rgba(0, 210, 255, 0.12);
        border: 1px solid rgba(0, 210, 255, 0.4);
    }
    .step-pending {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .step-icon {
        font-size: 1.6rem;
        display: block;
        margin-bottom: 0.2rem;
    }
    .step-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8892b0;
        font-weight: 500;
    }

    /* ── Report Card ── */
    .report-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }

    /* ── Source Links ── */
    .source-item {
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #00d2ff;
        border-radius: 0 8px 8px 0;
        transition: background 0.2s ease;
    }
    .source-item:hover {
        background: rgba(0, 210, 255, 0.08);
    }
    .source-item a {
        color: #64d8ff;
        text-decoration: none;
        font-weight: 500;
    }
    .source-item a:hover {
        color: #00d2ff;
        text-decoration: underline;
    }

    /* ── Chat History ── */
    .history-item {
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ── Section Dividers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.5rem 0 0.8rem;
    }
    .section-header h3 {
        margin: 0;
        font-weight: 600;
    }

    /* ── Validation Badge ── */
    .validation-valid {
        background: rgba(0, 210, 100, 0.15);
        border: 1px solid rgba(0, 210, 100, 0.3);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #00d264;
        font-weight: 500;
    }
    .validation-issues {
        background: rgba(255, 165, 0, 0.15);
        border: 1px solid rgba(255, 165, 0, 0.3);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        color: #ffa500;
        font-weight: 500;
    }

    /* ── Key rotation info ── */
    .key-plan {
        background: rgba(123, 47, 247, 0.1);
        border: 1px solid rgba(123, 47, 247, 0.2);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        font-size: 0.8rem;
        color: #c4b5fd;
        margin-top: 0.5rem;
    }

    /* ── Hide Streamlit defaults ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # [{"question": ..., "answer": ...}, ...]
if "current_report" not in st.session_state:
    st.session_state.current_report = None      # Full agent result state
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""         # For follow-up/expansion queries


# ══════════════════════════════════════════════════════════
# SIDEBAR — API Keys (auto-loaded), Session History, How It Works
# ══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Configuration")

    # Auto-load keys from .env
    api_keys = get_api_keys()

    if api_keys:
        num = len(api_keys)
        if num == 1:
            st.warning(f"{num} API key loaded - may hit rate limits")
        elif num == 2:
            st.info(f"{num} API keys loaded - good speed!")
        else:
            st.success(f"{num} API keys loaded - maximum speed!")

        # Show key rotation plan
        key_labels = ["Key A", "Key B", "Key C"]
        steps = ["Rephrase", "Generate", "Validate", "Finalize"]
        rotation_plan = " → ".join(
            f"{steps[i]}({key_labels[i % num]})"
            for i in range(len(steps))
        )
        st.markdown(
            f"<div class='key-plan'>Rotation plan: {rotation_plan}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.error("No API keys found!")
        st.code(
            "GROQ_API_KEY_1=gsk_your_key_here\n"
            "GROQ_API_KEY_2=gsk_your_key_here\n"
            "GROQ_API_KEY_3=gsk_your_key_here",
            language="bash",
        )
        st.caption("Create a `.env` file in the project root with the keys above")
        st.markdown("[Get free Groq API key](https://console.groq.com/keys)")

    st.divider()

    # ── Session History (Conversational RAG memory) ──
    st.markdown("### Session History")

    if st.session_state.chat_history:
        for i, h in enumerate(st.session_state.chat_history):
            q_preview = h["question"][:50]
            with st.expander(f"Q{i+1}: {q_preview}...", expanded=False):
                st.markdown(f"**Question:** {h['question']}")
                answer_preview = h["answer"][:300] if h["answer"] else "No answer"
                st.markdown(f"**Answer:** {answer_preview}...")

        if st.button("Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_report = None
            st.session_state.pending_query = ""
            st.rerun()
    else:
        st.caption("No research history yet. Ask a question to begin!")

    st.divider()

    # ── How It Works ──
    st.markdown("### How It Works")
    st.caption(
        "1. You ask a research question\n\n"
        "2. The agent rephrases it using chat context\n\n"
        "3. It searches the web via DuckDuckGo\n\n"
        "4. It chunks and embeds content for RAG\n\n"
        "5. The LLM generates a structured report\n\n"
        "6. The report is validated against sources\n\n"
        "7. Follow-up questions are suggested"
    )

    st.divider()
    st.caption("Built with LangGraph • Groq • FAISS • Streamlit")


# ══════════════════════════════════════════════════════════
# MAIN AREA — Header
# ══════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>Agentic AI Research Assistant</h1>
    <p>Ask any research question — the agent will search, analyze, and generate a structured report</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# RESEARCH INPUT
# ══════════════════════════════════════════════════════════

# Use pending_query to auto-populate from follow-up/expansion clicks
default_query = st.session_state.pending_query
if default_query:
    st.session_state.research_input = default_query
    st.session_state.pending_query = ""  # Clear after use

col_input, col_btn = st.columns([5, 1])
with col_input:
    user_question = st.text_input(
        "Research Question",
        placeholder="e.g., What are recent breakthroughs in quantum computing?",
        label_visibility="collapsed",
        key="research_input",
    )
with col_btn:
    run_clicked = st.button("Research", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════
# PROGRESS TRACKER
# ══════════════════════════════════════════════════════════

STEPS = [
    ("rephrase",  "1", "Rephrase"),
    ("search",    "2", "Search"),
    ("retrieve",  "3", "Retrieve"),
    ("generate",  "4", "Generate"),
    ("validate",  "5", "Validate"),
    ("finalize",  "6", "Report"),
]


def render_progress(current_step: str, completed_steps: set):
    """Render the 6-step progress bar with real-time status."""
    cols = st.columns(len(STEPS))
    for i, (name, icon, label) in enumerate(STEPS):
        with cols[i]:
            if name in completed_steps:
                st.markdown(
                    f"<div class='step-item step-done'>"
                    f"<span class='step-icon'>Done</span>"
                    f"<div class='step-label'>{label}</div></div>",
                    unsafe_allow_html=True,
                )
            elif name == current_step:
                st.markdown(
                    f"<div class='step-item step-active'>"
                    f"<span class='step-icon'>Now</span>"
                    f"<div class='step-label'>{label}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='step-item step-pending'>"
                    f"<span class='step-icon'>{icon}</span>"
                    f"<div class='step-label'>{label}</div></div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════
# AGENT EXECUTION
# ══════════════════════════════════════════════════════════

if run_clicked and user_question and api_keys:

    # Placeholders for live progress
    progress_placeholder = st.empty()
    status_text = st.empty()

    completed_steps = set()

    # Status labels for each step
    step_labels = {
        "rephrase":  "Rephrasing question with conversation context...",
        "search":    "Searching the web with DuckDuckGo...",
        "retrieve":  "Chunking, embedding, and retrieving for RAG...",
        "generate":  "Generating a structured research report...",
        "validate":  "Validating claims against source material...",
        "finalize":  "Preparing the final report and follow-up questions...",
    }

    def progress_callback(node_name: str, status: str):
        """Called by the agent after each node completes."""
        with progress_placeholder.container():
            render_progress(node_name, completed_steps)
        status_text.info(step_labels.get(node_name, f"Processing {node_name}..."))

        if "_done" in status or status == "complete":
            completed_steps.add(node_name)

    # Show initial progress
    with progress_placeholder.container():
        render_progress("rephrase", set())
    status_text.info("Starting research pipeline...")

    try:
        # Run the LangGraph agent with real-time progress + key rotation
        result = run_research_agent(
            question=user_question,
            api_keys=api_keys,
            chat_history=st.session_state.chat_history,
            status_callback=progress_callback,
        )

        # Save result
        st.session_state.current_report = result

        # Add to chat history (session memory for conversational RAG)
        report_text = result.get("report", "")
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": report_text,
        })

        # Trim history if too long
        if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_CHAT_HISTORY:]

        # Show completion
        completed_steps = {"rephrase", "search", "retrieve", "generate", "validate", "finalize"}
        with progress_placeholder.container():
            render_progress("finalize", completed_steps)
        status_text.success("Research complete!")

    except Exception as e:
        status_text.error(f"Error: {str(e)}")
        st.stop()

elif run_clicked and not api_keys:
    st.error("No API keys found. Add your Groq keys to the .env file.")

elif run_clicked and not user_question:
    st.warning("Please enter a research question.")


# ══════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ══════════════════════════════════════════════════════════

result = st.session_state.current_report

if result:
    error = result.get("error", "")
    report = result.get("report", "")

    # ── Case 1: Error with no report (search failed / early exit) ──
    if error and not report:
        st.markdown("---")
        st.error(error)
        st.info("Tips: Check spelling, use more specific terms, or try a different question.")

    # ── Case 2: Successful report ──
    elif report:
        # Show any warnings from the pipeline
        if error:
            st.warning(error)

        # ── Research Report ──
        st.markdown("---")
        st.markdown("### Research Report")
        st.markdown(
            f'<div class="report-card">{""}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(report)

        # ── Export Buttons (only if report has real content) ──
        if "## Title" in report or "## Key Findings" in report or "## Abstract" in report:
            st.markdown("---")
            col_pdf, col_md, col_spacer = st.columns([1, 1, 3])

            with col_pdf:
                try:
                    pdf_path = export_pdf(report)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "Download PDF",
                            data=f.read(),
                            file_name="research_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"PDF export failed: {e}")

            with col_md:
                try:
                    md_path = export_markdown(report)
                    with open(md_path, "r") as f:
                        st.download_button(
                            "Download Markdown",
                            data=f.read(),
                            file_name="research_report.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"Markdown export failed: {e}")

        # ── Sources ──
        sources = result.get("sources", [])
        if sources:
            st.markdown("---")
            st.markdown("### Sources")
            for src in sources:
                title = src.get("title", "Untitled")
                url = src.get("url", "#")
                st.markdown(
                    f"<div class='source-item'>"
                    f"<a href='{url}' target='_blank'>{title}</a>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── Follow-up Questions (only if report was successful) ──
        follow_ups = result.get("follow_up_questions", [])
        if follow_ups:
            st.markdown("---")
            st.markdown("### Follow-up Questions")
            st.caption("Click any question to research it next:")
            for i, q in enumerate(follow_ups):
                if st.button(q, key=f"followup_{i}", use_container_width=True):
                    st.session_state.pending_query = q
                    st.rerun()

        # ── Topic Expansion (only if report was successful) ──
        expanded = result.get("expanded_queries", [])
        if expanded:
            st.markdown("---")
            st.markdown("### Explore Related Topics")
            st.caption("Broaden your research into related areas:")
            for i, q in enumerate(expanded):
                if st.button(q, key=f"expand_{i}", use_container_width=True):
                    st.session_state.pending_query = q
                    st.rerun()

        # ── Validation Details ──
        validation = result.get("validation", "")
        if validation:
            st.markdown("---")
            with st.expander("Validation Details", expanded=False):
                if "VALID" in validation and "NEEDS" not in validation:
                    st.markdown(
                        f"<div class='validation-valid'>Valid: {validation}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='validation-issues'>Needs attention: {validation}</div>",
                        unsafe_allow_html=True,
                    )
