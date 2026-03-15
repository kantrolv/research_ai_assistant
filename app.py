import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
import traceback
from io import StringIO
import html
import time
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Utilities
from utils.preprocessing import preprocess_text
from utils.summarizer import generate_summary
from utils.semantic_search import get_similar_papers, generate_topic_map, get_query_embedding

st.set_page_config(
    page_title="Agentic AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# ─── Premium Light-Theme CSS ───
st.html('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">')

_css = """
<style>
/* ===================== GLOBAL ===================== */

body, .stApp {
    background: #f7f8fc !important;
    font-family: 'Inter', 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #1e293b;
}

/* subtle dot-grid background */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image: radial-gradient(circle, #d4d8e8 1px, transparent 1px);
    background-size: 28px 28px;
    opacity: 0.35;
    pointer-events: none;
    z-index: 0;
}

/* ===================== SCROLLBAR ===================== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #a5b4fc; }

/* ===================== TYPOGRAPHY ===================== */
h1, h2, h3, h4 { color: #0f172a !important; font-family: 'Inter', sans-serif !important; letter-spacing: -0.02em; }

/* ===================== HEADER ===================== */
.header-container {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 36px 36px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
}
.header-container::after {
    content: '';
    position: absolute; top: 0; right: 0;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(79,70,229,0.06), transparent 70%);
    border-radius: 50%;
    transform: translate(30%, -30%);
    pointer-events: none;
}
.app-title {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 40%, #0d9488 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    font-size: 2.8rem;
    line-height: 1.15;
    margin: 0 0 4px 0;
    letter-spacing: -0.03em;
}
.app-subtitle {
    color: #334155;
    font-size: 1.15rem;
    font-weight: 600;
    margin: 0 0 6px 0;
    letter-spacing: -0.01em;
}
.app-desc {
    font-size: 0.95rem;
    color: #64748b;
    margin: 0;
    line-height: 1.5;
}
.status-pills {
    display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap;
}
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #ffffff; border: 1px solid #e2e8f0;
    padding: 6px 14px; border-radius: 9999px;
    font-size: 0.8rem; font-weight: 600; color: #475569;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.status-pill .dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block;
}
.dot-green { background: #22c55e; box-shadow: 0 0 6px rgba(34,197,94,0.4); }
.dot-blue  { background: #6366f1; box-shadow: 0 0 6px rgba(99,102,241,0.4); }

/* ===================== CARDS ===================== */
.glass-wrapper {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
    transition: box-shadow 0.25s ease, transform 0.25s ease;
    position: relative;
}
.glass-wrapper:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
    transform: translateY(-2px);
}
.glass-wrapper h3 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    margin-bottom: 16px !important;
    display: flex; align-items: center; gap: 8px;
}

/* ── Card accent variants ── */
.card-accent-indigo { border-top: 3px solid #6366f1; }
.card-accent-teal   { border-top: 3px solid #14b8a6; }
.card-accent-amber  { border-top: 3px solid #f59e0b; }
.card-accent-rose   { border-top: 3px solid #f43f5e; }
.card-accent-purple { border-top: 3px solid #8b5cf6; }

/* ===================== KEYWORD BADGES ===================== */
.keyword-badge {
    display: inline-flex; align-items: center;
    background: linear-gradient(135deg, #eef2ff, #e0e7ff);
    color: #4338ca;
    padding: 7px 16px;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid #c7d2fe;
    transition: all 0.2s ease;
    letter-spacing: 0.01em;
}
.keyword-badge:hover {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff;
    border-color: #6366f1;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99,102,241,0.25);
}
.badge-container {
    display: flex; flex-wrap: wrap;
    gap: 8px; margin-top: 12px;
}

/* ===================== METRIC HIGHLIGHT ===================== */
.metric-highlight {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 8px 0;
    letter-spacing: -0.03em;
}

/* ===================== SUMMARY CALLOUT ===================== */
.summary-container {
    background: #f8fafc;
    border-left: 4px solid #4f46e5;
    padding: 20px 24px;
    border-radius: 8px;
    font-size: 1rem;
    color: #334155;
    font-weight: 500;
    line-height: 1.7;
    margin-bottom: 16px;
}

/* ===================== SIDEBAR ===================== */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    letter-spacing: -0.02em;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8 !important;
    margin-top: 24px !important;
    margin-bottom: 8px !important;
}
.sidebar-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0 20px 0;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 12px;
}
.sidebar-brand img { border-radius: 12px; }
.sidebar-brand-text {
    font-weight: 800; font-size: 1.1rem;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ===================== TABS ===================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f1f5f9;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab-list"] button {
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #64748b !important;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background: #ffffff !important;
    color: #4f46e5 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem;
    font-weight: 600;
}

/* ===================== SEPARATOR ===================== */
hr {
    margin: 1.5rem 0;
    border: 0;
    border-top: 1px solid #f1f5f9;
}

/* ===================== BUTTONS ===================== */
.stButton button[kind="primary"],
.stButton button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.18) !important;
}
.stButton button[kind="primary"]:hover,
.stButton button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 4px 16px rgba(79,70,229,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton button[kind="secondary"],
.stButton button[data-testid="stBaseButton-secondary"] {
    background: #f8fafc !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton button[kind="secondary"]:hover,
.stButton button[data-testid="stBaseButton-secondary"]:hover {
    background: #f1f5f9 !important;
    border-color: #cbd5e1 !important;
}

/* ===================== TEXT INPUT ===================== */
.stTextInput div div input,
.stTextArea div div textarea {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    color: #1e293b !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput div div input:focus,
.stTextArea div div textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* ===================== FILE UPLOADER ===================== */
[data-testid="stFileUploader"] {
    border: 2px dashed #d4d8e8 !important;
    border-radius: 12px !important;
    padding: 8px !important;
    transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366f1 !important;
}

/* ===================== CHAT ===================== */
[data-testid="stChatMessage"] {
    background: #f8fafc !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 8px !important;
}

/* ===================== METRICS ===================== */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetric"] label {
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8 !important;
    font-weight: 700 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 800 !important;
    color: #1e293b !important;
}

/* ===================== DATAFRAME ===================== */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
}

/* ===================== SUCCESS / INFO / WARNING ===================== */
.stAlert {
    border-radius: 10px !important;
    font-size: 0.9rem !important;
}

/* ===================== WELCOME CARD ===================== */
.welcome-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 50px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
}
.welcome-card::before {
    content: '';
    position: absolute; top: -30px; right: -30px;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(79,70,229,0.05), transparent 70%);
    border-radius: 50%;
}
.welcome-card::after {
    content: '';
    position: absolute; bottom: -40px; left: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(20,184,166,0.05), transparent 70%);
    border-radius: 50%;
}
.welcome-title {
    color: #1e293b; font-size: 1.6rem; font-weight: 800;
    margin-bottom: 12px; letter-spacing: -0.02em;
}
.welcome-text {
    color: #64748b; font-size: 1rem; line-height: 1.6; max-width: 560px; margin: 0 auto;
}

/* ===================== SOURCE CARD ===================== */
.source-card {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.source-card:hover {
    border-color: #c7d2fe;
    background: #fafaff;
}

/* ===================== ANIMATION ===================== */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.glass-wrapper { animation: fadeInUp 0.4s ease forwards; }
.header-container { animation: fadeInUp 0.5s ease forwards; }

</style>
"""
st.markdown(_css, unsafe_allow_html=True)

# LOAD MODELS
@st.cache_resource
def load_models():
    """Cache models so Streamlit doesn’t reload them every run"""
    import pickle
    import os
    
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
        
    with open("models/lda_model.pkl", "rb") as f:
        lda = pickle.load(f)
        
    with open("models/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
        
    return vectorizer, lda, kmeans

try:
    vectorizer, lda, kmeans = load_models()
    models_loaded = True
    error_msg = ""
except Exception as e:
    vectorizer, lda, kmeans = None, None, None
    models_loaded = False
    error_msg = str(e)
    st.error("Failed to load models.")
    st.exception(e)

# Session State for Analysis Persistence & Chat
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# Session State for Agent Research (Milestone-2)
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "agent_query" not in st.session_state:
    st.session_state.agent_query = ""
if "agent_session_history" not in st.session_state:
    st.session_state.agent_session_history = []

# --- Header Section ---
model_dot = "dot-green" if models_loaded else "dot-blue"
model_label = "Models Loaded" if models_loaded else "Models Unavailable"
st.markdown(f"""
<div class="header-container">
    <div class="app-title">Agentic AI Research Assistant</div>
    <div class="app-subtitle">AI-powered research paper analysis &amp; insight engine</div>
    <div class="app-desc">Upload or paste research text to discover topics, clusters, summaries, and semantic insights — powered by TF-IDF, LDA, KMeans, and sentence embeddings.</div>
    <div class="status-pills">
        <span class="status-pill"><span class="dot {model_dot}"></span>{model_label}</span>
        <span class="status-pill"><span class="dot dot-green"></span>NLP Pipeline Ready</span>
        <span class="status-pill"><span class="dot dot-blue"></span>Agent Available</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Control Panel ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <img src="https://cdn-icons-png.flaticon.com/512/2103/2103285.png" width="36" height="36" />
        <span class="sidebar-brand-text">Research AI</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📥 Input Options")
    uploaded_file = st.file_uploader("Upload .txt or .pdf file", type=["txt", "pdf"])
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".pdf"):
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text() or ""
            st.session_state.input_text = pdf_text.strip()
        else:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            st.session_state.input_text = stringio.read()
        
    text_input = st.text_area("Or paste research abstract:", value=st.session_state.input_text, height=200, placeholder="Enter your text here...")

    if text_input != st.session_state.last_text:
        st.session_state.last_text = text_input
        st.session_state.analyzed = False
    
    st.markdown("### ⚙️ Controls")
    analyze_btn = st.button("🚀 Analyze Research Paper", use_container_width=True, type="primary")
    
    if analyze_btn and text_input:
        st.session_state.analyzed = True

    clear_btn = st.button("🔄 Reset", use_container_width=True)
    if clear_btn:
        st.session_state.clear()
        st.rerun()
        
    st.markdown("### 📊 Model Info")
    if models_loaded:
        st.success("All models loaded successfully")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric(label="Topics", value=lda.n_components)
        with col_m2:
            st.metric(label="Clusters", value=kmeans.n_clusters)
    else:
        st.error(f"Model Error: {error_msg}")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 8px 0; color: #94a3b8; font-size: 0.75rem;">
        Built with Streamlit · NLP Pipeline v2.0
    </div>
    """, unsafe_allow_html=True)

# --- Helper Functions for Features ---
def extract_keywords(text, vectorizer, top_n=10):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    tfidf_matrix = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = sorted(zip(tfidf_matrix.tocoo().col, tfidf_matrix.tocoo().data), key=lambda x: x[1], reverse=True)
    
    keywords = []
    for idx, score in sorted_items:
        word = feature_names[idx].lower()
        if word not in ENGLISH_STOP_WORDS and len(word) > 2:
            keywords.append(word)
        if len(keywords) == top_n:
            break
            
    return keywords

def generate_wordcloud(text):
    wc = WordCloud(
        width=900, height=450,
        background_color='#ffffff',
        colormap='cool',
        mode='RGB',
        max_words=80,
        contour_width=0,
        prefer_horizontal=0.85
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def generate_topic_network(topic_dist):
    G = nx.Graph()
    main_topic = topic_dist.argmax()
    
    # Add predicted topic as central node
    G.add_node(f"Predicted\nTopic {main_topic}", size=3500, color='#6366f1')
    
    for i, prob in enumerate(topic_dist):
        if i != main_topic and prob > 0.05:
            G.add_node(f"Topic {i}", size=1500, color='#a5b4fc')
            G.add_edge(f"Predicted\nTopic {main_topic}", f"Topic {i}", weight=prob*10)
            
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    pos = nx.spring_layout(G, seed=42)
    
    node_colors = [nx.get_node_attributes(G, 'color').get(node, '#c7d2fe') for node in G.nodes()]
    node_sizes = [nx.get_node_attributes(G, 'size').get(node, 1000) for node in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes,
            font_size=10, font_weight='bold', font_color='#1e293b', edge_color='#e2e8f0',
            width=2, edgecolors='#ffffff', linewidths=2)
            
    ax.axis('off')
    plt.tight_layout(pad=0.5)
    return fig

# --- Main Logic Trigger ---
if st.session_state.analyzed:
    # We only need to run the heavy AI process if we don't already have results for this exact text
    if not text_input.strip():
        st.warning("Please provide a research abstract to analyze.")
        st.session_state.analyzed = False
    elif not models_loaded:
        st.error("Models failed to load. Please check your model files.")
        st.session_state.analyzed = False
    elif "clean_text" not in st.session_state.analysis_results or st.session_state.analysis_results.get("text") != text_input:
        with st.spinner("Analyzing research paper..."):
            try:
                # 1. Preprocessing
                clean_text = preprocess_text(text_input)
                X = vectorizer.transform([clean_text])
                
                # 2. Extract Features
                topic_dist = lda.transform(X)[0]
                predicted_topic = topic_dist.argmax()
                
                predicted_cluster = kmeans.predict(X)[0]
                
                # 3. Text Summarization & Keyword Extraction
                summary = generate_summary(text_input, sentences_count=3)
                keywords = extract_keywords(clean_text, vectorizer, top_n=6)
                
                # 4. Similar Papers (Semantic Search Feature 4)
                try:
                    similar_papers = get_similar_papers(text_input, top_n=5)
                except Exception as eval_e:
                    similar_papers = []
                
                # 5. Query Embedding for UMAP Topic Map (Feature 5)
                try:
                    query_embedding = get_query_embedding(text_input)
                except Exception as e:
                    query_embedding = np.array([])
                
                # 6. Save to Session State
                st.session_state.analysis_results = {
                    "text": text_input,
                    "clean_text": clean_text,
                    "summary": summary,
                    "topic": predicted_topic,
                    "topic_dist": topic_dist,
                    "cluster": predicted_cluster,
                    "keywords": keywords,
                    "similar_papers": similar_papers,
                    "query_embedding": query_embedding
                }
                
                st.session_state.input_text = text_input
                st.session_state.messages = []
                st.success("Analysis complete")
                
            except Exception as e:
                st.error("🚨 An error occurred during analysis.")
                st.code(traceback.format_exc())
                st.session_state.analyzed = False

# --- Dashboard Layout ---
if st.session_state.analyzed:
    res = st.session_state.analysis_results
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📝 Summary & Tags", "🎯 Topics & Clusters", "📊 Visualizations", "🔍 Semantic Search", "💬 Chat Assistant", "🤖 Agent Research"])
    
    # Tab 1: AI Summary & Keyword Tags & Explain Like I'm 10
    with tab1:
        st.markdown('<div class="glass-wrapper card-accent-teal">', unsafe_allow_html=True)
        st.markdown("<h3>📝 AI-Generated Summary</h3>", unsafe_allow_html=True)
        st.markdown(f'<div class="summary-container">{res["summary"]}</div>', unsafe_allow_html=True)
        
        # Feature 6: "Explain Like I'm 10" Button
        if st.button("🧒 Explain This Paper Simply"):
            kws = res['keywords']
            kw1 = kws[0] if len(kws) > 0 else "something"
            kw2 = kws[1] if len(kws) > 1 else "something else"
            simple_explanation = f"This paper is basically about **{kw1}** and **{kw2}**. It proposes a method to break large problems into smaller parts so computers can analyze them more efficiently, making the technology smarter and faster!"
            st.success(f"👶 **So, basically...**\n\n{simple_explanation}")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-wrapper card-accent-indigo">', unsafe_allow_html=True)
        st.markdown("<h3>🔑 Extracted Keywords</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; font-size:0.88rem; margin-bottom:12px;'>Top keywords extracted using TF-IDF vectorization</p>", unsafe_allow_html=True)
        tags_html = "".join([f'<span class="keyword-badge">{kw}</span>' for kw in res['keywords']])
        st.markdown(f'<div class="badge-container">{tags_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 2: Topic Detection & Cluster Prediction
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="glass-wrapper card-accent-indigo" style="text-align: center;">', unsafe_allow_html=True)
            st.markdown("<h3 style='justify-content:center;'>🎯 Predicted Topic</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-highlight'>Topic {res['topic']}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#64748b; font-size:0.85rem;'>Confidence: {res['topic_dist'][res['topic']]:.1%}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="glass-wrapper card-accent-purple" style="text-align: center;">', unsafe_allow_html=True)
            st.metric("🧩 Cluster Category", f"Cluster {res['cluster']}", "Semantic Assignment")
            st.markdown("<p style='color: #64748b; font-size: 0.85rem; margin-top: 12px;'>Semantic grouping based on KMeans clustering</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('<div class="glass-wrapper">', unsafe_allow_html=True)
        st.markdown("<h3>📊 Topic Probability Distribution</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; font-size:0.88rem; margin-bottom:8px;'>Probability distribution across all LDA topics</p>", unsafe_allow_html=True)
        df_topics = pd.DataFrame({
            "Probability": res['topic_dist'],
            "Topic": [f"Topic {i}" for i in range(len(res['topic_dist']))]
        }).set_index("Topic")
        st.bar_chart(df_topics, color="#6366f1")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Tab 3: Word Cloud & Networkx & Topic Map
    with tab3:
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown('<div class="glass-wrapper card-accent-amber">', unsafe_allow_html=True)
            st.markdown("<h3>☁️ Word Cloud</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color:#64748b; font-size:0.85rem; margin-bottom:8px;'>Most frequent terms weighted by TF-IDF scores</p>", unsafe_allow_html=True)
            fig_wc = generate_wordcloud(res['clean_text'])
            st.pyplot(fig_wc)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_v2:
            st.markdown('<div class="glass-wrapper card-accent-indigo">', unsafe_allow_html=True)
            st.markdown("<h3>🕸️ Topic Network</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color:#64748b; font-size:0.85rem; margin-bottom:8px;'>Relationship graph between detected topics</p>", unsafe_allow_html=True)
            fig_net = generate_topic_network(res['topic_dist'])
            st.pyplot(fig_net)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature 5: Research Topic Map
        st.markdown('<div class="glass-wrapper card-accent-teal">', unsafe_allow_html=True)
        st.markdown("<h3>🗺️ Research Topic Map</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; font-size:0.88rem; margin-bottom:8px;'>2D UMAP projection — your paper is marked with a ★ star relative to the corpus database</p>", unsafe_allow_html=True)
        if hasattr(res['query_embedding'], 'size') and res['query_embedding'].size > 0:
            fig_umap = generate_topic_map(res['query_embedding'])
            st.pyplot(fig_umap)
        else:
            st.info("Embedding map could not be generated. Please ensure SentenceTransformers is loaded properly.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Tab 4: Semantic Search & Similarity Explorer
    with tab4:
        st.markdown('<div class="glass-wrapper card-accent-indigo">', unsafe_allow_html=True)
        st.markdown("<h3>🔍 Semantic Similarity Search</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; font-size:0.88rem; margin-bottom:12px;'>Find papers with similar topics using sentence embeddings</p>", unsafe_allow_html=True)
        
        search_query = st.text_input("Search query:", placeholder="Enter keywords or paste an abstract to find similar papers...")
        
        if search_query:
            st.markdown(f"**Showing results for:** *{search_query}*")
            search_results = get_similar_papers(search_query, top_n=5)
            for idx, p in enumerate(search_results):
                sim_pct = f"{p['similarity']:.0%}"
                st.markdown(f"""
                <div class="source-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <strong style="color:#1e293b;">{p['title']}</strong>
                        <span class="keyword-badge" style="font-size:0.75rem; padding:4px 10px;">Match: {sim_pct}</span>
                    </div>
                    <p style="color:#475569; font-size:0.9rem; line-height:1.5; margin:0;">{p['abstract'][:250]}...</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("#### Top 5 Similar Papers")
            if len(res['similar_papers']) > 0:
                for idx, p in enumerate(res['similar_papers']):
                    sim_pct = f"{p['similarity']:.0%}"
                    st.markdown(f"""
                    <div class="source-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <strong style="color:#1e293b;">{p['title']}</strong>
                            <span class="keyword-badge" style="font-size:0.75rem; padding:4px 10px;">Match: {sim_pct}</span>
                        </div>
                        <p style="color:#475569; font-size:0.9rem; line-height:1.5; margin:0;">{p['abstract'][:250]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No similar papers found in the database.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 5: ChatGPT-Style Chat
    with tab5:
        st.markdown('<div class="glass-wrapper card-accent-teal">', unsafe_allow_html=True)
        st.markdown("<h3>💬 Research Assistant Chat</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b; font-size:0.88rem; margin-bottom:12px;'>Ask questions about the analyzed paper — topics, keywords, summaries, and more</p>", unsafe_allow_html=True)
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Using standard chat input within a container format 
        prompt = st.chat_input("Ask a question about the research paper (e.g., 'What is this paper about?', 'Explain the main idea.')")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
                
            q = prompt.lower()
            response = ""
            if "about" in q or "main idea" in q or "explain" in q or "summary" in q:
                kw_str = " and ".join(res['keywords'][:2]) if len(res['keywords']) > 0 else "the text"
                response = f"**Here is a summary based on the paper:**\n\n> {res['summary']}\n\nThe main idea revolves around {kw_str}."
            elif "topic" in q or "area" in q or "research" in q or "field" in q:
                response = f"This paper is most closely related to **Topic {res['topic']}** (with a probability of {res['topic_dist'][res['topic']]:.2%}) and resides in **Cluster {res['cluster']}**."
            elif "keyword" in q or "term" in q:
                response = "**The primary keywords extracted using TF-IDF are:**\n\n" + ", ".join(f"`{k}`" for k in res['keywords'])
            else:
                response = "I am an AI specialized in analyzing your research abstract. I can answer questions regarding its main idea, topics, and keywords."

            with st.chat_message("assistant"):
                st.markdown(response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 6: Agentic AI Research (Milestone-2 — Fully Autonomous)
    with tab6:
        st.markdown('<div class="glass-wrapper card-accent-purple">', unsafe_allow_html=True)
        st.markdown("<h3>🤖 Autonomous AI Research Agent</h3>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#64748b; font-size:0.88rem;'>Enter an open-ended research question. "
            "The agent will autonomously search, validate, reason, analyze, and generate a full "
            "structured report with follow-up questions and markdown export.</p>",
            unsafe_allow_html=True
        )

        agent_query = st.text_input(
            "Research Question:",
            value=st.session_state.agent_query,
            placeholder="e.g. What are the latest advances in Graph Neural Networks?",
            key="agent_query_input"
        )

        run_agent_btn = st.button("🚀 Run AI Research Agent", use_container_width=True, type="primary", key="run_agent")

        if run_agent_btn and agent_query.strip():
            st.session_state.agent_query = agent_query.strip()

            from agents.research_agent import run_research_agent

            # Pass session history for memory
            history = st.session_state.get("agent_session_history", [])

            with st.spinner("🔍 Agent is researching… Planning → Searching → Validating → Reasoning → Analyzing → Reporting"):
                start_time = time.time()
                result = run_research_agent(agent_query.strip(), session_history=history)
                elapsed = time.time() - start_time

            st.session_state.agent_result = result
            st.session_state.agent_session_history = result.get("session_history", [])

            if result.get("error"):
                st.error(f"⚠️ {result['error']}")
            else:
                st.success(f"✅ Research complete in {elapsed:.1f}s")

        elif run_agent_btn and not agent_query.strip():
            st.warning("Please enter a research question.")

        st.markdown('</div>', unsafe_allow_html=True)

        # ---- Display Agent Results ----
        if st.session_state.agent_result and not st.session_state.agent_result.get("error"):
            agent_res = st.session_state.agent_result

            atab1, atab2, atab3, atab4, atab5 = st.tabs([
                "💡 Explanation", "📄 Full Report", "📊 Analysis", "🔗 Sources", "🔮 Follow-Up"
            ])

            with atab1:
                st.markdown('<div class="glass-wrapper card-accent-indigo">', unsafe_allow_html=True)
                st.subheader("💡 Explanation")
                st.markdown(agent_res.get("explanation", "No explanation generated."))
                st.markdown('</div>', unsafe_allow_html=True)

                # Key Findings section
                findings = agent_res.get("findings", "")
                if findings:
                    st.markdown('<div class="glass-wrapper card-accent-teal">', unsafe_allow_html=True)
                    st.markdown("<h3>🔬 Key Findings (RAG)</h3>", unsafe_allow_html=True)
                    st.markdown(findings)
                    st.markdown('</div>', unsafe_allow_html=True)

            with atab2:
                st.markdown('<div class="glass-wrapper card-accent-teal">', unsafe_allow_html=True)
                st.markdown("<h3>📄 Structured Research Report</h3>", unsafe_allow_html=True)
                report_text = agent_res.get("report", "No report generated.")
                st.markdown(report_text)
                st.markdown('</div>', unsafe_allow_html=True)

                # Markdown Download Button
                if report_text:
                    st.download_button(
                        label="📥 Download Report as Markdown",
                        data=report_text,
                        file_name=f"research_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                    export_path = agent_res.get("export_path", "")
                    if export_path:
                        st.caption(f"Auto-saved to: `{export_path}`")

            with atab3:
                analysis = agent_res.get("analysis", {})
                if analysis and not analysis.get("error"):
                    st.markdown('<div class="glass-wrapper">', unsafe_allow_html=True)
                    st.markdown("<h3>📊 ML Analysis Results</h3>", unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("🎯 Predicted Topic", f"Topic {analysis.get('predicted_topic', '?')}")
                    with col_b:
                        st.metric("🧩 Predicted Cluster", f"Cluster {analysis.get('predicted_cluster', '?')}")

                    keywords = analysis.get("keywords", [])
                    if keywords:
                        kw_html = "".join([f'<span class="keyword-badge">{kw}</span>' for kw in keywords])
                        st.markdown(f'<div class="badge-container">{kw_html}</div>', unsafe_allow_html=True)

                    topic_dist = analysis.get("topic_distribution", [])
                    if topic_dist:
                        df_t = pd.DataFrame({
                            "Probability": topic_dist,
                            "Topic": [f"Topic {i}" for i in range(len(topic_dist))]
                        }).set_index("Topic")
                        st.bar_chart(df_t, color="#8b5cf6")

                    per_item = analysis.get("per_item_analysis", [])
                    if per_item:
                        st.markdown("#### Per-Source Analysis")
                        df_per = pd.DataFrame(per_item)
                        st.dataframe(df_per, use_container_width=True, hide_index=True)

                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("ML analysis could not be generated for this query.")

            with atab4:
                sources = agent_res.get("summarized_results", [])
                if sources:
                    st.markdown('<div class="glass-wrapper card-accent-amber">', unsafe_allow_html=True)
                    st.markdown("<h3>🔗 Sources & References</h3>", unsafe_allow_html=True)
                    for src in sources:
                        title = src.get("title", "Untitled")
                        link = src.get("link", "")
                        snippet = src.get("snippet", "")
                        if link:
                            st.markdown(f"**[{title}]({link})**")
                        else:
                            st.markdown(f"**{title}**")
                        st.caption(snippet[:200])
                        st.divider()
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No sources available.")

            with atab5:
                st.markdown('<div class="glass-wrapper card-accent-rose">', unsafe_allow_html=True)
                st.markdown("<h3>🔮 Follow-Up Research Questions</h3>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='color:#64748b; font-size:0.88rem;'>AI-generated questions to deepen your research. "
                    "Click any question to start a new investigation.</p>",
                    unsafe_allow_html=True
                )
                followups = agent_res.get("follow_up_questions", [])
                if followups:
                    for i, fq in enumerate(followups):
                        if st.button(f"🔎 {fq}", key=f"followup_{i}", use_container_width=True):
                            st.session_state.agent_query = fq
                            st.rerun()
                else:
                    st.info("No follow-up questions were generated.")
                st.markdown('</div>', unsafe_allow_html=True)

                # Session History
                session_hist = st.session_state.get("agent_session_history", [])
                if len(session_hist) > 1:
                    st.markdown('<div class="glass-wrapper">', unsafe_allow_html=True)
                    st.markdown("<h3>🕐 Session History</h3>", unsafe_allow_html=True)
                    for idx, entry in enumerate(reversed(session_hist), 1):
                        st.markdown(f"**{idx}.** {entry.get('query', 'N/A')}")
                    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Landing state
    st.markdown("""
        <div class="welcome-card">
            <div style="font-size: 3rem; margin-bottom: 16px;">🔬</div>
            <div class="welcome-title">Welcome to the Research Dashboard</div>
            <div class="welcome-text">
                Upload a research paper or paste an abstract in the sidebar, then click
                <span style="color:#4f46e5; font-weight:700;">Analyze Research Paper</span>
                to generate AI-powered insights, topic models, and semantic analysis.
            </div>
            <div style="margin-top:20px; display:flex; gap:12px; justify-content:center; flex-wrap:wrap;">
                <span class="status-pill"><span class="dot dot-green"></span>Topic Modeling</span>
                <span class="status-pill"><span class="dot dot-blue"></span>Clustering</span>
                <span class="status-pill"><span class="dot dot-green"></span>Summarization</span>
                <span class="status-pill"><span class="dot dot-blue"></span>Semantic Search</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Agent Research — also accessible from landing page
    st.markdown("---")
    st.markdown('<div class="glass-wrapper card-accent-purple">', unsafe_allow_html=True)
    st.markdown("<h3>🤖 Autonomous AI Research Agent</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b; font-size:0.88rem;'>Or skip the upload — enter a research question and let the AI agent "
        "autonomously search, validate, reason, analyze, and generate a full report.</p>",
        unsafe_allow_html=True
    )

    agent_query_landing = st.text_input(
        "Research Question:",
        value=st.session_state.agent_query,
        placeholder="e.g. What are the latest advances in Graph Neural Networks?",
        key="agent_query_landing"
    )

    run_agent_landing = st.button("🚀 Run AI Research Agent", use_container_width=True, type="primary", key="run_agent_landing")

    if run_agent_landing and agent_query_landing.strip():
        st.session_state.agent_query = agent_query_landing.strip()

        from agents.research_agent import run_research_agent

        history = st.session_state.get("agent_session_history", [])

        with st.spinner("🔍 Agent is researching… Planning → Searching → Validating → Reasoning → Analyzing → Reporting"):
            start_time = time.time()
            result = run_research_agent(agent_query_landing.strip(), session_history=history)
            elapsed = time.time() - start_time

        st.session_state.agent_result = result
        st.session_state.agent_session_history = result.get("session_history", [])

        if result.get("error"):
            st.error(f"⚠️ {result['error']}")
        else:
            st.success(f"✅ Research complete in {elapsed:.1f}s")

    elif run_agent_landing and not agent_query_landing.strip():
        st.warning("Please enter a research question.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Display agent results on landing page too
    if st.session_state.agent_result and not st.session_state.agent_result.get("error"):
        agent_res = st.session_state.agent_result

        lt1, lt2, lt3, lt4, lt5 = st.tabs([
            "💡 Explanation", "📄 Full Report", "📊 Analysis", "🔗 Sources", "🔮 Follow-Up"
        ])

        with lt1:
            st.markdown('<div class="glass-wrapper card-accent-indigo">', unsafe_allow_html=True)
            st.subheader("💡 Explanation")
            st.markdown(agent_res.get("explanation", "No explanation generated."))
            st.markdown('</div>', unsafe_allow_html=True)

            findings = agent_res.get("findings", "")
            if findings:
                st.markdown('<div class="glass-wrapper card-accent-teal">', unsafe_allow_html=True)
                st.markdown("<h3>🔬 Key Findings (RAG)</h3>", unsafe_allow_html=True)
                st.markdown(findings)
                st.markdown('</div>', unsafe_allow_html=True)

        with lt2:
            st.markdown('<div class="glass-wrapper card-accent-teal">', unsafe_allow_html=True)
            st.markdown("<h3>📄 Structured Research Report</h3>", unsafe_allow_html=True)
            report_text = agent_res.get("report", "No report generated.")
            st.markdown(report_text)
            st.markdown('</div>', unsafe_allow_html=True)

            if report_text:
                st.download_button(
                    label="📥 Download Report as Markdown",
                    data=report_text,
                    file_name="research_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="download_landing",
                )
                export_path = agent_res.get("export_path", "")
                if export_path:
                    st.caption(f"Auto-saved to: `{export_path}`")

        with lt3:
            analysis = agent_res.get("analysis", {})
            if analysis and not analysis.get("error"):
                st.markdown('<div class="glass-wrapper">', unsafe_allow_html=True)
                st.markdown("<h3>📊 ML Analysis Results</h3>", unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("🎯 Predicted Topic", f"Topic {analysis.get('predicted_topic', '?')}")
                with col_b:
                    st.metric("🧩 Predicted Cluster", f"Cluster {analysis.get('predicted_cluster', '?')}")

                keywords = analysis.get("keywords", [])
                if keywords:
                    kw_html = "".join([f'<span class="keyword-badge">{kw}</span>' for kw in keywords])
                    st.markdown(f'<div class="badge-container">{kw_html}</div>', unsafe_allow_html=True)

                topic_dist = analysis.get("topic_distribution", [])
                if topic_dist:
                    df_t = pd.DataFrame({
                        "Probability": topic_dist,
                        "Topic": [f"Topic {i}" for i in range(len(topic_dist))]
                    }).set_index("Topic")
                    st.bar_chart(df_t, color="#8b5cf6")

                per_item = analysis.get("per_item_analysis", [])
                if per_item:
                    st.markdown("#### Per-Source Analysis")
                    df_per = pd.DataFrame(per_item)
                    st.dataframe(df_per, use_container_width=True, hide_index=True)

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No ML analysis available.")

        with lt4:
            sources = agent_res.get("summarized_results", [])
            if sources:
                st.markdown('<div class="glass-wrapper card-accent-amber">', unsafe_allow_html=True)
                st.markdown("<h3>🔗 Sources & References</h3>", unsafe_allow_html=True)
                for src in sources:
                    title = src.get("title", "Untitled")
                    link = src.get("link", "")
                    snippet = src.get("snippet", "")
                    if link:
                        st.markdown(f"**[{title}]({link})**")
                    else:
                        st.markdown(f"**{title}**")
                    st.caption(snippet[:200])
                    st.divider()
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No sources available.")

        with lt5:
            st.markdown('<div class="glass-wrapper card-accent-rose">', unsafe_allow_html=True)
            st.markdown("<h3>🔮 Follow-Up Research Questions</h3>", unsafe_allow_html=True)
            followups = agent_res.get("follow_up_questions", [])
            if followups:
                for i, fq in enumerate(followups):
                    if st.button(f"🔎 {fq}", key=f"followup_landing_{i}", use_container_width=True):
                        st.session_state.agent_query = fq
                        st.rerun()
            else:
                st.info("No follow-up questions were generated.")
            st.markdown('</div>', unsafe_allow_html=True)

