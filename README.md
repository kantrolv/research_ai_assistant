# Agentic AI Research Assistant

AI-powered system for analyzing research papers.

## Project Description

The Agentic AI Research Assistant is a modern, high-end web application that allows researchers to easily analyze abstracts and full research text. It uses machine learning models to provide deep insights, semantic summaries, and intelligent topic mapping, wrapped in a professional AI SaaS-style dashboard.

### Core Features

- **ChatGPT-Style Research Assistant:** Ask natural language questions about the paper (e.g., "What is the main idea?", "What area does this belong to?").
- **Research Summary:** AI generated extractive summarization displayed in modern glass UI cards.
- **Topic Detection:** Identify the primary topic and visualize the probability distribution using interactive charts.
- **Cluster Category:** Determine logical groupings via semantic cluster assignments.
- **Keyword Tags:** Extracts key feature terms represented as interactive badges.
- **Word Cloud Visualization:** Highlights prominent research vocabulary.
- **Topic Network Graph:** Provides a structural relationship of the dominant topics mapped out via NetworkX.
- **Paper Similarity Search:** Computes TF-IDF cosine similarity to match the abstract against a library of other papers.

## How to Install

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

*(Note: The `sumy` library requires NLTK tokenizers. The app usually attempts to download these automatically, but you can manually download `punkt` and `punkt_tab` if needed.)*

## How to Run

To start the dashboard locally:

```bash
streamlit run app.py
```

After running the command, open the provided `localhost` URL in your web browser.
