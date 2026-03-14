from core.model import generate_text

def generate_answer(query, papers):
    context = "\n\n".join([p["abstract"] for p in papers[:3]])

    prompt = f"""
You are an expert researcher.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Structure:
1. Definition
2. Key Concepts
3. Example
4. Applications

Be precise and avoid hallucination.
"""

    return generate_text(prompt)
