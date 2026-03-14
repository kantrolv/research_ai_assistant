from core.model import generate_text

def summarize(text):
    prompt = f"""
Summarize this into 3 key points:

{text}
"""
    return generate_text(prompt, max_length=150)
