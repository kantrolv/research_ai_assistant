from core.model import generate_text

def critique_answer(answer):
    prompt = f"""
Check the following answer for:
- correctness
- missing details
- vagueness

If needed, improve it.

Answer:
{answer}
"""
    return generate_text(prompt)
