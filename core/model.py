"""
Core Model Loader — Shared flan-t5-small (fast CPU inference)
==============================================================
Uses flan-t5-small for fast CPU inference (~77MB vs 3GB for large).
Lazy-loaded singleton so the model is only loaded ONCE on first use.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

_tokenizer = None
_model = None


def _load():
    """Lazy-load the model and tokenizer on first use."""
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _model is None:
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# Lazy proxy so `from core.model import tokenizer, model` still works
class _LazyProxy:
    def __init__(self, loader):
        self._loader = loader
        self._obj = None

    def __getattr__(self, name):
        if self._obj is None:
            self._obj = self._loader()
        return getattr(self._obj, name)

    def __call__(self, *args, **kwargs):
        if self._obj is None:
            self._obj = self._loader()
        return self._obj(*args, **kwargs)


tokenizer = _LazyProxy(lambda: AutoTokenizer.from_pretrained(MODEL_NAME))
model = _LazyProxy(lambda: AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME))


def generate_text(prompt, max_length=300):
    """Generate text with the shared model. Used by all agents."""
    _load()
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = _model.generate(
        inputs["input_ids"],
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
