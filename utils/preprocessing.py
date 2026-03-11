import re

def preprocess_text(text):
    """
    Clean and preprocess text by removing noise, URLs, special characters,
    and extra spaces, returning a clean string for vectorization and analysis.
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Convert to lower case
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_text(text):
    """
    Used for general text cleaning in the pipeline (e.g. search snippets).
    """
    if not text or not isinstance(text, str):
        return ""
    # Strip leading/trailing whitespace
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

def is_valid_source(source_text):
    """
    Filter out irrelevant or low-quality sources based on length and content.
    """
    if not source_text or not isinstance(source_text, str):
        return False
    
    # Typically, a valid research abstract or snippet is at least somewhat long
    if len(source_text.strip()) < 50:
        return False
        
    # Check for login pages, forums, or common non-academic keywords
    invalid_keywords = ['login', 'sign in', 'cookie policy', 'forum', 'buy now']
    lower_text = source_text.lower()
    for kw in invalid_keywords:
        if kw in lower_text:
            return False
            
    return True
