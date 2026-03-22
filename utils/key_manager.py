"""
utils/key_manager.py — API Key Rotation Manager

Rotates through multiple Groq API keys to avoid hitting the free-tier
rate limit (6000 TPM per key). Each LLM call uses the next key in rotation.

Usage:
    manager = KeyManager(["gsk_key1", "gsk_key2", "gsk_key3"])
    key = manager.get_next_key()  # Returns key1
    key = manager.get_next_key()  # Returns key2
    key = manager.get_next_key()  # Returns key3
    key = manager.get_next_key()  # Returns key1 (wraps around)
"""


class KeyManager:
    """Round-robin API key rotator for Groq rate limit avoidance."""

    def __init__(self, keys: list[str]):
        """
        Initialize with a list of API keys.

        Args:
            keys: List of Groq API key strings (empty strings are filtered out)

        Raises:
            ValueError: If no valid keys are provided
        """
        self.keys = [k.strip() for k in keys if k and k.strip()]
        self.counter = 0
        if not self.keys:
            raise ValueError("At least one API key is required")

    def get_next_key(self) -> str:
        """Get the next API key in rotation (round-robin)."""
        key = self.keys[self.counter % len(self.keys)]
        self.counter += 1
        return key

    def reset(self):
        """Reset the rotation counter to start from the first key."""
        self.counter = 0

    @property
    def num_keys(self) -> int:
        """Number of valid API keys available."""
        return len(self.keys)
