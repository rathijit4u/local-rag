class LLMConnectionError(Exception):
    """Raised when the LLM server cannot be reached."""
    def __init__(self, message: str, url: str = None):
        full_message = f"LLMConnectionError: {message}"
        if url:
            full_message += f" | URL: {url}"
        super().__init__(full_message)
        self.url = url