"""
Configuration management for the Hello-Agent application.
"""
import os
from typing import Optional


class Config:
    """Application configuration."""

    def __init__(self):
        self.model_id: str = os.getenv("MODEL_ID", "")
        self.tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate that all required configuration is present.

        Returns:
            tuple: (is_valid, error_message)
        """
        if not self.gemini_api_key:
            return False, "GEMINI_API_KEY is not set"
        if not self.model_id:
            return False, "MODEL_ID is not set (e.g., gemini-1.5-flash)"
        if not self.tavily_api_key:
            return False, "TAVILY_API_KEY is not set"
        if not self.google_api_key:
            return False, "GOOGLE_API_KEY is not set"
        return True, None


def get_config() -> Config:
    """Get application configuration."""
    return Config()
