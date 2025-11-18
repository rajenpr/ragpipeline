from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Dict, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ragdb"

    # OpenAI (via Portkey)
    openai_api_key: str

    # Portkey Configuration
    portkey_api_key: str
    portkey_virtual_key: Optional[str] = None  # Optional: only needed if using virtual keys
    portkey_base_url: str = "https://api.portkey.ai/v1"

    # Models
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4-turbo-preview"
    embedding_dimension: int = 1536

    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def portkey_chat_headers(self) -> Dict[str, str]:
        """Headers for Portkey chat completions."""
        headers = {"x-portkey-api-key": self.portkey_api_key}
        if self.portkey_virtual_key:
            headers["x-portkey-virtual-key"] = self.portkey_virtual_key
        return headers

    @property
    def portkey_embedding_headers(self) -> Dict[str, str]:
        """Headers for Portkey embeddings."""
        headers = {"x-portkey-api-key": self.portkey_api_key}
        if self.portkey_virtual_key:
            headers["x-portkey-virtual-key"] = self.portkey_virtual_key
        return headers


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
