from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Dict, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "postgresql://n8n_user:redhat%40123@10.121.210.176:5432/rag_system"

    # OpenAI (via Portkey)
    # Optional: Only needed if NOT using virtual keys or provider routing
    openai_api_key: Optional[str] = None

    # Portkey Configuration
    portkey_api_key: str
    portkey_virtual_key: Optional[str] = None  # Use either virtual_key OR provider headers
    portkey_chat_provider: Optional[str] = None  # e.g., "@azure-openai-eus-global/gpt-4.1-dzs"
    portkey_embedding_provider: Optional[str] = None  # e.g., "@azure-openai-eus-global/text-embedding-3-large-std"
    portkey_base_url: str = "https://api.portkey.ai/v1"

    # Models
    embedding_model: str = "text-embedding-3-large-std"
    chat_model: str = "gpt-4.1-dzs"
    embedding_dimension: int = 3072  # text-embedding-3-large uses 3072 dimensions

    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def effective_openai_api_key(self) -> str:
        """
        Get the effective OpenAI API key.
        When using virtual keys, returns a placeholder since Portkey manages the actual key.
        """
        if self.openai_api_key:
            return self.openai_api_key
        # When using virtual keys, we need a placeholder key for LangChain initialization
        # The actual authentication is handled by Portkey via the virtual key
        return "sk-placeholder-for-virtual-key"

    @property
    def portkey_chat_headers(self) -> Dict[str, str]:
        """
        Headers for Portkey chat completions.
        Supports both virtual keys and provider routing.
        """
        headers = {"x-portkey-api-key": self.portkey_api_key}

        # Use provider routing if specified (takes precedence)
        if self.portkey_chat_provider:
            headers["x-portkey-provider"] = self.portkey_chat_provider
        # Fall back to virtual key if specified
        elif self.portkey_virtual_key:
            headers["x-portkey-virtual-key"] = self.portkey_virtual_key

        return headers

    @property
    def portkey_embedding_headers(self) -> Dict[str, str]:
        """
        Headers for Portkey embeddings.
        Supports both virtual keys and provider routing.
        """
        headers = {"x-portkey-api-key": self.portkey_api_key}

        # Use provider routing if specified (takes precedence)
        if self.portkey_embedding_provider:
            headers["x-portkey-provider"] = self.portkey_embedding_provider
        # Fall back to virtual key if specified
        elif self.portkey_virtual_key:
            headers["x-portkey-virtual-key"] = self.portkey_virtual_key

        return headers


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
