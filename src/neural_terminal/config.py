"""Configuration for Neural Terminal."""
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore unknown environment variables (e.g., Streamlit vars like STREAMLIT_SERVER_PORT)
        extra="ignore",
    )
    
    # API Configuration
    openrouter_api_key: SecretStr = Field(
        default=SecretStr("test-key"),
        description="OpenRouter API key",
        validation_alias="OPENROUTER_API_KEY"
    )
    openrouter_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias="OPENROUTER_BASE_URL"
    )
    openrouter_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        validation_alias="OPENROUTER_TIMEOUT"
    )
    
    # Application
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    database_url: str = Field(
        default="sqlite:///neural_terminal.db",
        pattern=r"^sqlite://.*$"  # Force SQLite for now
    )
    
    # Circuit Breaker
    circuit_failure_threshold: int = Field(default=5, ge=1, le=20)
    circuit_recovery_timeout: int = Field(default=30, ge=5, le=300)
    
    @field_validator("database_url")
    @classmethod
    def ensure_absolute_path(cls, v: str) -> str:
        """Convert relative SQLite paths to absolute."""
        if v.startswith("sqlite:///./"):
            path = Path(v.replace("sqlite:///./", "")).resolve()
            return f"sqlite:///{path}"
        return v
    
    @property
    def db_path(self) -> Path:
        """Extract path for migrations."""
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", ""))
        raise ValueError("Not a SQLite database")


# Global settings instance
settings = Settings()
