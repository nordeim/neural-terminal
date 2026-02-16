"""Unit tests for configuration.

Tests for Phase 1: Configuration validation and loading.
"""
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from neural_terminal.config import Settings


class TestSettings:
    """Tests for Pydantic Settings."""

    def test_settings_load_with_defaults(self, monkeypatch):
        """Test that settings load with default values when env vars not set."""
        # Mock env file to not exist by using a non-existent path
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        
        # Create settings with mocked env file location
        from pydantic_settings import SettingsConfigDict
        
        class TestSettings(Settings):
            model_config = SettingsConfigDict(
                env_file="/nonexistent/.env",  # Prevent loading real .env
                env_file_encoding="utf-8",
                extra="ignore",
            )
        
        settings = TestSettings()
        
        assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
        assert settings.openrouter_timeout == 60
        assert settings.app_env == "development"
        assert settings.log_level == "INFO"
        assert settings.database_url == "sqlite:///neural_terminal.db"
        assert settings.circuit_failure_threshold == 5
        assert settings.circuit_recovery_timeout == 30

    def test_settings_from_environment(self, monkeypatch):
        """Test loading settings from environment variables."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-api-key")
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        
        settings = Settings()
        
        assert settings.openrouter_api_key.get_secret_value() == "env-api-key"
        assert settings.app_env == "production"
        assert settings.log_level == "DEBUG"

    def test_openrouter_api_key_is_secret(self, monkeypatch):
        """Test that API key is stored as SecretStr."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
        settings = Settings()
        
        # Should be masked when printed
        assert "secret-key" not in str(settings.openrouter_api_key)
        assert "secret-key" not in repr(settings.openrouter_api_key)
        
        # Should be accessible via get_secret_value
        assert settings.openrouter_api_key.get_secret_value() == "secret-key"

    def test_database_url_validation(self, monkeypatch):
        """Test that database URL must start with sqlite://."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        
        # Valid SQLite URLs
        monkeypatch.setenv("DATABASE_URL", "sqlite:///path/to/db.db")
        settings = Settings()
        assert settings.database_url == "sqlite:///path/to/db.db"
        
        monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
        settings = Settings()
        assert settings.database_url == "sqlite:///:memory:"
        
        # Invalid URL should raise validation error
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
        with pytest.raises(ValidationError):
            Settings()

    def test_database_url_absolute_path_conversion(self, monkeypatch):
        """Test relative paths are converted to absolute."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        monkeypatch.setenv("DATABASE_URL", "sqlite:///./relative.db")
        
        settings = Settings()
        
        # Should be converted to absolute path
        assert not settings.database_url.startswith("sqlite:///./")
        assert settings.database_url.startswith("sqlite:///")

    def test_db_path_property(self, monkeypatch):
        """Test db_path property extracts path correctly."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        monkeypatch.setenv("DATABASE_URL", "sqlite:///path/to/database.db")
        
        settings = Settings()
        
        path = settings.db_path
        assert isinstance(path, Path)
        assert "path/to/database.db" in str(path)

    def test_db_path_raises_on_non_sqlite(self, monkeypatch):
        """Test db_path raises error for non-SQLite URLs."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
        
        settings = Settings()
        
        # Temporarily change the URL
        object.__setattr__(settings, 'database_url', 'postgresql://localhost/db')
        
        with pytest.raises(ValueError, match="Not a SQLite database"):
            _ = settings.db_path

    def test_timeout_validation(self, monkeypatch):
        """Test timeout must be between 10 and 300."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        
        # Valid values
        monkeypatch.setenv("OPENROUTER_TIMEOUT", "10")
        settings = Settings()
        assert settings.openrouter_timeout == 10
        
        monkeypatch.setenv("OPENROUTER_TIMEOUT", "300")
        settings = Settings()
        assert settings.openrouter_timeout == 300
        
        # Invalid values
        monkeypatch.setenv("OPENROUTER_TIMEOUT", "5")
        with pytest.raises(ValidationError):
            Settings()
        
        monkeypatch.setenv("OPENROUTER_TIMEOUT", "400")
        with pytest.raises(ValidationError):
            Settings()

    def test_circuit_breaker_threshold_validation(self, monkeypatch):
        """Test circuit breaker threshold must be between 1 and 20."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        
        # Valid values
        monkeypatch.setenv("CIRCUIT_FAILURE_THRESHOLD", "1")
        settings = Settings()
        assert settings.circuit_failure_threshold == 1
        
        monkeypatch.setenv("CIRCUIT_FAILURE_THRESHOLD", "20")
        settings = Settings()
        assert settings.circuit_failure_threshold == 20
        
        # Invalid values
        monkeypatch.setenv("CIRCUIT_FAILURE_THRESHOLD", "0")
        with pytest.raises(ValidationError):
            Settings()
        
        monkeypatch.setenv("CIRCUIT_FAILURE_THRESHOLD", "25")
        with pytest.raises(ValidationError):
            Settings()

    def test_circuit_breaker_timeout_validation(self, monkeypatch):
        """Test circuit breaker timeout must be between 5 and 300."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        
        # Valid values
        monkeypatch.setenv("CIRCUIT_RECOVERY_TIMEOUT", "5")
        settings = Settings()
        assert settings.circuit_recovery_timeout == 5
        
        monkeypatch.setenv("CIRCUIT_RECOVERY_TIMEOUT", "300")
        settings = Settings()
        assert settings.circuit_recovery_timeout == 300
        
        # Invalid values
        monkeypatch.setenv("CIRCUIT_RECOVERY_TIMEOUT", "3")
        with pytest.raises(ValidationError):
            Settings()
        
        monkeypatch.setenv("CIRCUIT_RECOVERY_TIMEOUT", "400")
        with pytest.raises(ValidationError):
            Settings()

    def test_app_env_validation(self, monkeypatch):
        """Test app_env must be valid value."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        
        # Valid values
        for env in ["development", "staging", "production"]:
            monkeypatch.setenv("APP_ENV", env)
            settings = Settings()
            assert settings.app_env == env
        
        # Invalid value
        monkeypatch.setenv("APP_ENV", "invalid")
        with pytest.raises(ValidationError):
            Settings()

    def test_log_level_validation(self, monkeypatch):
        """Test log_level must be valid value."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        
        # Valid values
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            monkeypatch.setenv("LOG_LEVEL", level)
            settings = Settings()
            assert settings.log_level == level
        
        # Invalid value
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError):
            Settings()
