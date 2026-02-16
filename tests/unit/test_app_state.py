"""Tests for application state management."""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from neural_terminal.app_state import (
    AppConfig,
    SessionState,
    ApplicationState,
    get_app_state,
    init_app,
)


class TestAppConfig:
    """Tests for AppConfig."""
    
    def test_default_values(self):
        """Default configuration values."""
        config = AppConfig()
        
        assert config.default_model == "meta/llama-3.1-8b-instruct"
        assert config.budget_limit is None
        assert config.theme == "amber"
        assert config.max_tokens_per_message == 8192
        assert config.temperature == 0.7
        assert config.system_prompt == "You are a helpful assistant"
    
    def test_to_dict(self):
        """Convert to dictionary."""
        config = AppConfig(
            budget_limit=Decimal("10.00"),
        )
        
        data = config.to_dict()
        
        # API key not stored in config (comes from .env)
        assert "openrouter_api_key" not in data
        assert data["budget_limit"] == "10.00"
        assert data["theme"] == "amber"
        assert data["system_prompt"] == "You are a helpful assistant"
    
    def test_to_dict_no_budget(self):
        """Dictionary with no budget."""
        config = AppConfig()
        
        data = config.to_dict()
        
        assert data["budget_limit"] is None
    
    def test_from_dict(self):
        """Create from dictionary."""
        data = {
            "default_model": "gpt-4",
            "budget_limit": "25.00",
            "theme": "minimal",
            "max_tokens_per_message": 2000,
            "temperature": 0.5,
            "system_prompt": "Custom prompt",
        }
        
        config = AppConfig.from_dict(data)
        
        # API key not stored in config (comes from .env)
        assert not hasattr(config, 'openrouter_api_key')
        assert config.default_model == "gpt-4"
        assert config.budget_limit == Decimal("25.00")
        assert config.theme == "minimal"
        assert config.system_prompt == "Custom prompt"
    
    def test_from_dict_partial(self):
        """Create from partial dictionary."""
        data = {"theme": "minimal"}
        
        config = AppConfig.from_dict(data)
        
        assert config.theme == "minimal"
        assert config.default_model == "meta/llama-3.1-8b-instruct"  # Default


class TestSessionState:
    """Tests for SessionState."""
    
    def test_default_values(self):
        """Default session state."""
        state = SessionState()
        
        assert isinstance(state.app_config, AppConfig)
        assert state.initialized is False
        assert state.error_message is None
        assert state.current_page == "chat"
        assert state.conversations == []
        assert state.current_conversation_id is None
        assert state.is_streaming is False
        assert state.total_cost == Decimal("0.00")


class TestApplicationState:
    """Tests for ApplicationState."""
    
    def test_singleton_pattern(self):
        """ApplicationState is a singleton."""
        app1 = ApplicationState()
        app2 = ApplicationState()
        
        assert app1 is app2
    
    @patch("neural_terminal.app_state.st")
    def test_session_property(self, mock_st):
        """Session property returns state."""
        mock_st.session_state = {}
        
        app = ApplicationState()
        session = app.session
        
        assert isinstance(session, SessionState)
    
    @patch("neural_terminal.app_state.st")
    def test_config_property(self, mock_st):
        """Config property access."""
        mock_st.session_state = {}
        
        app = ApplicationState()
        config = app.config
        
        assert isinstance(config, AppConfig)
    
    @patch("neural_terminal.app_state.st")
    def test_is_initialized(self, mock_st):
        """Check initialization status."""
        mock_st.session_state = {}
        
        app = ApplicationState()
        
        assert app.is_initialized() is False
        
        app.session.initialized = True
        assert app.is_initialized() is True
    
    @patch("neural_terminal.app_state.st")
    def test_update_config(self, mock_st):
        """Update configuration."""
        mock_st.session_state = {}
        
        app = ApplicationState()
        app.update_config(theme="amber", temperature=0.9)
        
        assert app.config.theme == "amber"
        assert app.config.temperature == 0.9
    
    @patch("neural_terminal.app_state.st")
    def test_set_current_conversation(self, mock_st):
        """Set current conversation."""
        mock_st.session_state = {}
        
        app = ApplicationState()
        app.set_current_conversation("conv-123")
        
        assert app.session.current_conversation_id == "conv-123"
    
    @patch("neural_terminal.app_state.st")
    def test_delete_conversation(self, mock_st):
        """Delete conversation from list."""
        mock_st.session_state = {}
        
        app = ApplicationState()
        # Manually set initialized to True for test
        app._initialized = True
        app._orchestrator = MagicMock()  # Mock orchestrator
        
        app.session.conversations = [
            {"id": "conv-1", "title": "Test 1"},
            {"id": "conv-2", "title": "Test 2"},
        ]
        
        result = app.delete_conversation("conv-1")
        
        assert result is True
        assert len(app.session.conversations) == 1
        assert app.session.conversations[0]["id"] == "conv-2"
    
    @patch("neural_terminal.app_state.st")
    def test_delete_current_conversation(self, mock_st):
        """Delete current conversation resets ID."""
        mock_st.session_state = {}
        
        app = ApplicationState()
        # Manually set initialized to True for test
        app._initialized = True
        app._orchestrator = MagicMock()  # Mock orchestrator
        
        app.session.conversations = [{"id": "conv-1", "title": "Test"}]
        app.session.current_conversation_id = "conv-1"
        
        app.delete_conversation("conv-1")
        
        assert app.session.current_conversation_id is None


class TestGetAppState:
    """Tests for get_app_state function."""
    
    def test_returns_singleton(self):
        """Returns singleton instance."""
        app1 = get_app_state()
        app2 = get_app_state()
        
        assert app1 is app2


class TestInitApp:
    """Tests for init_app function."""
    
    @patch("neural_terminal.app_state.ApplicationState")
    def test_initializes_app(self, mock_app_class):
        """Initializes application."""
        mock_app = MagicMock()
        mock_app.is_initialized.return_value = False
        mock_app_class.return_value = mock_app
        
        result = init_app()
        
        mock_app.initialize.assert_called_once()
        assert result is mock_app
    
    @patch("neural_terminal.app_state.ApplicationState")
    def test_skips_if_initialized(self, mock_app_class):
        """Skips initialization if already done."""
        mock_app = MagicMock()
        mock_app.is_initialized.return_value = True
        mock_app_class.return_value = mock_app
        
        result = init_app()
        
        mock_app.initialize.assert_not_called()
        assert result is mock_app
