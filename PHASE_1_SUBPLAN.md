# Phase 1: Foundation - Configuration & Domain Layer
## Sub-Plan with Integrated Checklist

**Phase Objective:** Establish immutable infrastructure, complete domain layer, and project tooling.  
**Estimated Duration:** 4-6 hours  
**Success Criteria:** All tests pass, `make lint` passes with zero errors, `make test` runs successfully.  
**Dependencies:** Phase 0 complete (✅)  
**Methodology:** Test-Driven Development (TDD) - RED | GREEN | REFACTOR

---

## Phase 1 Architecture Overview

```
Phase 1 Deliverables:
├── Project Configuration
│   ├── pyproject.toml            [COMPLETE] Update with all dependencies
│   ├── .env.example              [NEW] Environment template
│   └── Makefile                  [NEW] Development commands
│
├── Domain Layer (Pure Business Logic)
│   ├── domain/exceptions.py      [PARTIAL] Add missing exceptions
│   └── domain/models.py          [PARTIAL] Already has core models
│
├── Infrastructure Layer (I/O Concerns)
│   ├── config.py                 [COMPLETE] Already created
│   ├── infrastructure/database.py [COMPLETE] Already created
│   ├── infrastructure/repositories.py [COMPLETE] Already created
│   └── infrastructure/circuit_breaker.py [COMPLETE] Already created
│
└── Test Infrastructure
    ├── tests/conftest.py         [NEW] Pytest fixtures
    ├── tests/unit/test_config.py [NEW] Configuration tests
    └── tests/unit/test_exceptions.py [NEW] Exception hierarchy tests
```

---

## 1.1 Project Setup & Dependencies

### Current State Analysis

The `pyproject.toml` was created in Phase 0 with minimal dependencies:
- pydantic, pydantic-settings, sqlalchemy
- pytest, pytest-asyncio, pytest-cov, mypy, ruff, black

**Missing Dependencies (from design docs):**
- streamlit (core framework)
- httpx (HTTP client for OpenRouter)
- alembic (database migrations)
- python-dotenv (environment loading)
- structlog (structured logging)
- tiktoken (token counting)
- bleach (XSS sanitization)
- respx (HTTP mocking for tests)

### Files to Create/Update

#### File: `pyproject.toml` (UPDATE)

**Current State:**
```toml
[tool.poetry.dependencies]
python = "^3.11"
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
sqlalchemy = "^2.0.23"
```

**Target State:**
```toml
[tool.poetry.dependencies]
python = "^3.11"
streamlit = "^1.28.0"
httpx = "^0.25.0"
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
sqlalchemy = "^2.0.23"
alembic = "^1.12.0"
python-dotenv = "^1.0.0"
structlog = "^23.2.0"
tiktoken = "^0.5.0"
bleach = "^6.1.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
pytest-cov = "^4.1.0"
respx = "^0.20.0"
mypy = "^1.7.0"
ruff = "^0.1.0"
black = "^23.0.0"
```

**TDD Checklist - 1.1:**
- [ ] **RED**: Document current dependency state
- [ ] **GREEN**: Add all missing runtime dependencies
- [ ] **GREEN**: Add all missing dev dependencies
- [ ] **GREEN**: Update tool configurations (mypy strict, ruff, black)
- [ ] **VALIDATE**: `pip install -e .` succeeds with all dependencies

---

#### File: `.env.example` (CREATE)

**Interface Specification:**
```bash
# Neural Terminal Environment Configuration
# Copy this file to .env and fill in your values

# ============================================================================
# OpenRouter API Configuration
# ============================================================================
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# ============================================================================
# Application Configuration
# ============================================================================
APP_ENV=development
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///neural_terminal.db

# ============================================================================
# Circuit Breaker Configuration
# ============================================================================
CIRCUIT_FAILURE_THRESHOLD=5
CIRCUIT_RECOVERY_TIMEOUT=30

# ============================================================================
# Optional: Budget Limits (in USD)
# ============================================================================
# BUDGET_LIMIT=5.00
```

**TDD Checklist:**
- [ ] **GREEN**: Create .env.example with all required variables
- [ ] **GREEN**: Add comments explaining each variable
- [ ] **VALIDATE**: File is syntactically valid

---

#### File: `Makefile` (CREATE)

**Interface Specification:**
```makefile
.PHONY: install test lint format migrate run clean

# ============================================================================
# Installation
# ============================================================================
install:
	poetry install

# ============================================================================
# Testing
# ============================================================================
test:
	poetry run pytest -v --cov=src/neural_terminal --cov-report=term-missing

test-unit:
	poetry run pytest tests/unit -v

test-integration:
	poetry run pytest tests/integration -v

test-coverage:
	poetry run pytest --cov=src/neural_terminal --cov-report=html --cov-report=term

# ============================================================================
# Linting & Formatting
# ============================================================================
lint:
	poetry run ruff check src tests
	poetry run mypy src

format:
	poetry run black src tests
	poetry run ruff check --fix src tests

format-check:
	poetry run black --check src tests

# ============================================================================
# Database
# ============================================================================
migrate:
	poetry run alembic upgrade head

migrate-create:
	poetry run alembic revision --autogenerate -m "$(message)"

# ============================================================================
# Application
# ============================================================================
run:
	poetry run streamlit run src/neural_terminal/app.py

# ============================================================================
# Cleanup
# ============================================================================
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf *.db
```

**TDD Checklist - 1.1:**
- [ ] **GREEN**: Create Makefile with all commands
- [ ] **GREEN**: Test `make install` (if poetry.lock exists)
- [ ] **GREEN**: Test `make test` (should run existing tests)
- [ ] **GREEN**: Test `make lint` (should check ruff and mypy)
- [ ] **VALIDATE**: All make targets work correctly

---

## 1.2 Domain Exceptions (Complete Hierarchy)

### Current State Analysis

`domain/exceptions.py` was created in Phase 0 with basic exceptions:
- `NeuralTerminalError` (base)
- `CircuitBreakerOpenError`
- `OpenRouterAPIError`
- `ValidationError`

**Missing from design docs:**
- More specific validation errors
- Rate limit error (for 429 responses)
- Model unavailable error (for 503 responses)
- Token limit error (for 400 context too long)

### Files to Modify

#### File: `src/neural_terminal/domain/exceptions.py` (UPDATE)

**Current State:** 41 lines, 4 exception classes

**Target State:** Complete hierarchy with specific error types

**Interface Specification:**
```python
"""Domain exceptions for Neural Terminal.

Provides a comprehensive error hierarchy for the application.
All exceptions inherit from NeuralTerminalError for consistent handling.
"""
from typing import Optional


class NeuralTerminalError(Exception):
    """Base exception for all Neural Terminal errors.
    
    Attributes:
        message: Human-readable error description
        code: Machine-readable error code (e.g., 'HTTP_429')
    """
    
    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.message = message
    
    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(NeuralTerminalError):
    """Raised when there's an error in application configuration."""
    pass


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(NeuralTerminalError):
    """Base class for validation errors."""
    pass


class InputTooLongError(ValidationError):
    """Raised when user input exceeds maximum length."""
    
    def __init__(self, message: str, max_length: int, actual_length: int):
        super().__init__(message, code="INPUT_TOO_LONG")
        self.max_length = max_length
        self.actual_length = actual_length


class EmptyInputError(ValidationError):
    """Raised when user input is empty or whitespace-only."""
    
    def __init__(self, message: str = "Input cannot be empty"):
        super().__init__(message, code="EMPTY_INPUT")


# ============================================================================
# Circuit Breaker Errors
# ============================================================================

class CircuitBreakerOpenError(NeuralTerminalError):
    """Raised when circuit breaker is open and operation is rejected.
    
    The circuit breaker prevents cascading failures by rejecting requests
    when the downstream service is failing.
    """
    
    def __init__(self, message: str):
        super().__init__(message, code="CIRCUIT_OPEN")


# ============================================================================
# API Errors
# ============================================================================

class APIError(NeuralTerminalError):
    """Base class for API-related errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int,
        code: Optional[str] = None
    ):
        super().__init__(message, code=code or f"HTTP_{status_code}")
        self.status_code = status_code


class OpenRouterAPIError(APIError):
    """Raised when OpenRouter API returns an error response.
    
    Attributes:
        status_code: HTTP status code
        response_body: Raw response body for debugging
    """
    
    def __init__(
        self,
        message: str,
        status_code: int,
        response_body: Optional[str] = None
    ):
        super().__init__(message, status_code)
        self.response_body = response_body


class RateLimitError(OpenRouterAPIError):
    """Raised when OpenRouter returns 429 Too Many Requests."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after  # Seconds to wait before retry


class ModelUnavailableError(OpenRouterAPIError):
    """Raised when requested model returns 503 Service Unavailable."""
    
    def __init__(self, message: str = "Model temporarily unavailable", model_id: Optional[str] = None):
        super().__init__(message, status_code=503)
        self.model_id = model_id


class TokenLimitError(OpenRouterAPIError):
    """Raised when context exceeds model's token limit (400 error)."""
    
    def __init__(
        self,
        message: str = "Context too long",
        max_tokens: Optional[int] = None,
        actual_tokens: Optional[int] = None
    ):
        super().__init__(message, status_code=400, code="TOKEN_LIMIT")
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens


# ============================================================================
# Service Errors
# ============================================================================

class ServiceError(NeuralTerminalError):
    """Base class for service-layer errors."""
    pass


class ConversationNotFoundError(ServiceError):
    """Raised when requested conversation doesn't exist."""
    
    def __init__(self, conversation_id: str):
        super().__init__(
            f"Conversation {conversation_id} not found",
            code="CONVERSATION_NOT_FOUND"
        )
        self.conversation_id = conversation_id


class MessageNotFoundError(ServiceError):
    """Raised when requested message doesn't exist."""
    pass


# ============================================================================
# Budget Errors
# ============================================================================

class BudgetError(NeuralTerminalError):
    """Base class for budget-related errors."""
    pass


class BudgetExceededError(BudgetError):
    """Raised when conversation cost exceeds budget limit."""
    
    def __init__(
        self,
        message: str = "Budget exceeded",
        accumulated: Optional[str] = None,
        limit: Optional[str] = None
    ):
        super().__init__(message, code="BUDGET_EXCEEDED")
        self.accumulated = accumulated
        self.limit = limit
```

**TDD Checklist - 1.2:**
- [ ] **RED**: Write test `test_exception_hierarchy()`
- [ ] **RED**: Write test `test_error_codes_present()`
- [ ] **RED**: Write test `test_open_router_api_error_attributes()`
- [ ] **RED**: Write test `test_rate_limit_error_retry_after()`
- [ ] **RED**: Write test `test_token_limit_error_attributes()`
- [ ] **RED**: Write test `test_budget_exceeded_error_attributes()`
- [ ] **GREEN**: Implement all exception classes
- [ ] **GREEN**: Add docstrings to all classes
- [ ] **REFACTOR**: Ensure consistent error code format
- [ ] **VALIDATE**: All exception tests pass

---

## 1.3 Test Infrastructure (conftest.py)

### Files to Create

#### File: `tests/conftest.py` (CREATE)

**Interface Specification:**
```python
"""Pytest configuration and fixtures.

Provides shared fixtures for all test types (unit, integration, e2e).
"""
import pytest
from decimal import Decimal
from uuid import uuid4
from datetime import datetime

from neural_terminal.domain.models import (
    Conversation,
    ConversationStatus,
    Message,
    MessageRole,
    TokenUsage,
)
from neural_terminal.infrastructure.repositories import SQLiteConversationRepository


# ============================================================================
# Domain Model Fixtures
# ============================================================================

@pytest.fixture
def sample_conversation():
    """Create a sample conversation for testing."""
    return Conversation(
        id=uuid4(),
        title="Test Conversation",
        model_id="openai/gpt-3.5-turbo",
        status=ConversationStatus.ACTIVE,
        total_cost=Decimal("0.05"),
        total_tokens=100,
        tags=["test"],
    )


@pytest.fixture
def sample_user_message():
    """Create a sample user message."""
    return Message(
        id=uuid4(),
        role=MessageRole.USER,
        content="Hello, this is a test message",
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_assistant_message():
    """Create a sample assistant message with token usage."""
    return Message(
        id=uuid4(),
        role=MessageRole.ASSISTANT,
        content="This is a response",
        token_usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
        cost=Decimal("0.001"),
        latency_ms=500,
        model_id="openai/gpt-3.5-turbo",
        created_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_token_usage():
    """Create sample token usage for testing."""
    return TokenUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
    )


# ============================================================================
# Repository Fixtures
# ============================================================================

@pytest.fixture
def repository():
    """Create a fresh repository instance.
    
    Note: This uses the actual database. For true isolation,
    consider using an in-memory database for unit tests.
    """
    return SQLiteConversationRepository()


@pytest.fixture
def empty_conversation(repository):
    """Create and save an empty conversation."""
    conv = Conversation(
        id=uuid4(),
        title="Empty Test Conversation",
        model_id="openai/gpt-3.5-turbo",
    )
    repository.save(conv)
    return conv


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    from neural_terminal import config
    
    original_settings = config.settings
    
    # Create test settings
    test_settings = config.Settings(
        openrouter_api_key="test-key",
        openrouter_base_url="https://test.openrouter.ai/api/v1",
        database_url="sqlite:///test.db",
        app_env="testing",
    )
    
    monkeypatch.setattr(config, "settings", test_settings)
    
    yield test_settings
    
    # Restore original settings
    monkeypatch.setattr(config, "settings", original_settings)
```

**TDD Checklist - 1.3:**
- [ ] **RED**: Verify conftest.py loads without errors
- [ ] **RED**: Write test using `sample_conversation` fixture
- [ ] **RED**: Write test using `sample_user_message` fixture
- [ ] **RED**: Write test using `repository` fixture
- [ ] **GREEN**: Implement all fixtures
- [ ] **VALIDATE**: Fixtures work correctly in tests

---

## 1.4 Configuration Tests

### Files to Create

#### File: `tests/unit/test_config.py` (CREATE)

**Interface Specification:**
```python
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

    def test_settings_load_with_defaults(self):
        """Test that settings load with default values."""
        settings = Settings(openrouter_api_key="test-key")
        
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
    
    def test_openrouter_api_key_is_secret(self):
        """Test that API key is stored as SecretStr."""
        settings = Settings(openrouter_api_key="secret-key")
        
        # Should be masked when printed
        assert "secret-key" not in str(settings.openrouter_api_key)
        assert "secret-key" not in repr(settings.openrouter_api_key)
        
        # Should be accessible via get_secret_value
        assert settings.openrouter_api_key.get_secret_value() == "secret-key"
    
    def test_database_url_validation(self):
        """Test that database URL must start with sqlite://."""
        # Valid SQLite URLs
        settings = Settings(
            openrouter_api_key="test",
            database_url="sqlite:///path/to/db.db"
        )
        assert settings.database_url == "sqlite:///path/to/db.db"
        
        settings = Settings(
            openrouter_api_key="test",
            database_url="sqlite:///:memory:"
        )
        assert settings.database_url == "sqlite:///:memory:"
        
        # Invalid URL should raise validation error
        with pytest.raises(ValidationError):
            Settings(
                openrouter_api_key="test",
                database_url="postgresql://localhost/db"
            )
    
    def test_database_url_absolute_path_conversion(self):
        """Test relative paths are converted to absolute."""
        settings = Settings(
            openrouter_api_key="test",
            database_url="sqlite:///./relative.db"
        )
        
        # Should be converted to absolute path
        assert not settings.database_url.startswith("sqlite:///./")
        assert settings.database_url.startswith("sqlite:///")
    
    def test_db_path_property(self):
        """Test db_path property extracts path correctly."""
        settings = Settings(
            openrouter_api_key="test",
            database_url="sqlite:///path/to/database.db"
        )
        
        path = settings.db_path
        assert isinstance(path, Path)
        assert str(path) == "/path/to/database.db"
    
    def test_db_path_raises_on_non_sqlite(self):
        """Test db_path raises error for non-SQLite URLs."""
        # This would require bypassing validation, so we test the method directly
        settings = Settings(openrouter_api_key="test")
        
        # Temporarily change the URL
        original_url = settings.database_url
        object.__setattr__(settings, 'database_url', 'postgresql://localhost/db')
        
        with pytest.raises(ValueError, match="Not a SQLite database"):
            _ = settings.db_path
        
        # Restore
        object.__setattr__(settings, 'database_url', original_url)
    
    def test_timeout_validation(self):
        """Test timeout must be between 10 and 300."""
        # Valid values
        settings = Settings(openrouter_api_key="test", openrouter_timeout=10)
        assert settings.openrouter_timeout == 10
        
        settings = Settings(openrouter_api_key="test", openrouter_timeout=300)
        assert settings.openrouter_timeout == 300
        
        # Invalid values
        with pytest.raises(ValidationError):
            Settings(openrouter_api_key="test", openrouter_timeout=5)
        
        with pytest.raises(ValidationError):
            Settings(openrouter_api_key="test", openrouter_timeout=400)
    
    def test_circuit_breaker_threshold_validation(self):
        """Test circuit breaker threshold must be between 1 and 20."""
        # Valid values
        settings = Settings(openrouter_api_key="test", circuit_failure_threshold=1)
        assert settings.circuit_failure_threshold == 1
        
        settings = Settings(openrouter_api_key="test", circuit_failure_threshold=20)
        assert settings.circuit_failure_threshold == 20
        
        # Invalid values
        with pytest.raises(ValidationError):
            Settings(openrouter_api_key="test", circuit_failure_threshold=0)
        
        with pytest.raises(ValidationError):
            Settings(openrouter_api_key="test", circuit_failure_threshold=25)
    
    def test_circuit_breaker_timeout_validation(self):
        """Test circuit breaker timeout must be between 5 and 300."""
        # Valid values
        settings = Settings(openrouter_api_key="test", circuit_recovery_timeout=5)
        assert settings.circuit_recovery_timeout == 5
        
        settings = Settings(openrouter_api_key="test", circuit_recovery_timeout=300)
        assert settings.circuit_recovery_timeout == 300
        
        # Invalid values
        with pytest.raises(ValidationError):
            Settings(openrouter_api_key="test", circuit_recovery_timeout=3)
        
        with pytest.raises(ValidationError):
            Settings(openrouter_api_key="test", circuit_recovery_timeout=400)
```

**TDD Checklist - 1.4:**
- [ ] **RED**: Write test `test_settings_load_with_defaults()`
- [ ] **RED**: Write test `test_settings_from_environment()`
- [ ] **RED**: Write test `test_openrouter_api_key_is_secret()`
- [ ] **RED**: Write test `test_database_url_validation()`
- [ ] **RED**: Write test `test_database_url_absolute_path_conversion()`
- [ ] **RED**: Write test `test_db_path_property()`
- [ ] **RED**: Write test `test_timeout_validation()`
- [ ] **RED**: Write test `test_circuit_breaker_threshold_validation()`
- [ ] **GREEN**: Ensure all tests pass
- [ ] **VALIDATE**: Configuration properly validated

---

## 1.5 Domain Model Tests (Complete Coverage)

### Files to Create

#### File: `tests/unit/test_models_complete.py` (CREATE)

**Note:** We already have `test_models.py` for C-1 fix. This adds comprehensive coverage.

**Interface Specification:**
```python
"""Complete unit tests for domain models.

Phase 1: Comprehensive domain model testing.
"""
from dataclasses import FrozenInstanceError
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from neural_terminal.domain.models import (
    Conversation,
    ConversationStatus,
    Message,
    MessageRole,
    TokenUsage,
)


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"
    
    def test_enum_comparison(self):
        """Test enum comparison works correctly."""
        assert MessageRole.USER == MessageRole.USER
        assert MessageRole.USER != MessageRole.ASSISTANT
        assert MessageRole.USER == "user"  # str comparison works


class TestConversationStatus:
    """Tests for ConversationStatus enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert ConversationStatus.ACTIVE.value == "active"
        assert ConversationStatus.ARCHIVED.value == "archived"
        assert ConversationStatus.FORKED.value == "forked"


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation_defaults(self):
        """Test message creation with default values."""
        msg = Message(content="Hello")
        
        assert isinstance(msg.id, UUID)
        assert msg.role == MessageRole.USER  # Default
        assert msg.content == "Hello"
        assert msg.conversation_id is None
        assert msg.token_usage is None
        assert msg.cost is None
        assert msg.latency_ms is None
        assert msg.model_id is None
        assert isinstance(msg.metadata, dict)
        assert len(msg.metadata) == 0
    
    def test_message_creation_with_values(self):
        """Test message creation with all values."""
        conv_id = uuid4()
        msg_id = uuid4()
        usage = TokenUsage(10, 20, 30)
        
        msg = Message(
            id=msg_id,
            conversation_id=conv_id,
            role=MessageRole.ASSISTANT,
            content="Response",
            token_usage=usage,
            cost=Decimal("0.001"),
            latency_ms=500,
            model_id="gpt-4",
            metadata={"key": "value"},
        )
        
        assert msg.id == msg_id
        assert msg.conversation_id == conv_id
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Response"
        assert msg.token_usage == usage
        assert msg.cost == Decimal("0.001")
        assert msg.latency_ms == 500
        assert msg.model_id == "gpt-4"
        assert msg.metadata == {"key": "value"}


class TestConversation:
    """Tests for Conversation dataclass."""

    def test_conversation_creation_defaults(self):
        """Test conversation creation with default values."""
        conv = Conversation()
        
        assert isinstance(conv.id, UUID)
        assert conv.title is None
        assert conv.model_id == "openai/gpt-3.5-turbo"  # Default
        assert conv.status == ConversationStatus.ACTIVE
        assert isinstance(conv.created_at, datetime)
        assert isinstance(conv.updated_at, datetime)
        assert conv.total_cost == Decimal("0.00")
        assert conv.total_tokens == 0
        assert conv.parent_conversation_id is None
        assert isinstance(conv.tags, list)
        assert len(conv.tags) == 0
    
    def test_update_cost(self):
        """Test update_cost method."""
        conv = Conversation()
        original_updated_at = conv.updated_at
        
        # Wait a tiny bit to ensure timestamp changes
        import time
        time.sleep(0.01)
        
        conv.update_cost(Decimal("0.05"))
        
        assert conv.total_cost == Decimal("0.05")
        assert conv.updated_at > original_updated_at
    
    def test_update_cost_accumulates(self):
        """Test that update_cost accumulates costs."""
        conv = Conversation()
        
        conv.update_cost(Decimal("0.05"))
        conv.update_cost(Decimal("0.10"))
        conv.update_cost(Decimal("0.03"))
        
        assert conv.total_cost == Decimal("0.18")
    
    def test_to_dict(self):
        """Test to_dict serialization."""
        conv = Conversation(
            id=uuid4(),
            title="Test",
            model_id="gpt-4",
            status=ConversationStatus.ACTIVE,
            total_cost=Decimal("0.50"),
            total_tokens=1000,
            tags=["test", "demo"],
        )
        
        data = conv.to_dict()
        
        assert isinstance(data, dict)
        assert data["id"] == str(conv.id)  # UUID -> string
        assert data["title"] == "Test"
        assert data["model_id"] == "gpt-4"
        assert data["status"] == "active"
        assert isinstance(data["created_at"], str)  # datetime -> ISO string
        assert isinstance(data["updated_at"], str)
        assert data["total_cost"] == "0.50"  # Decimal -> string
        assert data["total_tokens"] == 1000
        assert data["parent_conversation_id"] is None
        assert data["tags"] == ["test", "demo"]
    
    def test_to_dict_with_parent(self):
        """Test to_dict with parent conversation."""
        parent_id = uuid4()
        conv = Conversation(
            parent_conversation_id=parent_id,
        )
        
        data = conv.to_dict()
        
        assert data["parent_conversation_id"] == str(parent_id)
```

**TDD Checklist - 1.5:**
- [ ] **RED**: Write test `test_enum_values` for MessageRole
- [ ] **RED**: Write test `test_enum_comparison` for MessageRole
- [ ] **RED**: Write test `test_enum_values` for ConversationStatus
- [ ] **RED**: Write test `test_message_creation_defaults`
- [ ] **RED**: Write test `test_message_creation_with_values`
- [ ] **RED**: Write test `test_conversation_creation_defaults`
- [ ] **RED**: Write test `test_update_cost`
- [ ] **RED**: Write test `test_update_cost_accumulates`
- [ ] **RED**: Write test `test_to_dict`
- [ ] **RED**: Write test `test_to_dict_with_parent`
- [ ] **GREEN**: Ensure all model tests pass
- [ ] **VALIDATE**: Full model coverage achieved

---

## Phase 1 Integration Test

#### File: `tests/integration/test_phase1_foundation.py` (CREATE)

```python
"""Integration test verifying Phase 1 foundation is complete."""


def test_all_imports_work():
    """Test that all Phase 1 modules can be imported."""
    # Domain
    from neural_terminal.domain import exceptions, models
    
    # Infrastructure
    from neural_terminal.infrastructure import (
        circuit_breaker,
        database,
        repositories,
    )
    
    # Configuration
    from neural_terminal import config
    
    # If we get here, all imports succeeded
    assert True


def test_exception_hierarchy():
    """Test that all exceptions are properly defined."""
    from neural_terminal.domain.exceptions import (
        APIError,
        BudgetError,
        BudgetExceededError,
        CircuitBreakerOpenError,
        ConfigurationError,
        ConversationNotFoundError,
        EmptyInputError,
        InputTooLongError,
        ModelUnavailableError,
        NeuralTerminalError,
        OpenRouterAPIError,
        RateLimitError,
        ServiceError,
        TokenLimitError,
        ValidationError,
    )
    
    # Verify hierarchy
    assert issubclass(CircuitBreakerOpenError, NeuralTerminalError)
    assert issubclass(OpenRouterAPIError, APIError)
    assert issubclass(APIError, NeuralTerminalError)
    assert issubclass(ValidationError, NeuralTerminalError)
    assert issubclass(ServiceError, NeuralTerminalError)
    assert issubclass(BudgetError, NeuralTerminalError)


def test_settings_singleton():
    """Test that settings singleton works."""
    from neural_terminal.config import settings
    
    # Should be able to access settings
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.app_env in ["development", "staging", "production"]
```

---

## Phase 1 Validation Criteria

### Pre-Validation Checklist
- [ ] All dependencies added to pyproject.toml
- [ ] .env.example created with all required variables
- [ ] Makefile created with all standard commands
- [ ] conftest.py with comprehensive fixtures
- [ ] Exception hierarchy complete
- [ ] Configuration tests comprehensive
- [ ] Domain model tests comprehensive

### Validation Commands
```bash
# Dependency installation
make install

# Testing
make test
make test-unit
make test-coverage

# Linting
make lint
make format-check

# Application
make migrate
```

### Success Criteria
- [ ] `make install` succeeds
- [ ] `make test` passes (all tests)
- [ ] `make lint` passes with zero errors
- [ ] `make format-check` passes
- [ ] Code coverage > 90%
- [ ] All imports work without errors
- [ ] No TODO or FIXME comments remaining

---

## Phase 1 Exit Criteria

Before proceeding to Phase 2, the following MUST be true:

1. **All infrastructure in place**
2. **Test coverage > 90% for all files**
3. **Zero linting errors**
4. **Zero type checking errors**
5. **All imports verified**
6. **Makefile commands work**

---

## Time Estimates

| Task | Estimated | Actual |
|------|-----------|--------|
| 1.1 Project Setup | 30 min | ___ |
| 1.2 Domain Exceptions | 30 min | ___ |
| 1.3 Test Infrastructure | 20 min | ___ |
| 1.4 Configuration Tests | 30 min | ___ |
| 1.5 Domain Model Tests | 40 min | ___ |
| **Total** | **~2.5 hours** | ___ |

---

*Sub-Plan Version: 1.0*  
*Created: 2026-02-15*  
*Status: Ready for Review*
