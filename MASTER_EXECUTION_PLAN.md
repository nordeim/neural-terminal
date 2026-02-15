# Neural Terminal - Master Execution Plan
## Production-Grade Streamlit Chatbot with OpenRouter Integration

**Version:** 1.0  
**Date:** 2026-02-15  
**Status:** Ready for Execution  
**Methodology:** Test-Driven Development (TDD) - RED | GREEN | REFACTOR

---

## Executive Summary

This master plan consolidates the validated design from `new_design_1.md` through `new_design_5.md` and incorporates critical fixes from `new_design_critique.md` and `new_design_critique_2.md`. The architecture follows Clean Architecture principles with distinct domain, infrastructure, and application layers.

### Critical Pre-Implementation Notes (From Design Critique)

The design critique identified **17 concrete defects** across 4 severity levels. **All Critical (C-1 through C-7) defects MUST be addressed in Phase 0** before any feature implementation:

| Defect ID | Severity | Issue | Impact |
|-----------|----------|-------|--------|
| C-1 | Critical | `TokenUsage.cost` property with arguments | Runtime crash |
| C-2 | Critical | SQLite PRAGMA foreign_keys never enabled | Data corruption |
| C-3 | Critical | Repository session context manager leak | Connection pool exhaustion |
| C-4 | Critical | Circuit breaker + AsyncGenerator incompatibility | Streaming crashes |
| C-5 | Critical | Missing `import json` in openrouter.py | Runtime crash |
| C-6 | Critical | CostTracker creates orphan EventBus instances | Budget alerts fail silently |
| C-7 | Critical | Unit test will always fail | Test suite broken |

---

## Project Structure

```
neural-terminal/
├── pyproject.toml                 # Poetry config, dependencies, tool settings
├── .env.example                   # Environment template
├── alembic/
│   ├── env.py                     # Migration environment (FIXED per C-2)
│   ├── script.py.mako             # Migration template
│   └── versions/                  # Migration scripts
├── src/
│   └── neural_terminal/
│       ├── __init__.py
│       ├── config.py              # Pydantic settings with SecretStr
│       ├── domain/                # Pure business logic
│       │   ├── __init__.py
│       │   ├── models.py          # Entities: Conversation, Message, TokenUsage
│       │   └── exceptions.py      # Custom error hierarchy
│       ├── infrastructure/        # I/O and external concerns
│       │   ├── __init__.py
│       │   ├── database.py        # SQLAlchemy + SQLite (FIXED per C-2)
│       │   ├── repositories.py    # Repository pattern (FIXED per C-3)
│       │   ├── circuit_breaker.py # Thread-safe circuit breaker (FIXED per H-2)
│       │   ├── openrouter.py      # OpenRouter API client (FIXED per C-4, C-5)
│       │   └── token_counter.py   # Tiktoken integration
│       ├── application/           # Use cases and orchestration
│       │   ├── __init__.py
│       │   ├── events.py          # Event bus with typed observers
│       │   ├── cost_tracker.py    # Budget tracking (FIXED per C-6)
│       │   ├── orchestrator.py    # ChatOrchestrator (FIXED per C-4)
│       │   └── state.py           # Session state proxy
│       ├── components/            # UI components
│       │   ├── __init__.py
│       │   ├── renderers.py       # Message bubble renderer (FIXED per S-1)
│       │   ├── telemetry.py       # Cost/latency dashboard
│       │   └── stream_bridge.py   # Async-to-sync bridge (FIXED per H-3)
│       ├── styles/
│       │   └── theme.py           # CSS design tokens
│       └── app.py                 # Streamlit entry (FIXED per ST-2, H-3)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures
│   ├── unit/                      # Unit tests
│   │   ├── test_config.py
│   │   ├── test_models.py
│   │   ├── test_circuit_breaker.py (FIXED per C-7)
│   │   ├── test_token_counter.py
│   │   └── test_cost_tracker.py
│   ├── integration/               # Integration tests
│   │   ├── test_database.py
│   │   ├── test_openrouter.py
│   │   ├── test_repositories.py
│   │   └── test_streaming.py
│   └── e2e/                       # End-to-end tests (Phase 6)
│       └── test_chat_flow.py
├── Dockerfile                     # Multi-stage build (Phase 6)
├── docker-compose.yml             # Local development stack
└── Makefile                       # Common commands
```

---

## Phase 0: Critical Bug Fixes & Architecture Corrections

**Objective:** Fix all runtime-critical defects before feature implementation.  
**Estimated Effort:** 2-3 hours  
**Validation:** All unit tests pass, no runtime crashes on basic operations.

### Files to Create/Update

#### 1. `src/neural_terminal/domain/models.py` (FIXED per C-1, H-1)

**Changes Required:**
- Convert `TokenUsage.cost` from property to method (C-1)
- Simplify `Conversation.update_cost` to use regular assignment (H-1)
- Add `to_dict()` method for state serialization (H-4)

**Interface Definition:**
```python
@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    def calculate_cost(self, price_per_1k_prompt: Decimal, price_per_1k_completion: Decimal) -> Decimal:
        """Calculate cost - REGULAR METHOD, not property"""
        ...

@dataclass
class Conversation:
    # ... fields ...
    
    def update_cost(self, message_cost: Decimal) -> None:
        """Simple assignment - dataclass is not frozen"""
        self.total_cost += message_cost
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for session state"""
        ...
```

**TDD Checklist:**
- [ ] RED: Write test for `TokenUsage.calculate_cost()` with sample pricing
- [ ] GREEN: Implement method (not property)
- [ ] RED: Write test for `Conversation.to_dict()` serialization
- [ ] GREEN: Implement with proper Decimal/UUID handling
- [ ] Verify `calculate_cost` returns expected values for known token/pricing combinations

#### 2. `src/neural_terminal/infrastructure/database.py` (FIXED per C-2)

**Changes Required:**
- Fix missing imports (`Column`, `Text`, `datetime`)
- Fix event listener to target ENGINE INSTANCE, not function
- Use `DeclarativeBase` instead of `declarative_base()`
- Add `PRAGMA journal_mode=WAL` for better concurrency

**Interface Definition:**
```python
from datetime import datetime
from sqlalchemy import (
    JSON, Column, DateTime, Enum, ForeignKey, Integer, 
    Numeric, String, Text, Uuid, create_engine, event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, scoped_session, sessionmaker

class Base(DeclarativeBase):
    pass

# Create engine FIRST
engine = create_engine(...)

# Listen on ENGINE INSTANCE
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()
```

**TDD Checklist:**
- [ ] RED: Write test verifying foreign keys are enforced
- [ ] GREEN: Implement fixed event listener
- [ ] RED: Write test for cascading delete (conversation -> messages)
- [ ] GREEN: Verify cascade behavior works
- [ ] Verify `PRAGMA foreign_keys` returns 1 (enabled)

#### 3. `src/neural_terminal/infrastructure/repositories.py` (FIXED per C-3, H-5)

**Changes Required:**
- Replace broken context manager pattern with `_session_scope()`
- Add `get_messages()` method (H-5)
- Add `_message_to_domain()` converter
- Use `SessionLocal.remove()` for cleanup

**Interface Definition:**
```python
class ConversationRepository(ABC):
    @abstractmethod
    def get_by_id(self, conversation_id: UUID) -> Optional[Conversation]: ...
    
    @abstractmethod
    def get_messages(self, conversation_id: UUID) -> List[Message]: ...  # NEW
    
    @abstractmethod
    def save(self, conversation: Conversation) -> None: ...
    
    @abstractmethod
    def add_message(self, message: Message) -> None: ...
    
    @abstractmethod
    def list_active(self, limit: int = 50, offset: int = 0) -> List[Conversation]: ...

class SQLiteConversationRepository(ConversationRepository):
    @contextmanager
    def _session_scope(self) -> Generator[Session, None, None]:
        """Proper atomic unit of work"""
        ...
```

**TDD Checklist:**
- [ ] RED: Write test for `get_messages()` returning ordered messages
- [ ] GREEN: Implement `_session_scope()` and `get_messages()`
- [ ] RED: Write test for concurrent repository access
- [ ] GREEN: Verify `scoped_session` isolation works
- [ ] Verify no session leaks under load (stress test)

#### 4. `src/neural_terminal/infrastructure/circuit_breaker.py` (FIXED per H-2, C-4 prep)

**Changes Required:**
- Add `threading.Lock()` for thread safety (H-2)
- Lock around all state mutations
- Add `_check_state()` method for manual circuit state check (C-4 prep)

**Interface Definition:**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self._lock = threading.Lock()
        # ... other init ...
    
    def _check_state(self) -> None:
        """Check if circuit allows operation - for manual checks"""
        ...
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T: ...
    
    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T: ...
    
    def _on_success(self) -> None:
        with self._lock: ...
    
    def _on_failure(self) -> None:
        with self._lock: ...
```

**TDD Checklist:**
- [ ] RED: Write test for circuit opening after threshold failures (FIXED per C-7)
- [ ] GREEN: Implement with proper exception handling in test
- [ ] RED: Write test for thread-safe concurrent failures
- [ ] GREEN: Verify lock prevents race conditions
- [ ] Verify circuit transitions: CLOSED -> OPEN -> HALF_OPEN -> CLOSED

#### 5. `src/neural_terminal/infrastructure/openrouter.py` (FIXED per C-4, C-5)

**Changes Required:**
- Add missing `import json` (C-5)
- Split `chat_completion` into `chat_completion_stream` that yields directly (C-4)
- Connection errors surface before first yield
- Success/failure recorded manually by caller (not in circuit wrapper)

**Interface Definition:**
```python
class OpenRouterClient:
    async def get_available_models(self) -> List[OpenRouterModel]: ...
    
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Yields chunks. Connection errors surface before first yield."""
        ...
    
    async def close(self): ...
```

**TDD Checklist:**
- [ ] RED: Write test with mocked SSE stream
- [ ] GREEN: Implement streaming with proper error handling
- [ ] RED: Write test for JSON parsing errors
- [ ] GREEN: Verify graceful handling of malformed SSE
- [ ] Verify connection errors raised before first yield

#### 6. `src/neural_terminal/application/cost_tracker.py` (FIXED per C-6)

**Changes Required:**
- Inject `EventBus` in constructor instead of creating new instances
- Use `self._bus.emit()` instead of `EventBus()`

**Interface Definition:**
```python
class CostTracker(EventObserver):
    def __init__(self, event_bus: EventBus, budget_limit: Optional[Decimal] = None):
        self._bus = event_bus  # Injected singleton
        # ... rest of init ...
    
    def on_event(self, event: DomainEvent) -> None: ...
```

**TDD Checklist:**
- [ ] RED: Write test verifying budget events are emitted to same bus
- [ ] GREEN: Implement with injected EventBus
- [ ] RED: Write test for 80% threshold warning
- [ ] GREEN: Verify BUDGET_THRESHOLD event emitted
- [ ] RED: Write test for budget exceeded
- [ ] GREEN: Verify BUDGET_EXCEEDED event emitted

#### 7. `tests/unit/test_circuit_breaker.py` (FIXED per C-7)

**Changes Required:**
- Wrap failing calls in `pytest.raises()`
- Verify circuit opens after threshold failures

**TDD Checklist:**
- [ ] RED: Write test with proper exception context managers
- [ ] GREEN: Fix test assertions
- [ ] Verify all circuit states are tested

---

## Phase 1: Foundation - Configuration & Domain Layer

**Objective:** Establish immutable infrastructure, project structure, and domain models.  
**Estimated Effort:** 4-6 hours  
**Dependencies:** Phase 0 complete  
**Validation:** Unit tests pass, database migrations run successfully.

### 1.1 Project Setup

#### Files to Create:

**1. `pyproject.toml`**
- Poetry configuration
- Dependencies: streamlit, httpx, pydantic, sqlalchemy, alembic, tiktoken, bleach, structlog
- Dev dependencies: pytest, pytest-asyncio, pytest-cov, respx, mypy, ruff, black
- Tool configurations (mypy strict, ruff, black)

**TDD Checklist:**
- [ ] RED: Verify `poetry install` works
- [ ] GREEN: Fix any dependency conflicts
- [ ] Verify `poetry run pytest` executes (even if no tests yet)
- [ ] Verify `poetry run mypy` runs without errors on empty project

**2. `.env.example`**
```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
APP_ENV=development
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///neural_terminal.db
CIRCUIT_FAILURE_THRESHOLD=5
CIRCUIT_RECOVERY_TIMEOUT=30
```

**3. `Makefile`**
```makefile
.PHONY: install test lint format migrate run

install:
	poetry install

test:
	poetry run pytest -v --cov=src/neural_terminal

lint:
	poetry run ruff check src tests
	poetry run mypy src

format:
	poetry run black src tests
	poetry run ruff check --fix src tests

migrate:
	poetry run alembic upgrade head

run:
	poetry run streamlit run src/neural_terminal/app.py
```

### 1.2 Domain Layer

#### Files to Create:

**1. `src/neural_terminal/domain/exceptions.py`**

**Features:**
- `NeuralTerminalError` base exception with message and code
- `CircuitBreakerOpenError` for API disabled state
- `OpenRouterAPIError` with status_code and response_body
- `ValidationError` for input validation failures

**TDD Checklist:**
- [ ] RED: Write test for exception inheritance hierarchy
- [ ] GREEN: Implement exception classes
- [ ] Verify all exceptions have proper attributes

**2. `src/neural_terminal/domain/models.py`**

**Features:**
- `MessageRole` Enum (USER, ASSISTANT, SYSTEM)
- `ConversationStatus` Enum (ACTIVE, ARCHIVED, FORKED)
- `TokenUsage` dataclass (frozen) with `calculate_cost()` method
- `Message` dataclass with metadata dict
- `Conversation` dataclass with `update_cost()` and `to_dict()` methods

**TDD Checklist:**
- [ ] RED: Write test for TokenUsage.calculate_cost with known values
- [ ] GREEN: Implement TokenUsage
- [ ] RED: Write test for Conversation cost accumulation
- [ ] GREEN: Implement Conversation
- [ ] RED: Write test for Conversation.to_dict serialization
- [ ] GREEN: Implement serialization with Decimal/UUID handling
- [ ] Verify frozen dataclasses are immutable

### 1.3 Configuration Layer

#### Files to Create:

**1. `src/neural_terminal/config.py`**

**Features:**
- Pydantic Settings with `SettingsConfigDict`
- `SecretStr` for API key masking
- Field validators for database URL (absolute path conversion)
- Properties for derived values (db_path)

**Interface Definition:**
```python
class Settings(BaseSettings):
    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout: int = Field(default=60, ge=10, le=300)
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    database_url: str = Field(default="sqlite:///neural_terminal.db", pattern=r"^sqlite://.*$")
    circuit_failure_threshold: int = Field(default=5, ge=1, le=20)
    circuit_recovery_timeout: int = Field(default=30, ge=5, le=300)
```

**TDD Checklist:**
- [ ] RED: Write test for settings loading from env
- [ ] GREEN: Implement Settings class
- [ ] RED: Write test for SecretStr masking
- [ ] GREEN: Verify API key is masked in logs
- [ ] RED: Write test for database URL absolute path conversion
- [ ] GREEN: Implement validator

---

## Phase 2: Infrastructure - Database & External APIs

**Objective:** Build thread-safe persistence and resilient external API client.  
**Estimated Effort:** 6-8 hours  
**Dependencies:** Phase 1 complete  
**Validation:** Integration tests pass, database operations work under load.

### 2.1 Database Infrastructure

#### Files to Create:

**1. `src/neural_terminal/infrastructure/database.py`**

**Features:**
- SQLAlchemy 2.0 `DeclarativeBase`
- `ConversationORM` and `MessageORM` models
- Engine with `check_same_thread=False` for Streamlit
- Event listener for `PRAGMA foreign_keys=ON` and `PRAGMA journal_mode=WAL`
- `scoped_session` for thread safety
- `get_db_session()` context manager

**TDD Checklist:**
- [ ] RED: Write test for table creation
- [ ] GREEN: Implement ORM models
- [ ] RED: Write test for foreign key constraint enforcement
- [ ] GREEN: Implement event listener correctly
- [ ] RED: Write test for concurrent session access
- [ ] GREEN: Verify scoped_session isolation

**2. `alembic/` configuration**

**Files:**
- `alembic.ini` - configuration
- `alembic/env.py` - environment with proper imports
- `alembic/versions/001_initial.py` - initial migration

**TDD Checklist:**
- [ ] RED: Verify `alembic revision --autogenerate` works
- [ ] GREEN: Fix any import issues
- [ ] RED: Verify `alembic upgrade head` creates tables
- [ ] GREEN: Run migration and verify schema

### 2.2 Repository Pattern

#### Files to Create:

**1. `src/neural_terminal/infrastructure/repositories.py`**

**Features:**
- `ConversationRepository` ABC with all required methods
- `SQLiteConversationRepository` implementation
- `_session_scope()` context manager for atomic operations
- `_to_domain()` and `_message_to_domain()` converters
- `get_messages()` for conversation history (required by orchestrator)

**TDD Checklist:**
- [ ] RED: Write test for save and get_by_id roundtrip
- [ ] GREEN: Implement repository
- [ ] RED: Write test for get_messages ordering
- [ ] GREEN: Implement get_messages
- [ ] RED: Write test for list_active pagination
- [ ] GREEN: Implement pagination
- [ ] RED: Write test for cascading operations
- [ ] GREEN: Verify foreign key behavior

### 2.3 Circuit Breaker

#### Files to Create:

**1. `src/neural_terminal/infrastructure/circuit_breaker.py`**

**Features:**
- `CircuitState` Enum (CLOSED, OPEN, HALF_OPEN)
- Thread-safe with `threading.Lock()`
- `call()` for sync functions
- `call_async()` for async functions
- `_check_state()` for manual state verification (used by streaming)
- Configurable failure threshold and recovery timeout

**TDD Checklist:**
- [ ] RED: Write test for circuit remaining closed on success
- [ ] GREEN: Implement success path
- [ ] RED: Write test for circuit opening after failures
- [ ] GREEN: Implement failure counting
- [ ] RED: Write test for automatic recovery attempt
- [ ] GREEN: Implement HALF_OPEN state
- [ ] RED: Write test for thread safety under concurrent access
- [ ] GREEN: Add locks and verify

### 2.4 OpenRouter Client

#### Files to Create:

**1. `src/neural_terminal/infrastructure/openrouter.py`**

**Features:**
- `OpenRouterModel` Pydantic model with pricing properties
- `OpenRouterClient` with httpx.AsyncClient
- `get_available_models()` with caching consideration
- `chat_completion_stream()` returning AsyncGenerator (not wrapped in circuit)
- Proper error handling for HTTP status codes and timeouts
- `json` import included

**TDD Checklist:**
- [ ] RED: Write test with respx mocking for model fetching
- [ ] GREEN: Implement get_available_models
- [ ] RED: Write test for SSE streaming with mocked chunks
- [ ] GREEN: Implement chat_completion_stream
- [ ] RED: Write test for error handling (429, 503, timeout)
- [ ] GREEN: Implement error translation to OpenRouterAPIError
- [ ] RED: Write test for JSON parsing errors in SSE
- [ ] GREEN: Implement graceful error handling

### 2.5 Token Counter

#### Files to Create:

**1. `src/neural_terminal/infrastructure/token_counter.py`**

**Features:**
- Tiktoken integration with encoding caching
- Model-aware encoding selection (cl100k_base for GPT/Claude)
- `count_message()` for single message
- `count_messages()` for conversation history
- `truncate_context()` with sliding window strategy (keep system + recent)

**TDD Checklist:**
- [ ] RED: Write test for token counting accuracy vs known values
- [ ] GREEN: Implement count_message
- [ ] RED: Write test for context truncation
- [ ] GREEN: Implement truncate_context
- [ ] RED: Write test for system message preservation
- [ ] GREEN: Verify system message kept during truncation

---

## Phase 3: Application Layer - Events, Cost Tracking & Orchestration

**Objective:** Build event-driven business logic and conversation orchestration.  
**Estimated Effort:** 6-8 hours  
**Dependencies:** Phase 2 complete  
**Validation:** Integration tests pass, event flow works correctly.

### 3.1 Event System

#### Files to Create:

**1. `src/neural_terminal/application/events.py`**

**Features:**
- `DomainEvent` frozen dataclass (event_type, conversation_id, payload)
- `EventObserver` ABC with `on_event()` method
- `EventBus` with typed subscriber management
- Event type constants in `Events` class

**Interface Definition:**
```python
class Events:
    MESSAGE_STARTED = "message.started"
    TOKEN_GENERATED = "token.generated"
    MESSAGE_COMPLETED = "message.completed"
    BUDGET_THRESHOLD = "budget.threshold"
    BUDGET_EXCEEDED = "budget.exceeded"
    CONTEXT_TRUNCATED = "context.truncated"
```

**TDD Checklist:**
- [ ] RED: Write test for event emission to subscriber
- [ ] GREEN: Implement EventBus
- [ ] RED: Write test for global subscribers
- [ ] GREEN: Implement subscribe_all
- [ ] RED: Write test for error isolation (one subscriber fails, others get event)
- [ ] GREEN: Implement error handling

### 3.2 Cost Tracker

#### Files to Create:

**1. `src/neural_terminal/application/cost_tracker.py`**

**Features:**
- Implements `EventObserver` interface
- Injected `EventBus` for emitting budget events (not creating new instances)
- Real-time estimation during streaming (every 100 tokens)
- Final reconciliation with actual usage
- Budget threshold at 80% (warning) and 100% (exceeded)

**TDD Checklist:**
- [ ] RED: Write test for MESSAGE_STARTED initialization
- [ ] GREEN: Implement on_event handler
- [ ] RED: Write test for TOKEN_GENERATED estimation
- [ ] GREEN: Implement estimation logic
- [ ] RED: Write test for MESSAGE_COMPLETED reconciliation
- [ ] GREEN: Implement actual cost calculation
- [ ] RED: Write test for budget threshold event at 80%
- [ ] GREEN: Implement threshold checking
- [ ] RED: Write test for budget exceeded event
- [ ] GREEN: Implement exceeded logic

### 3.3 Session State Manager

#### Files to Create:

**1. `src/neural_terminal/application/state.py`**

**Features:**
- `AppState` dataclass for type-safe state container
- `StateManager` with namespace isolation (`neural_terminal_`)
- `update()` for atomic state updates
- `set_conversation()` with proper serialization (using `to_dict()`)
- `get_cached_conversation()` with deserialization
- Stream buffer management for streaming state

**TDD Checklist:**
- [ ] RED: Write test for state initialization
- [ ] GREEN: Implement StateManager
- [ ] RED: Write test for atomic update
- [ ] GREEN: Implement update method
- [ ] RED: Write test for conversation serialization/deserialization
- [ ] GREEN: Implement with proper type handling
- [ ] RED: Write test for stream buffer append/clear
- [ ] GREEN: Implement buffer management

### 3.4 Chat Orchestrator

#### Files to Create:

**1. `src/neural_terminal/application/orchestrator.py`**

**Features:**
- Dependency injection of repository, openrouter, event_bus, token_counter
- `load_models()` for fetching available models
- `create_conversation()` with optional system prompt
- `send_message()` returning AsyncGenerator of (delta, metadata)
- Context window management with truncation
- Circuit breaker state check before streaming (not wrapping async generator)
- Error handling with partial message persistence

**Key Fix (C-4):** Manual circuit state check, then direct streaming without circuit wrapper

```python
async def send_message(self, conversation_id: UUID, content: str, temperature: float = 0.7):
    # Check circuit state manually
    self._circuit._check_state()
    
    try:
        async for chunk in self._openrouter.chat_completion_stream(...):
            # ... yield deltas ...
        self._circuit._on_success()
    except Exception as e:
        self._circuit._on_failure()
        raise
```

**TDD Checklist:**
- [ ] RED: Write test for conversation creation
- [ ] GREEN: Implement create_conversation
- [ ] RED: Write test for message sending with mocked stream
- [ ] GREEN: Implement send_message
- [ ] RED: Write test for context truncation when over limit
- [ ] GREEN: Implement truncation logic
- [ ] RED: Write test for circuit breaker integration
- [ ] GREEN: Implement manual circuit checks
- [ ] RED: Write test for error handling with partial persistence
- [ ] GREEN: Implement error recovery

---

## Phase 4: UI Components - Terminal Aesthetic & Layout

**Objective:** Implement "The Neural Terminal" phosphor-green aesthetic with proper sanitization.  
**Estimated Effort:** 8-10 hours  
**Dependencies:** Phase 3 complete  
**Validation:** Visual regression tests pass, XSS vulnerabilities eliminated.

### 4.1 Design Tokens & CSS

#### Files to Create:

**1. `src/neural_terminal/styles/theme.py`**

**Features:**
- CSS variables for phosphor/void/amber palette
- Typography: IBM Plex Mono + Instrument Sans (not Space Grotesk per critique)
- Global reset for Streamlit chrome hiding
- Custom widget styling (input, button, select)
- Scrollbar styling
- Animation keyframes (fadeIn, cursor-blink)

**TDD Checklist:**
- [ ] RED: Verify CSS injects without syntax errors
- [ ] GREEN: Fix any CSS issues
- [ ] Verify contrast ratios meet WCAG AA
- [ ] Verify animations are performant

### 4.2 Component Library

#### Files to Create:

**1. `src/neural_terminal/components/renderers.py`** (FIXED per S-1)

**Features:**
- `render_message()` with brutalist log-entry aesthetic
- **XSS Protection:** Bleach sanitization on all content
- Role-based styling (User=amber, AI=green)
- Metadata display (cost, latency)
- Timestamp formatting

```python
import bleach

ALLOWED_TAGS = ["code", "pre", "b", "i", "em", "strong", "br", "p", "ul", "ol", "li"]
ALLOWED_ATTRS = {"code": ["class"]}

def render_message(role: str, content: str, cost=None, latency=None):
    safe_content = bleach.clean(content, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
    # ... render with safe_content ...
```

**TDD Checklist:**
- [ ] RED: Write test for XSS payload sanitization
- [ ] GREEN: Implement bleach sanitization
- [ ] RED: Write test for allowed HTML passthrough
- [ ] GREEN: Verify allowed tags work
- [ ] RED: Write test for script tag removal
- [ ] GREEN: Verify dangerous content stripped

**2. `src/neural_terminal/components/telemetry.py`**

**Features:**
- Budget gauge with percentage bar
- Cost accumulator display
- Token velocity metric
- Active model display
- Conversation archive list

**TDD Checklist:**
- [ ] RED: Write test for budget percentage calculation
- [ ] GREEN: Implement gauge
- [ ] RED: Write test for cost formatting
- [ ] GREEN: Implement cost display

### 4.3 Streaming Bridge

#### Files to Create:

**1. `src/neural_terminal/components/stream_bridge.py`** (FIXED per H-3)

**Features:**
- `StreamlitStreamBridge` class for async-to-sync bridging
- Queue-based producer-consumer pattern
- Threading for non-blocking async execution
- `run_async()` helper for safe coroutine execution in Streamlit
- Error handling with queue propagation

**Key Fix (H-3):** Use threading-based bridge instead of `asyncio.run()` inside Streamlit

```python
def run_async(coro):
    """Execute async coroutine safely within Streamlit."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # Run in thread to avoid nested loop issues
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = asyncio.run(coro)
            except Exception as e:
                exception[0] = e
        
        t = threading.Thread(target=target)
        t.start()
        t.join()
        
        if exception[0]:
            raise exception[0]
        return result[0]
    else:
        return asyncio.run(coro)
```

**TDD Checklist:**
- [ ] RED: Write test for async generator consumption
- [ ] GREEN: Implement bridge
- [ ] RED: Write test for error propagation
- [ ] GREEN: Implement error handling
- [ ] RED: Write test for run_async with nested loop
- [ ] GREEN: Implement threading fallback

---

## Phase 5: Integration - Streamlit App & Streaming Bridge

**Objective:** Wire all components together into cohesive Streamlit application.  
**Estimated Effort:** 6-8 hours  
**Dependencies:** Phase 4 complete  
**Validation:** End-to-end tests pass, full chat flow works.

### 5.1 Main Application

#### Files to Create:

**1. `src/neural_terminal/app.py`** (FIXED per ST-2, H-3)

**Features:**
- `@st.cache_resource` for singleton services (not session_state)
- Proper service initialization order
- `run_async()` helper usage (not raw `asyncio.run()`)
- Header with terminal aesthetic
- Sidebar with telemetry panel
- Main chat canvas with message history
- Chat input with streaming response
- Error boundaries with user-friendly messages

**Key Fix (ST-2):** Use `@st.cache_resource` for service singletons

```python
@st.cache_resource
def get_openrouter() -> OpenRouterClient:
    return OpenRouterClient()

@st.cache_resource
def get_orchestrator(_event_bus: EventBus, _openrouter: OpenRouterClient) -> ChatOrchestrator:
    return ChatOrchestrator(...)
```

**TDD Checklist:**
- [ ] RED: Write integration test for full chat flow
- [ ] GREEN: Implement app.py
- [ ] RED: Write test for conversation switching
- [ ] GREEN: Implement history loading
- [ ] RED: Write test for error display
- [ ] GREEN: Implement error boundaries
- [ ] RED: Write test for streaming display
- [ ] GREEN: Implement streaming UI

---

## Phase 6: Production Hardening - Testing & Deployment

**Objective:** Containerize, add observability, and comprehensive testing.  
**Estimated Effort:** 6-8 hours  
**Dependencies:** Phase 5 complete  
**Validation:** Docker builds, E2E tests pass, deployment ready.

### 6.1 Testing Infrastructure

#### Files to Create:

**1. `tests/conftest.py`**

**Features:**
- Pytest fixtures for database, repository, event bus
- Test database isolation (in-memory SQLite)
- Mock OpenRouter client fixture
- Async test configuration

**2. `tests/unit/` test files**

| File | Coverage |
|------|----------|
| `test_config.py` | Settings loading, validation |
| `test_models.py` | Domain model behavior |
| `test_circuit_breaker.py` | State transitions, thread safety |
| `test_token_counter.py` | Counting accuracy, truncation |
| `test_cost_tracker.py` | Event handling, budget checks |
| `test_repositories.py` | CRUD operations, pagination |

**3. `tests/integration/` test files**

| File | Coverage |
|------|----------|
| `test_database.py` | Migration, constraints |
| `test_openrouter.py` | API client with mocking |
| `test_streaming.py` | End-to-end streaming flow |

**4. `tests/e2e/test_chat_flow.py`**

**Features:**
- Full conversation flow
- Model switching
- Cost tracking accuracy
- Error recovery

### 6.2 Containerization

#### Files to Create:

**1. `Dockerfile`**

**Features:**
- Multi-stage build
- Python 3.11 slim base
- Poetry for dependency management
- Non-root user
- Health check
- Size target: <200MB

**TDD Checklist:**
- [ ] RED: Verify `docker build` succeeds
- [ ] GREEN: Fix any build issues
- [ ] RED: Verify `docker run` starts app
- [ ] GREEN: Fix runtime issues
- [ ] Verify image size < 200MB

**2. `docker-compose.yml`**

**Features:**
- App service with volume for SQLite
- Environment file support
- Port mapping

### 6.3 Observability

#### Files to Create:

**1. `src/neural_terminal/infrastructure/logging.py`**

**Features:**
- Structlog configuration
- JSON formatting for production
- PII redaction filters
- Correlation ID injection

**TDD Checklist:**
- [ ] RED: Write test for JSON log output
- [ ] GREEN: Implement structured logging
- [ ] RED: Write test for PII redaction
- [ ] GREEN: Implement redaction filters

---

## Validation Checkpoints by Phase

### Phase 0 Validation
- [ ] All unit tests pass
- [ ] No runtime crashes on basic operations
- [ ] Foreign keys enforced in SQLite
- [ ] Circuit breaker thread-safe

### Phase 1 Validation
- [ ] `poetry install` succeeds
- [ ] `make test` runs without errors
- [ ] `make lint` passes (no type errors)
- [ ] Database migrations run successfully

### Phase 2 Validation
- [ ] Repository CRUD operations work
- [ ] Foreign key constraints enforced
- [ ] Circuit breaker transitions correctly
- [ ] OpenRouter client handles errors gracefully
- [ ] Token counting accurate vs known values

### Phase 3 Validation
- [ ] Event bus delivers events to all subscribers
- [ ] Cost tracker emits budget events correctly
- [ ] Orchestrator streams messages
- [ ] Context truncation works under load

### Phase 4 Validation
- [ ] XSS payloads sanitized
- [ ] Visual design matches spec
- [ ] Contrast ratios meet WCAG AA
- [ ] Animations performant (60fps)

### Phase 5 Validation
- [ ] Full chat flow works end-to-end
- [ ] Conversation history loads correctly
- [ ] Streaming displays in real-time
- [ ] Cost tracking accurate within $0.0001

### Phase 6 Validation
- [ ] Docker image builds and runs
- [ ] E2E tests pass
- [ ] Image size < 200MB
- [ ] Structured logging works

---

## Success Criteria (From Original Design)

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Message Delivery | 99.9% | 1000 test messages |
| Time-to-First-Token | <800ms | Average over 100 calls |
| Full Render (1000 tokens) | <50ms | Browser performance API |
| Cost Tracking Accuracy | ±$0.0001 | Compare to OpenRouter dashboard |
| Lighthouse Best Practices | >95 | Chrome DevTools |
| Lighthouse Accessibility | >95 | Chrome DevTools |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Streamlit version breaks CSS | High | Medium | Pin to `streamlit==1.28.x`, visual regression tests |
| OpenRouter API changes | Medium | High | Circuit breaker, graceful degradation |
| SQLite concurrency issues | Medium | High | WAL mode, scoped sessions, connection pooling |
| XSS via LLM output | Medium | Critical | Bleach sanitization on all outputs |
| Token cost estimation drift | Medium | Medium | Hybrid estimation, reconciliation at completion |
| Nested event loop issues | High | High | Use `run_async()` helper, thread-based bridge |

---

## Next Steps

1. **Review this plan** - Validate phase breakdown and dependencies
2. **Confirm environment** - Ensure Python 3.11+, Poetry installed
3. **Obtain OpenRouter API key** - Required for integration tests
4. **Begin Phase 0** - Critical bug fixes

---

## Appendices

### A. Design Document Cross-Reference

| This Plan | Source Document | Section |
|-----------|-----------------|---------|
| Phase 0 fixes | `new_design_critique_2.md` | Critical Defects C-1 to C-7 |
| Architecture | `new_design_1.md` | Phase 2: Architectural Blueprint |
| Sprint 1 | `new_design_2.md` | Day 1-3 |
| Sprint 2 | `new_design_3.md` | Day 4-6 |
| Sprint 3 | `new_design_4.md` | Avant-Garde UI |
| Sprint 4 | `new_design_5.md` | Production Hardening |
| CSS Strategy | `new_design_critique.md` | Phosphor Grid |

### B. Key Technical Decisions

1. **Circuit Breaker + Streaming:** Manual state checks instead of wrapping async generators (C-4)
2. **Session Management:** `_session_scope()` context manager instead of broken get/close pattern (C-3)
3. **Async in Streamlit:** Thread-based bridge instead of `asyncio.run()` (H-3)
4. **Service Singletons:** `@st.cache_resource` instead of session_state (ST-2)
5. **XSS Protection:** Bleach sanitization on all LLM outputs (S-1)
6. **Layout:** Hybrid approach (st.columns + custom HTML) instead of CSS Grid hijack (ST-1)

---

*Plan Version: 1.0*  
*Generated: 2026-02-15*  
*Status: Ready for Execution*
