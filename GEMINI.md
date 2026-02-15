# Neural Terminal - Project Briefing Document

> **Version:** 0.1.0  
> **Last Updated:** 2026-02-15  
> **Status:** Production-Ready

---

## 1. Project Overview

### What is Neural Terminal?

Neural Terminal is a **production-grade chatbot interface** with OpenRouter integration, featuring a distinctive terminal/cyberpunk aesthetic. It provides a Streamlit-based web interface for interacting with multiple AI models through a unified API.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Support** | 8 AI models via OpenRouter (GPT-4, Claude 3, Gemini, Llama 2, Mistral) |
| **Real-time Streaming** | SSE-based streaming responses with live token generation |
| **Cost Tracking** | Per-message and session cost calculation with budget enforcement |
| **Persistent Conversations** | SQLite-backed conversation history with full-text search |
| **Terminal Aesthetic** | 3 themes (Terminal Green, Cyberpunk Amber, Minimal Dark) |
| **XSS Protection** | Bleach-based HTML sanitization for safe message rendering |
| **Circuit Breaker** | Resilience pattern for API failures with automatic recovery |
| **Context Management** | Automatic truncation with token counting via tiktoken |

### Technology Stack

- **Framework:** Streamlit 1.54.0
- **Language:** Python 3.11+
- **HTTP Client:** httpx 0.28.1 (async)
- **Database:** SQLite + SQLAlchemy 2.0 + Alembic
- **Configuration:** Pydantic Settings
- **Token Counting:** tiktoken
- **Security:** Bleach (XSS protection)
- **Deployment:** Docker + Docker Compose

---

## 2. Architecture

### Clean Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                        UI Layer                              │
│  (Streamlit Components - themes, chat, header, status)      │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                         │
│  (Orchestrator, Events, State Management, Cost Tracking)    │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                       │
│  (Database, Repositories, OpenRouter Client, Circuit Breaker)│
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                            │
│  (Models, Exceptions, Business Rules)                       │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Dependency Inversion:** Domain layer has no external dependencies
2. **Repository Pattern:** Abstract data access behind interfaces
3. **Event-Driven:** Decoupled components via EventBus
4. **Circuit Breaker:** Fail-fast for external API resilience
5. **Thread Safety:** All state mutations use locking

---

## 3. Directory Structure

```
neural-terminal/
├── app.py                          # Streamlit entry point
├── pyproject.toml                  # Poetry dependencies & tool config
├── poetry.lock                     # Locked dependency versions
├── Makefile                        # Development commands
├── docker-compose.yml              # Container orchestration
├── Dockerfile                      # Multi-stage production build
├── docker-entrypoint.sh            # Container startup script
├── .env.example                    # Environment variable template
├── README.md                       # User documentation
├── LICENSE / MIT                   # Dual licensing
│
├── src/neural_terminal/
│   ├── __init__.py
│   ├── main.py                     # NeuralTerminalApp orchestration
│   ├── app_state.py                # Global ApplicationState singleton
│   ├── config.py                   # Pydantic Settings
│   │
│   ├── domain/                     # Core business logic
│   │   ├── __init__.py
│   │   ├── models.py               # Conversation, Message, TokenUsage
│   │   └── exceptions.py           # 12 custom exception types
│   │
│   ├── infrastructure/             # External concerns
│   │   ├── __init__.py
│   │   ├── database.py             # SQLAlchemy 2.0 ORM setup
│   │   ├── repositories.py         # SQLiteConversationRepository
│   │   ├── openrouter.py           # Async OpenRouter API client
│   │   ├── circuit_breaker.py      # Thread-safe circuit breaker
│   │   └── token_counter.py        # Tiktoken wrapper
│   │
│   ├── application/                # Use cases & coordination
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # ChatOrchestrator service
│   │   ├── events.py               # EventBus & DomainEvent
│   │   ├── state.py                # StateManager (Streamlit wrapper)
│   │   └── cost_tracker.py         # Budget tracking observer
│   │
│   └── components/                 # UI components
│       ├── __init__.py
│       ├── themes.py               # 3 theme definitions
│       ├── styles.py               # CSS generation & injection
│       ├── message_renderer.py     # XSS-safe rendering
│       ├── chat_container.py       # Message display component
│       ├── header.py               # Terminal header + sidebar
│       ├── status_bar.py           # Cost & connection status
│       ├── error_handler.py        # User-friendly errors
│       └── stream_bridge.py        # Async-to-sync bridge
│
├── scripts/                        # Utility scripts
│   ├── init_db.py                  # Production DB initialization
│   └── health_check.py             # Database health monitoring
│
├── tests/                          # Test suite (~330 tests)
│   ├── conftest.py                 # Pytest fixtures
│   ├── unit/                       # 18 test files
│   │   ├── test_models.py
│   │   ├── test_exceptions.py
│   │   ├── test_repositories.py
│   │   ├── test_orchestrator.py
│   │   ├── test_events.py
│   │   ├── test_state.py
│   │   ├── test_app_state.py
│   │   ├── test_cost_tracker.py
│   │   ├── test_token_counter.py
│   │   ├── test_circuit_breaker.py
│   │   ├── test_stream_bridge.py
│   │   ├── test_config.py
│   │   └── components/
│   │       ├── test_themes.py
│   │       ├── test_styles.py
│   │       ├── test_message_renderer.py
│   │       ├── test_chat_container.py
│   │       ├── test_header.py
│   │       ├── test_status_bar.py
│   │       └── test_error_handler.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_database.py
│   └── e2e/
│       └── __init__.py
│
└── alembic/                        # Database migrations
    ├── alembic.ini
    ├── env.py
    ├── script.py.mako
    └── versions/
```

---

## 4. Key Components

### 4.1 Domain Models (`src/neural_terminal/domain/models.py`)

| Class | Purpose | Key Attributes |
|-------|---------|----------------|
| `MessageRole` | Enum: user/assistant/system | - |
| `ConversationStatus` | Enum: active/archived/forked | - |
| `TokenUsage` | Immutable token metrics | `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `Message` | Domain entity for chat messages | `id`, `role`, `content`, `token_usage`, `cost`, `latency_ms` |
| `Conversation` | Aggregate root | `id`, `title`, `total_cost`, `total_tokens`, `update_cost()` |

**Phase 0 Defect Fixes in Models:**
- C-1: `TokenUsage.calculate_cost()` converted from property to method
- H-1: `Conversation.update_cost()` uses simple assignment
- H-4: Added `to_dict()` for JSON serialization

### 4.2 Domain Exceptions (`src/neural_terminal/domain/exceptions.py`)

**Hierarchy:**
```
NeuralTerminalError (base)
├── ConfigurationError
├── ValidationError
│   ├── InputTooLongError
│   └── EmptyInputError
├── CircuitBreakerOpenError
├── APIError
│   ├── OpenRouterAPIError
│   │   ├── RateLimitError
│   │   ├── ModelUnavailableError
│   │   └── TokenLimitError
├── ServiceError
│   ├── ConversationNotFoundError
│   └── MessageNotFoundError
├── BudgetError
│   └── BudgetExceededError
```

All exceptions include:
- `message`: Human-readable description
- `code`: Machine-readable error code (e.g., `HTTP_429`, `CIRCUIT_OPEN`)

### 4.3 Infrastructure Layer

#### Database (`infrastructure/database.py`)

**ORM Models:**
- `ConversationORM`: Maps to `conversations` table
- `MessageORM`: Maps to `messages` table

**Production Optimizations (PRAGMAs):**
```sql
PRAGMA foreign_keys=ON          -- Enforce FK constraints
PRAGMA journal_mode=WAL         -- Write-Ahead Logging
PRAGMA synchronous=NORMAL       -- Balance safety/performance
PRAGMA cache_size=-64000        -- 64MB cache
PRAGMA temp_store=MEMORY        -- Fast temp operations
PRAGMA mmap_size=268435456      -- 256MB memory-mapped I/O
```

**Phase 0 Defect C-2 Fix:**
- Engine created BEFORE event listener
- Listen on ENGINE INSTANCE, not `create_engine` function

#### Repositories (`infrastructure/repositories.py`)

**Pattern:** Abstract base + SQLite implementation

**SQLiteConversationRepository Methods:**
- `get_by_id(conversation_id)` → Optional[Conversation]
- `get_messages(conversation_id)` → List[Message] *(ordered by created_at ASC)*
- `save(conversation)` → Upsert via `session.merge()`
- `add_message(message)` → Insert with validation
- `list_active(limit, offset)` → List by updated_at DESC

**Phase 0 Defect C-3 Fix:**
- Replaced broken context manager with `_session_scope()`
- Proper `SessionLocal.remove()` cleanup in finally block
- Added `_message_to_domain()` converter

#### OpenRouter Client (`infrastructure/openrouter.py`)

**Capabilities:**
- Async streaming via `chat_completion_stream()`
- SSE parsing with `json` module
- Error handling: 429 (RateLimit), 503 (ModelUnavailable), 400 (TokenLimit)
- Pricing model fetching via `get_available_models()`

**Phase 0 Defects Fixed:**
- C-4: Yields directly; connection errors surface before first yield
- C-5: `json` module imported at top of file

#### Circuit Breaker (`infrastructure/circuit_breaker.py`)

**States:** CLOSED → OPEN → HALF_OPEN → CLOSED

**Configuration:**
- `failure_threshold`: 5 failures before opening (default)
- `recovery_timeout`: 30 seconds before HALF_OPEN

**Thread Safety:**
- `threading.Lock()` around all state mutations (Phase 0 Defect H-2 Fix)
- `_check_state()` for manual circuit verification (Phase 0 Defect C-4)

**Usage Patterns:**
- Sync: `circuit.call(func, *args, **kwargs)`
- Async: `await circuit.call_async(func, *args, **kwargs)`
- Streaming: Manual `_check_state()` → iterate → `_on_success()`/`_on_failure()`

#### Token Counter (`infrastructure/token_counter.py`)

**Features:**
- Model-aware encoding via tiktoken
- Context truncation with system message preservation
- Caching of encoder instances

**Supported Encodings:**
- GPT-4/GPT-3.5: `cl100k_base`
- Claude: `cl100k_base` (approximation)
- Default: `cl100k_base`

**Truncation Strategy:**
1. Always keep system message (if first)
2. Keep recent messages from end
3. Add truncation marker if messages dropped

### 4.4 Application Layer

#### ChatOrchestrator (`application/orchestrator.py`)

**Responsibilities:**
- Conversation lifecycle management
- Message streaming coordination
- Cost calculation with pricing data
- Context length management

**Key Method: `send_message()`**
```python
async def send_message(
    conversation_id: UUID,
    content: str,
    temperature: float = 0.7
) -> AsyncGenerator[Tuple[str, Optional[dict]], None]
```

Yields: `(delta_text, None)` for streaming, `("", metadata)` for completion

**Phase 0 Defect C-4 Implementation:**
- Manual circuit breaker check before streaming
- Direct async generator iteration
- Manual success/failure recording

#### Event System (`application/events.py`)

**Event Types:**
```python
class Events:
    MESSAGE_STARTED = "message.started"
    TOKEN_GENERATED = "token.generated"
    MESSAGE_COMPLETED = "message.completed"
    BUDGET_THRESHOLD = "budget.threshold"
    BUDGET_EXCEEDED = "budget.exceeded"
    CONTEXT_TRUNCATED = "context.truncated"
```

**Pattern:** Observer with typed subscribers + global subscribers
- Error isolation: subscriber failures don't stop propagation
- Thread-safe event emission

#### CostTracker (`application/cost_tracker.py`)

**Implements:** `EventObserver`

**Features:**
- Real-time cost estimation during streaming
- Budget threshold warnings (80%)
- Budget exceeded enforcement

**Phase 0 Defect C-6 Fix:**
- EventBus injected via constructor (not created as orphan)

### 4.5 UI Components

#### Themes (`components/themes.py`)

**Available Themes:**

| Theme | Mode | Primary Accent | Font Stack |
|-------|------|----------------|------------|
| Terminal Green | TERMINAL | `#00FF41` (Matrix) | JetBrains Mono, Fira Code |
| Cyberpunk Amber | TERMINAL | `#FFB000` (Amber) | VT323, Share Tech Mono |
| Minimal Dark | DARK | `#569CD6` (Blue) | SF Mono, Consolas |

**Theme Structure:**
- `ColorPalette`: 17 color tokens
- `Typography`: 6 font sizes + line height
- `Spacing`: 6 scale tokens (xs to xxl)
- `Effects`: glow, scanlines, cursor blink, CRT flicker

#### Styles (`components/styles.py`)

**CSS Generation Functions:**
- `generate_base_css()`: Streamlit component overrides
- `generate_terminal_effects_css()`: Glow, cursor, scanlines
- `generate_message_css()`: Message bubble styling
- `generate_header_css()`: Header component styles
- `generate_input_css()`: Input area styling
- `generate_status_bar_css()`: Status bar styles

**Injection:** `inject_css(theme)` - Uses `st.markdown(unsafe_allow_html=True)`

#### MessageRenderer (`components/message_renderer.py`)

**Security:**
- Bleach sanitization with allowlists
- HTML escaping for inline content
- Safe markdown processing

**Allowed Tags:**
```python
['p', 'br', 'strong', 'em', 'code', 'pre', 'ul', 'ol', 'li',
 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote',
 'hr', 'table', 'thead', 'tbody', 'tr', 'th', 'td']
```

**Code Block Detection:**
- Fenced blocks: ```language\ncode\n```
- Inline code: `code`
- Language aliases (py→python, js→javascript, etc.)

#### Stream Bridge (`components/stream_bridge.py`)

**Problem:** Streamlit runs its own event loop; `asyncio.run()` conflicts

**Solution:** `run_async(coro)`
- Detects running loop
- If in loop: runs coroutine in thread
- If no loop: uses `asyncio.run()` directly

**StreamlitStreamBridge:**
- Producer-consumer pattern with threading
- Queue-based async-to-sync bridge
- Error propagation from async generator

---

## 5. Database Schema

### Tables

#### conversations

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| id | UUID | PRIMARY KEY | Unique identifier |
| title | VARCHAR(255) | NULLABLE | User-defined title |
| model_id | VARCHAR(100) | NOT NULL | Default model for conversation |
| status | Enum | DEFAULT 'active' | active/archived/forked |
| created_at | DateTime | DEFAULT utcnow | Creation timestamp |
| updated_at | DateTime | DEFAULT utcnow | Last update timestamp |
| total_cost | Numeric(10,6) | DEFAULT 0 | Accumulated cost |
| total_tokens | Integer | DEFAULT 0 | Accumulated tokens |
| parent_conversation_id | UUID | FK→conversations.id | For forking |
| tags | JSON | DEFAULT [] | Array of string tags |

#### messages

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| id | UUID | PRIMARY KEY | Unique identifier |
| conversation_id | UUID | FK→conversations.id, NOT NULL | Parent conversation |
| role | Enum | NOT NULL | user/assistant/system |
| content | TEXT | NOT NULL | Message text |
| prompt_tokens | Integer | NULLABLE | Input token count |
| completion_tokens | Integer | NULLABLE | Output token count |
| total_tokens | Integer | NULLABLE | Total tokens |
| cost | Numeric(10,6) | NULLABLE | Calculated cost |
| latency_ms | Integer | NULLABLE | Response time |
| model_id | VARCHAR(100) | NULLABLE | Model used |
| created_at | DateTime | DEFAULT utcnow | Timestamp |
| metadata | JSON | DEFAULT {} | Additional data |

### Indexes

```sql
-- Performance indexes
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);
CREATE INDEX idx_conversations_updated_at ON conversations(updated_at DESC);
CREATE INDEX idx_conversations_status ON conversations(status);
CREATE INDEX idx_conversations_model_id ON conversations(model_id);
```

### Relationships

- `ConversationORM.messages` → `MessageORM` (1:N, cascade delete)
- `MessageORM.conversation` → `ConversationORM` (N:1)

---

## 6. Testing Status

### Test Statistics

| Metric | Value |
|--------|-------|
| Total Test Files | 20 |
| Total Test Lines | ~4,710 |
| Source Lines | ~5,865 |
| Test Ratio | ~0.80:1 |

### Test Organization

```
tests/
├── unit/               # 18 files - Component isolation tests
│   ├── Domain tests (models, exceptions)
│   ├── Infrastructure tests (repositories, circuit breaker)
│   ├── Application tests (orchestrator, events, state)
│   └── Component tests (themes, rendering, UI)
├── integration/        # Database integration tests
└── e2e/               # End-to-end tests (placeholder)
```

### Coverage Areas

| Layer | Coverage | Notes |
|-------|----------|-------|
| Domain Models | ✅ High | Full property and method testing |
| Exceptions | ✅ High | All exception types |
| Repositories | ✅ High | SQLite with in-memory DB |
| Circuit Breaker | ✅ High | State transitions, thread safety |
| Token Counter | ✅ High | Truncation, counting accuracy |
| Cost Tracker | ✅ High | Event-based tracking |
| Events | ✅ High | Bus, subscribers, error isolation |
| State Management | ✅ High | Session state, persistence |
| UI Components | ✅ Medium | Theme rendering, message display |

### Running Tests

```bash
# All tests with coverage
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# With HTML coverage report
make test-coverage  # Output: htmlcov/index.html
```

### Coverage Configuration (pyproject.toml)

```toml
[tool.coverage.run]
source = ["src/neural_terminal"]
branch = true
fail_under = 90
```

---

## 7. Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key |
| `DATABASE_URL` | No | `sqlite:///neural_terminal.db` | SQLite database path |
| `DEFAULT_MODEL` | No | `openai/gpt-3.5-turbo` | Default AI model |
| `BUDGET_LIMIT` | No | None | Budget limit in USD |
| `APP_ENV` | No | `development` | Environment name |
| `LOG_LEVEL` | No | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `OPENROUTER_TIMEOUT` | No | 60 | API timeout (10-300s) |
| `CIRCUIT_FAILURE_THRESHOLD` | No | 5 | Failures before circuit opens |
| `CIRCUIT_RECOVERY_TIMEOUT` | No | 30 | Seconds before recovery attempt |

### Configuration Class (`config.py`)

```python
class Settings(BaseSettings):
    openrouter_api_key: SecretStr
    database_url: str = "sqlite:///neural_terminal.db"
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # ...
```

**Validation:**
- `database_url`: Must start with `sqlite://`
- Relative paths auto-converted to absolute
- SecretStr for API key (never logged/displayed)

### Available Models

| Model ID | Display Name |
|----------|--------------|
| `openai/gpt-4-turbo` | GPT-4 Turbo |
| `openai/gpt-4` | GPT-4 |
| `openai/gpt-3.5-turbo` | GPT-3.5 Turbo (default) |
| `anthropic/claude-3-opus` | Claude 3 Opus |
| `anthropic/claude-3-sonnet` | Claude 3 Sonnet |
| `google/gemini-pro` | Gemini Pro |
| `meta-llama/llama-2-70b-chat` | Llama 2 70B |
| `mistral/mistral-medium` | Mistral Medium |

---

## 8. Deployment

### Docker Architecture

**Multi-stage build:**
1. **Builder stage:** Install Poetry, compile dependencies
2. **Runtime stage:** Copy packages, run as non-root user

**Security Features:**
- Non-root user (`neural`, UID 1000)
- `no-new-privileges:true`
- Health check endpoint
- Read-only root (writable `/app/data` volume)

### Docker Commands

```bash
# Build image
make docker-build

# Run container
make docker-run OPENROUTER_API_KEY=sk-...

# Docker Compose
make docker-compose-up   # Start with persistent storage
make docker-compose-down # Stop
```

### Docker Compose Configuration

```yaml
services:
  neural-terminal:
    image: neural-terminal:latest
    ports:
      - "8501:8501"
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - DATABASE_URL=sqlite:////app/data/neural_terminal.db
    volumes:
      - neural_terminal_data:/app/data
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
```

### Persistent Storage

**Volume:** `neural_terminal_data` (named Docker volume)

**Contents:**
- `/app/data/neural_terminal.db` - Main database
- `/app/data/neural_terminal.db-wal` - WAL file
- `/app/data/neural_terminal.db-shm` - Shared memory

---

## 9. Scripts

### init_db.py

**Purpose:** Production database initialization

**Features:**
- Database integrity validation (`PRAGMA integrity_check`)
- Production optimizations (WAL, cache, mmap)
- Index creation
- Schema verification against ORM models
- Backup creation
- Statistics reporting

**Usage:**
```bash
python scripts/init_db.py              # Initialize
python scripts/init_db.py --backup     # Backup before init
python scripts/init_db.py --vacuum     # Vacuum after init
python scripts/init_db.py --stats      # Show statistics only
```

### health_check.py

**Purpose:** Database health monitoring

**Checks:**
- Integrity (corruption detection)
- Foreign key violations
- File sizes (warn if >1GB)
- WAL size (warn if >100MB)
- Performance metrics (conversation/message counts)
- Configuration validation

**Usage:**
```bash
python scripts/health_check.py              # Human-readable output
python scripts/health_check.py --json       # JSON output
python scripts/health_check.py --exit-code  # Exit 1 if unhealthy
```

**Thresholds:**
- Max DB size: 1GB
- Max WAL size: 100MB
- Max conversations: 10,000
- Max messages/conversation: 1,000

---

## 10. Known Issues & Technical Debt

### Phase 0 Defects (Documented in Code)

All documented as inline comments with `Phase 0 Defect X-X Fix/Note`:

| Defect | Location | Status | Description |
|--------|----------|--------|-------------|
| C-1 | domain/models.py | ✅ Fixed | `TokenUsage.cost` property → method |
| C-2 | infrastructure/database.py | ✅ Fixed | Event listener on engine instance |
| C-3 | infrastructure/repositories.py | ✅ Fixed | Proper session scope context manager |
| C-4 | infrastructure/openrouter.py | ✅ Fixed | Async generator yield pattern |
| C-5 | infrastructure/openrouter.py | ✅ Fixed | `json` import at top level |
| C-6 | application/cost_tracker.py | ✅ Fixed | EventBus injection (not orphan) |
| H-1 | domain/models.py | ✅ Fixed | `update_cost()` simple assignment |
| H-2 | infrastructure/circuit_breaker.py | ✅ Fixed | Threading lock for state |
| H-3 | components/stream_bridge.py | ✅ Fixed | Async-to-sync bridge pattern |
| H-4 | domain/models.py | ✅ Fixed | `to_dict()` for serialization |
| H-5 | infrastructure/repositories.py | ✅ Fixed | Added `get_messages()` method |

### Current Limitations

1. **Database:** SQLite only (PostgreSQL support planned)
2. **Conversation Delete:** Soft-delete not implemented (only removes from list)
3. **Model Pricing:** Cached on load, not refreshed during session
4. **Search:** No full-text search on conversation content
5. **Export:** No conversation export functionality
6. **Multi-user:** Single-user design (no authentication system)

### Development Notes

- **Streamlit Limitations:** Requires async-to-sync bridge for streaming
- **Thread Safety:** All state mutations use `threading.Lock()`
- **Event Loop:** Cannot use `asyncio.run()` inside Streamlit; use `run_async()`
- **Session State:** JSON serialization requires Decimal/UUID conversion

### Performance Considerations

- Message history limited to 100 visible messages in UI
- Context truncation at 4096 tokens (configurable per model)
- Database connection pooling via SQLAlchemy
- WAL mode for concurrent read/write
- Token encoder caching in TokenCounter

---

## 11. Development Commands

### Setup
```bash
poetry install          # Install dependencies
poetry install --with dev  # Include dev dependencies
```

### Quality
```bash
make lint               # Ruff + MyPy
make format             # Black + Ruff fix
make format-check       # Check formatting
```

### Testing
```bash
make test               # Run all tests with coverage
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-coverage      # HTML coverage report
```

### Database
```bash
make db-init            # Initialize for production
make db-backup          # Create backup
make db-vacuum          # Reclaim space
make db-stats           # Show statistics
make db-health          # Health check
```

### Application
```bash
make run                # Run Streamlit app
poetry run streamlit run app.py
```

---

## 12. File Inventory

### Source Files (~5,865 lines)

| Path | Lines | Purpose |
|------|-------|---------|
| `domain/models.py` | 163 | Domain entities |
| `domain/exceptions.py` | 188 | Custom exceptions |
| `infrastructure/database.py` | 166 | ORM setup |
| `infrastructure/repositories.py` | 220 | Data access |
| `infrastructure/openrouter.py` | 188 | API client |
| `infrastructure/circuit_breaker.py` | 167 | Resilience |
| `infrastructure/token_counter.py` | 165 | Token counting |
| `application/orchestrator.py` | 320 | Chat service |
| `application/events.py` | 109 | Event bus |
| `application/state.py` | 150 | Session state |
| `application/cost_tracker.py` | 147 | Budget tracking |
| `components/themes.py` | 340 | Theme system |
| `components/styles.py` | 703 | CSS generation |
| `components/message_renderer.py` | 528 | XSS-safe rendering |
| `components/chat_container.py` | 308 | Message display |
| `components/header.py` | 256 | Header + sidebar |
| `components/status_bar.py` | 251 | Status display |
| `components/error_handler.py` | 340 | Error display |
| `components/stream_bridge.py` | 156 | Async bridge |
| `app_state.py` | 433 | Global state |
| `config.py` | 57 | Settings |
| `main.py` | 451 | App orchestration |

### Test Files (~4,710 lines)

| Path | Lines | Coverage |
|------|-------|----------|
| `unit/test_models.py` | ~250 | Domain models |
| `unit/test_exceptions.py` | ~150 | Exceptions |
| `unit/test_repositories.py` | ~400 | Repository pattern |
| `unit/test_orchestrator.py` | ~500 | Chat orchestration |
| `unit/test_events.py` | ~200 | Event system |
| `unit/test_state.py` | ~300 | State management |
| `unit/test_app_state.py` | ~400 | Application state |
| `unit/test_cost_tracker.py` | ~250 | Budget tracking |
| `unit/test_token_counter.py` | ~300 | Token counting |
| `unit/test_circuit_breaker.py` | ~350 | Circuit breaker |
| `unit/test_stream_bridge.py` | ~200 | Async bridge |
| `unit/test_config.py` | ~150 | Configuration |
| `unit/components/*.py` | ~1,260 | UI components |
| `integration/test_database.py` | ~200 | DB integration |

---

## 13. Quick Reference

### Entry Points

| Entry Point | File | Purpose |
|-------------|------|---------|
| Streamlit App | `app.py` | Main application entry |
| Docker | `docker-entrypoint.sh` | Container commands |
| DB Init | `scripts/init_db.py` | Production setup |
| Health | `scripts/health_check.py` | Monitoring |

### Key Classes

| Class | Module | Responsibility |
|-------|--------|----------------|
| `NeuralTerminalApp` | `main.py` | Main application |
| `ApplicationState` | `app_state.py` | Global state singleton |
| `ChatOrchestrator` | `application/orchestrator.py` | Chat service |
| `EventBus` | `application/events.py` | Event coordination |
| `SQLiteConversationRepository` | `infrastructure/repositories.py` | Data access |
| `OpenRouterClient` | `infrastructure/openrouter.py` | API client |
| `CircuitBreaker` | `infrastructure/circuit_breaker.py` | Resilience |
| `MessageRenderer` | `components/message_renderer.py` | Safe rendering |

### External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ^1.54.0 | Web framework |
| httpx | ^0.28.1 | Async HTTP |
| sqlalchemy | ^2.0.46 | ORM |
| alembic | ^1.18.4 | Migrations |
| pydantic | ^2.12.5 | Validation |
| tiktoken | ^0.12.0 | Token counting |
| bleach | ^6.3.0 | XSS protection |
| markdown | ^3.10.2 | Rendering |

---

*End of Document*

**Maintained by:** Neural Terminal Team  
**Repository:** https://github.com/example/neural-terminal  
**License:** MIT
