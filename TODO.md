# Neural Terminal - Implementation TODO

## Legend
- [ ] Not started
- [/] In progress
- [x] Complete
- [!] Blocked/Issue

---

## Phase 0: Critical Bug Fixes & Architecture Corrections

### 0.1 TokenUsage.cost Property Fix (C-1) ✅ COMPLETE
- [x] RED: Write test for TokenUsage.calculate_cost() with sample pricing
- [x] GREEN: Convert `cost` property to `calculate_cost()` method
- [x] REFACTOR: Verify all usages updated
- [x] Validate: Test passes, no property usage remains

### 0.2 Database Foreign Keys Fix (C-2) ✅ COMPLETE
- [x] RED: Write test verifying foreign keys are enforced
- [x] GREEN: Fix missing imports (Column, Text, datetime)
- [x] GREEN: Fix event listener to target ENGINE INSTANCE
- [x] GREEN: Add PRAGMA journal_mode=WAL
- [x] Validate: PRAGMA foreign_keys returns 1

### 0.3 Repository Session Leak Fix (C-3) ✅ COMPLETE
- [x] RED: Write test for get_messages() returning ordered messages
- [x] GREEN: Implement _session_scope() context manager
- [x] GREEN: Implement get_messages() method
- [x] GREEN: Add _message_to_domain() converter
- [x] GREEN: Use SessionLocal.remove() for cleanup
- [x] Validate: No session leaks under load test

### 0.4 Circuit Breaker + Async Fix (C-4) ✅ COMPLETE
- [x] RED: Write test for circuit state check before streaming
- [x] GREEN: Add _check_state() method to CircuitBreaker
- [x] GREEN: Add threading.Lock() for thread safety (H-2)
- [x] Validate: Streaming works without await on async generator

### 0.5 Missing JSON Import Fix (C-5)
- [ ] RED: Verify json.loads usage in openrouter.py
- [ ] GREEN: Add `import json` to openrouter.py
- [ ] Validate: No NameError on JSON parsing

### 0.6 CostTracker EventBus Fix (C-6)
- [ ] RED: Write test verifying budget events emitted to same bus
- [ ] GREEN: Inject EventBus in CostTracker constructor
- [ ] GREEN: Replace EventBus() calls with self._bus
- [ ] Validate: Budget events received by subscribers

### 0.7 Circuit Breaker Test Fix (C-7) ✅ COMPLETE
- [x] RED: Write test with proper pytest.raises context managers
- [x] GREEN: Fix test_circuit_opens_after_threshold test
- [x] Validate: All circuit breaker tests pass

### Phase 0 Validation
- [ ] All unit tests pass
- [ ] No runtime crashes on import
- [ ] make lint passes

---

## Phase 1: Foundation - Configuration & Domain Layer

### 1.1 Project Setup
- [ ] Create pyproject.toml with Poetry config
- [ ] Add all dependencies (streamlit, httpx, pydantic, sqlalchemy, alembic, tiktoken, bleach, structlog)
- [ ] Add dev dependencies (pytest, pytest-asyncio, pytest-cov, respx, mypy, ruff, black)
- [ ] Create .env.example
- [ ] Create Makefile with install, test, lint, format, migrate, run targets
- [ ] Validate: poetry install succeeds

### 1.2 Domain Exceptions
- [ ] RED: Write test for exception hierarchy
- [ ] GREEN: Create NeuralTerminalError base class
- [ ] GREEN: Create CircuitBreakerOpenError
- [ ] GREEN: Create OpenRouterAPIError with status_code
- [ ] GREEN: Create ValidationError
- [ ] Validate: All exceptions have proper attributes

### 1.3 Domain Models
- [ ] RED: Write test for MessageRole Enum
- [ ] GREEN: Implement MessageRole
- [ ] RED: Write test for ConversationStatus Enum
- [ ] GREEN: Implement ConversationStatus
- [ ] RED: Write test for TokenUsage.calculate_cost
- [ ] GREEN: Implement TokenUsage with calculate_cost method
- [ ] RED: Write test for Message dataclass
- [ ] GREEN: Implement Message
- [ ] RED: Write test for Conversation.update_cost
- [ ] GREEN: Implement Conversation with update_cost
- [ ] RED: Write test for Conversation.to_dict serialization
- [ ] GREEN: Implement to_dict with Decimal/UUID handling
- [ ] Validate: All model tests pass

### 1.4 Configuration
- [ ] RED: Write test for settings loading from env
- [ ] GREEN: Create Settings class with Pydantic
- [ ] GREEN: Add SecretStr for API key
- [ ] GREEN: Add field validators
- [ ] GREEN: Add db_path property
- [ ] Validate: Settings load correctly from environment

### Phase 1 Validation
- [ ] make test passes
- [ ] make lint passes (no type errors)
- [ ] make format runs successfully

---

## Phase 2: Infrastructure - Database & External APIs

### 2.1 Database Infrastructure
- [ ] RED: Write test for Base metadata
- [ ] GREEN: Create Base with DeclarativeBase
- [ ] GREEN: Create ConversationORM
- [ ] GREEN: Create MessageORM
- [ ] GREEN: Create engine with proper config
- [ ] GREEN: Add event listener for PRAGMAs
- [ ] GREEN: Create scoped_session
- [ ] GREEN: Create get_db_session context manager
- [ ] Validate: Database tables created successfully

### 2.2 Alembic Configuration
- [ ] GREEN: Initialize alembic (alembic init)
- [ ] GREEN: Configure alembic/env.py with proper imports
- [ ] GREEN: Create initial migration (001_initial.py)
- [ ] Validate: alembic upgrade head runs successfully

### 2.3 Repository Pattern
- [ ] RED: Write test for ConversationRepository ABC
- [ ] GREEN: Create ConversationRepository ABC
- [ ] RED: Write test for SQLiteConversationRepository.save
- [ ] GREEN: Implement save
- [ ] RED: Write test for get_by_id
- [ ] GREEN: Implement get_by_id
- [ ] RED: Write test for get_messages
- [ ] GREEN: Implement get_messages
- [ ] RED: Write test for add_message
- [ ] GREEN: Implement add_message
- [ ] RED: Write test for list_active
- [ ] GREEN: Implement list_active
- [ ] Validate: All repository tests pass

### 2.4 Circuit Breaker
- [ ] RED: Write test for CircuitState Enum
- [ ] GREEN: Create CircuitState Enum
- [ ] RED: Write test for circuit remaining closed on success
- [ ] GREEN: Implement __init__ and call method
- [ ] RED: Write test for circuit opening after failures
- [ ] GREEN: Implement failure counting and OPEN state
- [ ] RED: Write test for automatic recovery (HALF_OPEN)
- [ ] GREEN: Implement HALF_OPEN state
- [ ] RED: Write test for _check_state method
- [ ] GREEN: Implement _check_state
- [ ] RED: Write test for thread safety
- [ ] GREEN: Add threading.Lock()
- [ ] GREEN: Lock _on_success and _on_failure
- [ ] Validate: Thread-safe concurrent test passes

### 2.5 OpenRouter Client
- [ ] RED: Write test for OpenRouterModel
- [ ] GREEN: Create OpenRouterModel Pydantic model
- [ ] GREEN: Add prompt_price and completion_price properties
- [ ] RED: Write test for get_available_models
- [ ] GREEN: Implement get_available_models with respx mock
- [ ] RED: Write test for chat_completion_stream
- [ ] GREEN: Implement chat_completion_stream
- [ ] RED: Write test for SSE parsing
- [ ] GREEN: Implement SSE chunk parsing
- [ ] RED: Write test for error handling (429, 503)
- [ ] GREEN: Implement error translation
- [ ] RED: Write test for timeout handling
- [ ] GREEN: Implement timeout error
- [ ] Validate: All client tests pass

### 2.6 Token Counter
- [ ] RED: Write test for encoder caching
- [ ] GREEN: Implement encoder caching
- [ ] RED: Write test for count_message
- [ ] GREEN: Implement count_message
- [ ] RED: Write test for count_messages
- [ ] GREEN: Implement count_messages
- [ ] RED: Write test for truncate_context
- [ ] GREEN: Implement truncate_context
- [ ] RED: Write test for system message preservation
- [ ] GREEN: Ensure system message kept during truncation
- [ ] Validate: Token counting accurate vs known values

### Phase 2 Validation
- [ ] Repository integration tests pass
- [ ] OpenRouter client tests pass with mocked API
- [ ] Circuit breaker stress test passes

---

## Phase 3: Application Layer - Events, Cost Tracking & Orchestration

### 3.1 Event System
- [ ] RED: Write test for DomainEvent creation
- [ ] GREEN: Create DomainEvent dataclass
- [ ] RED: Write test for EventObserver ABC
- [ ] GREEN: Create EventObserver ABC
- [ ] RED: Write test for EventBus.subscribe
- [ ] GREEN: Implement subscribe
- [ ] RED: Write test for EventBus.subscribe_all
- [ ] GREEN: Implement subscribe_all
- [ ] RED: Write test for EventBus.emit to specific subscribers
- [ ] GREEN: Implement emit
- [ ] RED: Write test for EventBus.emit to global subscribers
- [ ] GREEN: Implement global emission
- [ ] RED: Write test for error isolation
- [ ] GREEN: Implement error handling in emit
- [ ] GREEN: Create Events constants class
- [ ] Validate: All event tests pass

### 3.2 Cost Tracker
- [ ] RED: Write test for CostTracker initialization
- [ ] GREEN: Implement __init__ with EventBus injection
- [ ] RED: Write test for MESSAGE_STARTED handling
- [ ] GREEN: Implement on_event MESSAGE_STARTED
- [ ] RED: Write test for TOKEN_GENERATED estimation
- [ ] GREEN: Implement estimation logic
- [ ] RED: Write test for MESSAGE_COMPLETED reconciliation
- [ ] GREEN: Implement actual cost calculation
- [ ] RED: Write test for 80% budget threshold (BUDGET_THRESHOLD event)
- [ ] GREEN: Implement threshold checking
- [ ] RED: Write test for budget exceeded (BUDGET_EXCEEDED event)
- [ ] GREEN: Implement exceeded logic
- [ ] RED: Write test for accumulated_cost property
- [ ] GREEN: Implement accumulated_cost
- [ ] Validate: All cost tracker tests pass

### 3.3 Session State Manager
- [ ] RED: Write test for AppState creation
- [ ] GREEN: Create AppState dataclass
- [ ] RED: Write test for StateManager initialization
- [ ] GREEN: Implement __init__ with namespace
- [ ] RED: Write test for state property retrieval
- [ ] GREEN: Implement state property
- [ ] RED: Write test for update method
- [ ] GREEN: Implement update
- [ ] RED: Write test for set_conversation serialization
- [ ] GREEN: Implement set_conversation
- [ ] RED: Write test for get_cached_conversation deserialization
- [ ] GREEN: Implement get_cached_conversation
- [ ] RED: Write test for clear_stream_buffer
- [ ] GREEN: Implement clear_stream_buffer
- [ ] RED: Write test for append_stream_buffer
- [ ] GREEN: Implement append_stream_buffer
- [ ] Validate: All state manager tests pass

### 3.4 Chat Orchestrator
- [ ] RED: Write test for __init__ dependency injection
- [ ] GREEN: Implement __init__
- [ ] RED: Write test for load_models
- [ ] GREEN: Implement load_models
- [ ] RED: Write test for get_model_config
- [ ] GREEN: Implement get_model_config
- [ ] RED: Write test for create_conversation
- [ ] GREEN: Implement create_conversation
- [ ] RED: Write test for send_message basic flow
- [ ] GREEN: Implement send_message structure
- [ ] RED: Write test for context truncation
- [ ] GREEN: Implement context window management
- [ ] RED: Write test for circuit breaker integration
- [ ] GREEN: Implement manual circuit checks
- [ ] RED: Write test for streaming event emission
- [ ] GREEN: Implement event emission during stream
- [ ] RED: Write test for error handling with partial message
- [ ] GREEN: Implement error recovery
- [ ] Validate: Orchestrator integration tests pass

### Phase 3 Validation
- [ ] Event flow integration test passes
- [ ] Cost tracking accuracy test passes
- [ ] Streaming flow test passes

---

## Phase 4: UI Components - Terminal Aesthetic & Layout

### 4.1 Design Tokens & CSS
- [ ] GREEN: Create styles/theme.py
- [ ] GREEN: Define CSS variables (void, phosphor, amber)
- [ ] GREEN: Add typography (IBM Plex Mono, Instrument Sans)
- [ ] GREEN: Add global reset for Streamlit chrome
- [ ] GREEN: Add widget styling
- [ ] GREEN: Add scrollbar styling
- [ ] GREEN: Add animation keyframes
- [ ] Validate: CSS injects without errors

### 4.2 Message Renderer (with XSS Protection)
- [ ] RED: Write test for XSS payload sanitization
- [ ] GREEN: Add bleach import and ALLOWED_TAGS/ATTRS
- [ ] GREEN: Implement sanitize_content function
- [ ] GREEN: Create render_message function
- [ ] RED: Write test for allowed HTML passthrough
- [ ] GREEN: Verify allowed tags work
- [ ] RED: Write test for script tag removal
- [ ] GREEN: Verify dangerous content stripped
- [ ] RED: Write test for role-based styling
- [ ] GREEN: Implement amber for user, green for AI
- [ ] RED: Write test for metadata display (cost, latency)
- [ ] GREEN: Implement metadata display
- [ ] Validate: XSS test suite passes

### 4.3 Telemetry Panel
- [ ] RED: Write test for budget gauge calculation
- [ ] GREEN: Create render_telemetry_panel
- [ ] GREEN: Implement budget percentage bar
- [ ] GREEN: Implement cost accumulator
- [ ] RED: Write test for token velocity metric
- [ ] GREEN: Implement token velocity display
- [ ] RED: Write test for active model display
- [ ] GREEN: Implement model display
- [ ] GREEN: Add conversation archive list
- [ ] Validate: Telemetry renders correctly

### 4.4 Streaming Bridge
- [ ] RED: Write test for StreamlitStreamBridge initialization
- [ ] GREEN: Create StreamlitStreamBridge class
- [ ] RED: Write test for async generator consumption
- [ ] GREEN: Implement stream method
- [ ] RED: Write test for queue-based communication
- [ ] GREEN: Implement producer-consumer pattern
- [ ] RED: Write test for error propagation
- [ ] GREEN: Implement error handling
- [ ] RED: Write test for run_async helper
- [ ] GREEN: Implement run_async with threading fallback
- [ ] RED: Write test for nested event loop handling
- [ ] GREEN: Verify thread-based execution
- [ ] Validate: Streaming bridge tests pass

### Phase 4 Validation
- [ ] Visual inspection of terminal aesthetic
- [ ] XSS penetration test passes
- [ ] Animation performance test (60fps)

---

## Phase 5: Integration - Streamlit App

### 5.1 Main Application
- [ ] GREEN: Create app.py structure
- [ ] GREEN: Add @st.cache_resource for get_openrouter
- [ ] GREEN: Add @st.cache_resource for get_event_bus
- [ ] GREEN: Add @st.cache_resource for get_orchestrator
- [ ] GREEN: Implement get_cost_tracker
- [ ] GREEN: Implement run_async helper
- [ ] GREEN: Create inject_theme function
- [ ] GREEN: Create render_header function
- [ ] GREEN: Create render_empty_state function
- [ ] RED: Write test for service initialization
- [ ] GREEN: Implement main() structure
- [ ] RED: Write test for conversation creation flow
- [ ] GREEN: Implement new session button
- [ ] RED: Write test for message sending flow
- [ ] GREEN: Implement chat input and streaming
- [ ] RED: Write test for conversation switching
- [ ] GREEN: Implement conversation list
- [ ] RED: Write test for error display
- [ ] GREEN: Add error boundaries
- [ ] Validate: Full chat flow works end-to-end

### Phase 5 Validation
- [ ] E2E test: Create conversation -> Send message -> Receive response
- [ ] E2E test: Switch conversations
- [ ] E2E test: Cost tracking accuracy
- [ ] Manual test: Streaming display smoothness

---

## Phase 6: Production Hardening

### 6.1 Testing Infrastructure
- [ ] GREEN: Create tests/conftest.py with fixtures
- [ ] GREEN: Add database fixture
- [ ] GREEN: Add repository fixture
- [ ] GREEN: Add event bus fixture
- [ ] GREEN: Add mock openrouter fixture
- [ ] GREEN: Create tests/unit/test_config.py
- [ ] GREEN: Create tests/unit/test_models.py
- [ ] GREEN: Create tests/unit/test_circuit_breaker.py
- [ ] GREEN: Create tests/unit/test_token_counter.py
- [ ] GREEN: Create tests/unit/test_cost_tracker.py
- [ ] GREEN: Create tests/unit/test_repositories.py
- [ ] GREEN: Create tests/integration/test_database.py
- [ ] GREEN: Create tests/integration/test_openrouter.py
- [ ] GREEN: Create tests/integration/test_streaming.py
- [ ] GREEN: Create tests/e2e/test_chat_flow.py
- [ ] Validate: Full test suite passes (pytest)

### 6.2 Containerization
- [ ] GREEN: Create Dockerfile
- [ ] GREEN: Use Python 3.11 slim base
- [ ] GREEN: Multi-stage build
- [ ] GREEN: Poetry for dependencies
- [ ] GREEN: Non-root user
- [ ] GREEN: Health check
- [ ] RED: Verify docker build succeeds
- [ ] GREEN: Fix any build issues
- [ ] RED: Verify docker run starts app
- [ ] GREEN: Fix any runtime issues
- [ ] Validate: Image size < 200MB

### 6.3 Docker Compose
- [ ] GREEN: Create docker-compose.yml
- [ ] GREEN: Add app service
- [ ] GREEN: Add volume for SQLite persistence
- [ ] GREEN: Add environment file support
- [ ] Validate: docker-compose up works

### 6.4 Observability
- [ ] RED: Write test for structured logging
- [ ] GREEN: Create logging.py with structlog
- [ ] GREEN: Add JSON formatter
- [ ] RED: Write test for PII redaction
- [ ] GREEN: Implement redaction filters
- [ ] RED: Write test for correlation ID
- [ ] GREEN: Add correlation ID injection
- [ ] Validate: Logs output JSON format

### Phase 6 Validation
- [ ] make test passes with >90% coverage
- [ ] make lint passes
- [ ] Docker image builds and runs
- [ ] E2E tests pass in container

---

## Final Validation

### Pre-Release Checklist
- [ ] All phases complete
- [ ] All tests passing
- [ ] No critical security vulnerabilities (bandit scan)
- [ ] No high-severity dependency issues (safety check)
- [ ] Documentation complete (README, ADRs)
- [ ] Docker image size verified < 200MB
- [ ] Performance benchmarks met:
  - [ ] Time-to-first-token < 800ms
  - [ ] Full render < 50ms for 1000 tokens
  - [ ] Cost accuracy ±$0.0001

---

## Current Status

**Overall Progress:** 0% (Planning Complete)  
**Current Phase:** 0 - Critical Bug Fixes  
**Next Action:** Begin Phase 0.1 - TokenUsage.cost property fix

---

*Last Updated: 2026-02-15*
