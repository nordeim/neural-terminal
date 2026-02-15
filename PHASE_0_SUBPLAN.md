# Phase 0: Critical Bug Fixes & Architecture Corrections
## Sub-Plan with Integrated Checklist

**Phase Objective:** Fix all 7 critical defects (C-1 through C-7) before any feature implementation.  
**Estimated Duration:** 2-3 hours  
**Success Criteria:** All unit tests pass, no runtime crashes on import, `make lint` passes with zero errors.  
**Dependencies:** None (foundation phase)

---

## Phase 0 Architecture Overview

```
Phase 0 Deliverables:
├── src/neural_terminal/
│   ├── domain/
│   │   └── models.py              [C-1, H-1, H-4] FIXED
│   ├── infrastructure/
│   │   ├── database.py            [C-2] FIXED
│   │   ├── repositories.py        [C-3, H-5] FIXED
│   │   ├── circuit_breaker.py     [H-2, C-4 prep] FIXED
│   │   └── openrouter.py          [C-4, C-5] FIXED
│   └── application/
│       └── cost_tracker.py        [C-6] FIXED
└── tests/
    └── unit/
        └── test_circuit_breaker.py [C-7] FIXED
```

---

## Defect 0.1: TokenUsage.cost Property Fix (C-1)

### Problem Analysis
```python
# BROKEN - Properties cannot accept arguments
@property
def cost(self, price_per_1k_prompt: Decimal, price_per_1k_completion: Decimal) -> Decimal:
    ...

# Calling usage.cost raises TypeError
```

### Solution Design
Convert to a regular method `calculate_cost()` with the same signature.

### Files to Modify

#### File: `src/neural_terminal/domain/models.py`

**Interface Specification:**
```python
@dataclass(frozen=True)
class TokenUsage:
    """Immutable token consumption metrics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    def calculate_cost(
        self, 
        price_per_1k_prompt: Decimal, 
        price_per_1k_completion: Decimal
    ) -> Decimal:
        """Calculate cost based on pricing - REGULAR METHOD, not property.
        
        Args:
            price_per_1k_prompt: Price per 1000 prompt tokens
            price_per_1k_completion: Price per 1000 completion tokens
            
        Returns:
            Total cost as Decimal
        """
        prompt_cost = (Decimal(self.prompt_tokens) / 1000) * price_per_1k_prompt
        completion_cost = (Decimal(self.completion_tokens) / 1000) * price_per_1k_completion
        return prompt_cost + completion_cost
```

**TDD Checklist - Defect 0.1:**
- [ ] **RED**: Create test file `tests/unit/test_models.py`
- [ ] **RED**: Write test `test_token_usage_calculate_cost()` with known values
  - Input: `TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)`
  - Pricing: $0.0015/1K prompt, $0.002/1K completion
  - Expected: $(1.0 * 0.0015) + (0.5 * 0.002) = $0.0025
- [ ] **GREEN**: Change `@property def cost(...)` to `def calculate_cost(...)`
- [ ] **GREEN**: Remove `@property` decorator
- [ ] **REFACTOR**: Verify method naming is clear and consistent
- [ ] **VALIDATE**: Test passes with exact decimal precision

**Edge Cases to Test:**
- [ ] Zero tokens (cost = 0)
- [ ] Large token counts (overflow protection)
- [ ] Decimal precision preservation (no float conversion)

---

## Defect 0.2: SQLite Foreign Keys Fix (C-2)

### Problem Analysis
```python
# BROKEN - Event listener targets the FUNCTION, not an ENGINE INSTANCE
@event.listens_for(create_engine, "connect")  # ← Wrong target!
def set_sqlite_pragma(dbapi_conn, connection_record):
    ...

# Also missing: Column, Text, datetime imports
```

### Solution Design
1. Add all missing imports
2. Create engine FIRST (at module level)
3. Listen on ENGINE INSTANCE, not the function
4. Add `PRAGMA journal_mode=WAL` for better concurrency
5. Use `DeclarativeBase` instead of `declarative_base()` (SQLAlchemy 2.0 style)

### Files to Modify

#### File: `src/neural_terminal/infrastructure/database.py`

**Interface Specification:**
```python
from datetime import datetime
from sqlalchemy import (
    JSON, Column, DateTime, Enum, ForeignKey, Integer, 
    Numeric, String, Text, Uuid, create_engine, event,
)
from sqlalchemy.orm import (
    DeclarativeBase, Session, relationship, 
    scoped_session, sessionmaker,
)

from neural_terminal.config import settings
from neural_terminal.domain.models import ConversationStatus, MessageRole


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 declarative base"""
    pass


# Create engine FIRST at module level
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # Required for Streamlit threads
    pool_pre_ping=True,
)


# Listen on the ENGINE INSTANCE, not the function
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable foreign key constraints and WAL mode on connection."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()
```

**TDD Checklist - Defect 0.2:**
- [ ] **RED**: Create test file `tests/integration/test_database.py`
- [ ] **RED**: Write test `test_foreign_keys_enabled()`
  - Query: `SELECT * FROM pragma_foreign_keys`
  - Expected: Returns `[(1,)]` (enabled)
- [ ] **GREEN**: Add missing imports (`Column`, `Text`, `datetime`)
- [ ] **GREEN**: Change `declarative_base()` to `DeclarativeBase`
- [ ] **GREEN**: Create engine at module level BEFORE event listener
- [ ] **GREEN**: Change `@event.listens_for(create_engine, "connect")` to `@event.listens_for(engine, "connect")`
- [ ] **GREEN**: Add `cursor.execute("PRAGMA journal_mode=WAL")`
- [ ] **RED**: Write test `test_cascading_delete()`
  - Create conversation with messages
  - Delete conversation
  - Verify messages also deleted (foreign key cascade)
- [ ] **GREEN**: Verify cascade behavior works
- [ ] **REFACTOR**: Ensure consistent import ordering
- [ ] **VALIDATE**: Both tests pass, foreign keys enforced

**Edge Cases to Test:**
- [ ] Foreign key violation raises proper error
- [ ] WAL mode active (check `PRAGMA journal_mode`)
- [ ] Multiple connections don't deadlock

---

## Defect 0.3: Repository Session Leak Fix (C-3)

### Problem Analysis
```python
# BROKEN - Creates orphaned context manager, never properly closed
def _get_session(self) -> Session:
    return get_db_session().__enter__()  # ← Orphaned!

def _close_session(self, session: Session) -> None:
    get_db_session().__exit__(None, None, None)  # ← Different context manager!
```

### Solution Design
Replace broken pattern with proper `_session_scope()` context manager using `@contextmanager` decorator.

### Files to Modify

#### File: `src/neural_terminal/infrastructure/repositories.py`

**Interface Specification:**
```python
from contextlib import contextmanager
from typing import Generator, List, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from neural_terminal.domain.models import (
    Conversation, ConversationStatus, Message, MessageRole, TokenUsage,
)
from neural_terminal.infrastructure.database import (
    ConversationORM, MessageORM, SessionLocal,
)


class ConversationRepository(ABC):
    @abstractmethod
    def get_by_id(self, conversation_id: UUID) -> Optional[Conversation]: ...
    
    @abstractmethod
    def get_messages(self, conversation_id: UUID) -> List[Message]: ...  # NEW METHOD
    
    @abstractmethod
    def save(self, conversation: Conversation) -> None: ...
    
    @abstractmethod
    def add_message(self, message: Message) -> None: ...
    
    @abstractmethod
    def list_active(self, limit: int = 50, offset: int = 0) -> List[Conversation]: ...


class SQLiteConversationRepository(ConversationRepository):
    """Thread-safe repository using scoped sessions."""
    
    @contextmanager
    def _session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations.
        
        Yields:
            SQLAlchemy Session
            
        Ensures:
            - Commit on success
            - Rollback on exception
            - Session cleanup via scoped_session.remove()
        """
        session = SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            SessionLocal.remove()  # Critical for scoped_session cleanup
    
    def _to_domain(self, orm: ConversationORM) -> Conversation:
        """Convert ORM to domain model."""
        return Conversation(
            id=orm.id,
            title=orm.title,
            model_id=orm.model_id,
            status=orm.status,
            created_at=orm.created_at,
            updated_at=orm.updated_at,
            total_cost=orm.total_cost or Decimal("0"),
            total_tokens=orm.total_tokens or 0,
            parent_conversation_id=orm.parent_conversation_id,
            tags=orm.tags or [],
        )
    
    def _message_to_domain(self, orm: MessageORM) -> Message:
        """Convert Message ORM to domain model."""
        usage = None
        if orm.prompt_tokens is not None:
            usage = TokenUsage(
                prompt_tokens=orm.prompt_tokens,
                completion_tokens=orm.completion_tokens or 0,
                total_tokens=orm.total_tokens or 0,
            )
        return Message(
            id=orm.id,
            conversation_id=orm.conversation_id,
            role=orm.role,
            content=orm.content,
            token_usage=usage,
            cost=orm.cost,
            latency_ms=orm.latency_ms,
            model_id=orm.model_id,
            created_at=orm.created_at,
            metadata=orm.metadata or {},
        )
    
    def get_by_id(self, conversation_id: UUID) -> Optional[Conversation]:
        with self._session_scope() as session:
            result = session.execute(
                select(ConversationORM).where(ConversationORM.id == conversation_id)
            ).scalar_one_or_none()
            return self._to_domain(result) if result else None
    
    def get_messages(self, conversation_id: UUID) -> List[Message]:
        """Retrieve all messages for a conversation, ordered by creation time."""
        with self._session_scope() as session:
            results = session.execute(
                select(MessageORM)
                .where(MessageORM.conversation_id == conversation_id)
                .order_by(MessageORM.created_at.asc())
            ).scalars().all()
            return [self._message_to_domain(r) for r in results]
    
    def save(self, conversation: Conversation) -> None:
        with self._session_scope() as session:
            orm = ConversationORM(
                id=conversation.id,
                title=conversation.title,
                model_id=conversation.model_id,
                status=conversation.status,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                total_cost=conversation.total_cost,
                total_tokens=conversation.total_tokens,
                parent_conversation_id=conversation.parent_conversation_id,
                tags=conversation.tags,
            )
            session.merge(orm)
    
    def add_message(self, message: Message) -> None:
        if message.conversation_id is None:
            raise ValueError("Message must belong to a conversation")
        
        with self._session_scope() as session:
            orm = MessageORM(
                id=message.id,
                conversation_id=message.conversation_id,
                role=message.role,
                content=message.content,
                prompt_tokens=message.token_usage.prompt_tokens if message.token_usage else None,
                completion_tokens=message.token_usage.completion_tokens if message.token_usage else None,
                total_tokens=message.token_usage.total_tokens if message.token_usage else None,
                cost=message.cost,
                latency_ms=message.latency_ms,
                model_id=message.model_id,
                metadata=message.metadata,
            )
            session.add(orm)
    
    def list_active(self, limit: int = 50, offset: int = 0) -> List[Conversation]:
        with self._session_scope() as session:
            results = session.execute(
                select(ConversationORM)
                .where(ConversationORM.status == ConversationStatus.ACTIVE)
                .order_by(ConversationORM.updated_at.desc())
                .limit(limit)
                .offset(offset)
            ).scalars().all()
            return [self._to_domain(r) for r in results]
```

**TDD Checklist - Defect 0.3:**
- [ ] **RED**: Write test `test_get_messages_ordered()`
  - Create conversation with 3 messages at different times
  - Call get_messages
  - Verify returned in created_at ascending order
- [ ] **GREEN**: Add `get_messages()` method to ABC
- [ ] **GREEN**: Implement `_session_scope()` with `@contextmanager`
- [ ] **GREEN**: Implement `get_messages()` with proper ordering
- [ ] **GREEN**: Implement `_message_to_domain()` converter
- [ ] **GREEN**: Update all existing methods to use `_session_scope()`
- [ ] **GREEN**: Remove broken `_get_session()` and `_close_session()` methods
- [ ] **GREEN**: Ensure `SessionLocal.remove()` called in finally block
- [ ] **RED**: Write test `test_concurrent_repository_access()`
  - Spawn multiple threads accessing repository simultaneously
  - Verify no connection pool exhaustion
- [ ] **GREEN**: Verify scoped_session isolation works
- [ ] **REFACTOR**: Clean up imports and type hints
- [ ] **VALIDATE**: All repository tests pass, no session leaks

**Edge Cases to Test:**
- [ ] Empty conversation returns empty message list
- [ ] Message without token usage handles None gracefully
- [ ] Concurrent saves don't corrupt data

---

## Defect 0.4: Circuit Breaker + Async Fix (C-4)

### Problem Analysis
```python
# BROKEN - Cannot await an AsyncGenerator
stream_gen = await self._circuit.call_async(
    self._openrouter.chat_completion,  # ← Returns AsyncGenerator
    ...
)
# TypeError: object async_generator can't be used in 'await' expression
```

### Solution Design
1. Split `chat_completion` into `chat_completion_stream` that yields directly
2. Add `_check_state()` method to CircuitBreaker for manual state verification
3. In orchestrator: check circuit state, then stream directly, manually record success/failure

### Files to Modify

#### File: `src/neural_terminal/infrastructure/circuit_breaker.py`

**Additions:**
```python
def _check_state(self) -> None:
    """Check if circuit allows operation.
    
    Raises:
        CircuitBreakerOpenError: If circuit is open and recovery timeout hasn't elapsed
    """
    if self._state == CircuitState.OPEN:
        if time.time() - (self._last_failure_time or 0) > self.recovery_timeout:
            self._state = CircuitState.HALF_OPEN
        else:
            raise CircuitBreakerOpenError(
                f"Circuit is OPEN. Retry after {self.recovery_timeout}s"
            )
```

#### File: `src/neural_terminal/infrastructure/openrouter.py`

**Interface Specification:**
```python
async def chat_completion_stream(
    self,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Streaming chat completion with SSE parsing.
    
    Yields chunks during streaming. Connection errors surface before first yield.
    Final yield contains usage metadata.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model ID (e.g., 'openai/gpt-3.5-turbo')
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Yields:
        Dict with keys:
        - 'type': 'delta' | 'final'
        - 'content': str (for delta)
        - 'usage': TokenUsage (for final)
        - 'latency_ms': int (for final)
    """
    import json  # C-5 FIX: Add missing import
    
    client = await self._get_client()
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    
    start_time = time.time()
    
    async with client.stream("POST", "/chat/completions", json=payload) as response:
        # Connection errors surface HERE, before first yield
        if response.status_code >= 400:
            body = await response.aread()
            raise OpenRouterAPIError(
                message=f"OpenRouter error: {body.decode()}",
                status_code=response.status_code,
                response_body=body.decode(),
            )
        
        full_content = ""
        usage: Optional[TokenUsage] = None
        
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            
            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                
                if content:
                    full_content += content
                    yield {"type": "delta", "content": content}
                
                if "usage" in chunk:
                    u = chunk["usage"]
                    usage = TokenUsage(
                        prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0),
                        total_tokens=u.get("total_tokens", 0),
                    )
            except json.JSONDecodeError:
                continue
        
        latency = int((time.time() - start_time) * 1000)
        yield {
            "type": "final",
            "content": full_content,
            "usage": usage,
            "latency_ms": latency,
            "model": model,
        }
```

#### File: `src/neural_terminal/application/orchestrator.py`

**Update to `send_message` method:**
```python
async def send_message(
    self,
    conversation_id: UUID,
    content: str,
    temperature: float = 0.7
) -> AsyncGenerator[Tuple[str, Optional[dict]], None]:
    """Send message and stream response."""
    # ... setup code ...
    
    # Check circuit state manually (don't wrap the async generator)
    self._circuit._check_state()
    
    try:
        async for chunk in self._openrouter.chat_completion_stream(
            messages=api_messages,
            model=conv.model_id,
            temperature=temperature,
        ):
            if chunk["type"] == "delta":
                delta = chunk["content"]
                assistant_content += delta
                
                # Emit token event for cost tracking
                self._event_bus.emit(DomainEvent(
                    event_type=Events.TOKEN_GENERATED,
                    conversation_id=conversation_id,
                    payload={"delta": delta}
                ))
                
                yield (delta, None)
            
            elif chunk["type"] == "final":
                # Record success
                self._circuit._on_success()
                # ... process final metadata ...
                yield ("", metadata)
    
    except Exception as e:
        # Record failure
        self._circuit._on_failure()
        # ... error handling ...
        raise
```

**TDD Checklist - Defect 0.4:**
- [ ] **RED**: Write test `test_circuit_breaker_check_state()`
  - When closed: no exception
  - When open and timeout elapsed: transitions to half_open
  - When open and timeout not elapsed: raises CircuitBreakerOpenError
- [ ] **GREEN**: Add `_check_state()` method to CircuitBreaker
- [ ] **RED**: Write test `test_chat_completion_stream_yields_chunks()` with mocked SSE
- [ ] **GREEN**: Implement `chat_completion_stream()` method
- [ ] **GREEN**: Add `import json` at top of openrouter.py (C-5 fix)
- [ ] **RED**: Write test for connection error before first yield
- [ ] **GREEN**: Ensure errors raised during `async with client.stream()`
- [ ] **RED**: Update orchestrator test to verify manual circuit checks
- [ ] **GREEN**: Update `send_message()` to use `_check_state()` and direct streaming
- [ ] **GREEN**: Add `self._circuit._on_success()` after successful stream
- [ ] **GREEN**: Add `self._circuit._on_failure()` in except block
- [ ] **REFACTOR**: Ensure clean separation of concerns
- [ ] **VALIDATE**: All streaming tests pass

**Edge Cases to Test:**
- [ ] Circuit open blocks streaming before connection
- [ ] Malformed SSE data handled gracefully
- [ ] Empty stream yields no deltas but still yields final

---

## Defect 0.5: Missing JSON Import Fix (C-5)

### Problem Analysis
```python
# BROKEN - json.loads used but json never imported
chunk = json.loads(data)  # NameError: name 'json' is not defined
```

### Solution Design
Add `import json` to `openrouter.py`. Already included in C-4 fix above.

**TDD Checklist - Defect 0.5:**
- [ ] **VALIDATE**: Verify `import json` present in openrouter.py (covered by C-4)

---

## Defect 0.6: CostTracker EventBus Fix (C-6)

### Problem Analysis
```python
# BROKEN - Creates new EventBus instance, events go nowhere
def _check_budget(self, estimated_cost: Decimal):
    bus = EventBus()  # ← Orphan instance!
    bus.emit(DomainEvent(...))
```

### Solution Design
Inject EventBus in constructor, use `self._bus.emit()` throughout.

### Files to Modify

#### File: `src/neural_terminal/application/cost_tracker.py`

**Interface Specification:**
```python
class CostTracker(EventObserver):
    """Real-time cost accumulator with budget enforcement.
    
    Implements Observer pattern for decoupled economic tracking.
    
    Args:
        event_bus: Shared EventBus instance for emitting budget events
        budget_limit: Optional budget limit in USD
    """
    
    def __init__(self, event_bus: EventBus, budget_limit: Optional[Decimal] = None):
        self._bus = event_bus  # Injected singleton - use this!
        self._accumulated = Decimal("0.00")
        self._budget_limit = budget_limit
        self._current_model_price: Optional[OpenRouterModel] = None
        self._estimated_tokens = 0
        self._is_tracking = False
    
    def _check_budget(self, estimated_cost: Decimal):
        """Check if approaching budget limit and emit events."""
        if not self._budget_limit:
            return
        
        projected = self._accumulated + estimated_cost
        
        if projected > self._budget_limit:
            self._bus.emit(DomainEvent(
                event_type=Events.BUDGET_EXCEEDED,
                payload={"accumulated": str(self._accumulated)}
            ))
        elif projected > self._budget_limit * Decimal("0.8"):
            self._bus.emit(DomainEvent(
                event_type=Events.BUDGET_THRESHOLD,
                payload={
                    "accumulated": str(self._accumulated),
                    "limit": str(self._budget_limit),
                }
            ))
    
    def _emit_budget_exceeded(self):
        """Emit budget exceeded event using injected bus."""
        self._bus.emit(DomainEvent(
            event_type=Events.BUDGET_EXCEEDED,
            payload={"accumulated": str(self._accumulated)}
        ))
```

**TDD Checklist - Defect 0.6:**
- [ ] **RED**: Write test `test_cost_tracker_injected_event_bus()`
  - Create shared EventBus
  - Create tracker with injected bus
  - Subscribe test observer to bus
  - Emit budget threshold condition
  - Verify test observer receives event
- [ ] **GREEN**: Modify `__init__` to accept `event_bus` parameter
- [ ] **GREEN**: Store as `self._bus`
- [ ] **GREEN**: Replace `EventBus()` with `self._bus` in `_check_budget()`
- [ ] **GREEN**: Replace `EventBus()` with `self._bus` in `_emit_budget_exceeded()`
- [ ] **RED**: Write test for 80% threshold warning
- [ ] **GREEN**: Verify BUDGET_THRESHOLD event emitted at 80%
- [ ] **RED**: Write test for budget exceeded
- [ ] **GREEN**: Verify BUDGET_EXCEEDED event emitted at 100%
- [ ] **REFACTOR**: Remove any remaining `from neural_terminal.application.events import EventBus` inside methods
- [ ] **VALIDATE**: All cost tracker tests pass, events received by shared bus

**Edge Cases to Test:**
- [ ] No budget limit set = no events emitted
- [ ] Exactly 80% triggers threshold
- [ ] Multiple threshold crossings only emit once per message

---

## Defect 0.7: Circuit Breaker Test Fix (C-7)

### Problem Analysis
```python
# BROKEN - Test crashes on first exception, never reaches circuit open
def test_circuit_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=2)
    def fail():
        raise ValueError("error")
    
    cb.call(fail)  # ← Raises ValueError, test crashes HERE
    cb.call(fail)  # ← Never reached
    # Circuit open never verified
```

### Solution Design
Wrap failing calls in `pytest.raises()` context managers.

### Files to Modify

#### File: `tests/unit/test_circuit_breaker.py`

**Interface Specification:**
```python
def test_circuit_opens_after_threshold():
    """Test that circuit opens after failure threshold reached."""
    cb = CircuitBreaker(failure_threshold=2)
    
    def fail():
        raise ValueError("error")
    
    # 1st failure - exception propagates, circuit still closed
    with pytest.raises(ValueError):
        cb.call(fail)
    
    # 2nd failure - exception propagates, circuit now open
    with pytest.raises(ValueError):
        cb.call(fail)
    
    # 3rd call - circuit is open, should raise CircuitBreakerOpenError
    with pytest.raises(CircuitBreakerOpenError):
        cb.call(fail)
```

**TDD Checklist - Defect 0.7:**
- [ ] **RED**: Run existing test, verify it fails as described
- [ ] **GREEN**: Wrap first `cb.call(fail)` in `with pytest.raises(ValueError)`
- [ ] **GREEN**: Wrap second `cb.call(fail)` in `with pytest.raises(ValueError)`
- [ ] **GREEN**: Wrap third `cb.call(fail)` in `with pytest.raises(CircuitBreakerOpenError)`
- [ ] **RED**: Add assertion verifying circuit state is OPEN after threshold
- [ ] **GREEN**: Add state assertion
- [ ] **REFACTOR**: Ensure test is clear and well-commented
- [ ] **VALIDATE**: Test passes, circuit behavior verified

---

## Phase 0 Integration Test

After all individual fixes, create a comprehensive integration test:

**File: `tests/integration/test_phase0_fixes.py`**

```python
"""Integration test verifying all Phase 0 critical fixes work together."""

async def test_end_to_end_conversation_flow():
    """Test complete flow: create conversation -> send message -> verify persistence."""
    # This test exercises:
    # - C-1: TokenUsage.calculate_cost()
    # - C-2: Foreign key constraints
    # - C-3: Repository session management
    # - C-4: Streaming with circuit breaker
    # - C-5: JSON parsing
    # - C-6: Cost tracking with shared event bus
    
    # Setup
    event_bus = EventBus()
    repo = SQLiteConversationRepository()
    client = OpenRouterClient()
    counter = TokenCounter()
    
    orchestrator = ChatOrchestrator(
        repository=repo,
        openrouter=client,
        event_bus=event_bus,
        token_counter=counter
    )
    
    # Create conversation
    conv = await orchestrator.create_conversation(
        title="Test",
        model_id="openai/gpt-3.5-turbo"
    )
    
    # Send message (mocked stream)
    # ... mock implementation ...
    
    # Verify:
    # - Message persisted
    # - Cost calculated correctly (C-1)
    # - Foreign keys enforced (C-2)
    # - No session leaks (C-3)
    # - Circuit breaker state correct (C-4)
    # - Events emitted to shared bus (C-6)
```

**TDD Checklist - Integration:**
- [ ] **RED**: Write integration test covering all 6 defects
- [ ] **GREEN**: Implement with mocked OpenRouter
- [ ] **VALIDATE**: Integration test passes

---

## Phase 0 Validation Criteria

### Pre-Validation Checklist
- [ ] All 7 defect fixes implemented
- [ ] Unit tests created for each fix
- [ ] Integration test created
- [ ] No `TODO` or `FIXME` comments remaining

### Validation Tests
- [ ] Run `python -c "from neural_terminal.domain.models import TokenUsage; print('C-1 OK')"`
- [ ] Run `python -c "from neural_terminal.infrastructure.database import engine; print('C-2 OK')"`
- [ ] Run `python -c "from neural_terminal.infrastructure.repositories import SQLiteConversationRepository; print('C-3 OK')"`
- [ ] Run `python -c "from neural_terminal.infrastructure.openrouter import OpenRouterClient; print('C-4, C-5 OK')"`
- [ ] Run `python -c "from neural_terminal.application.cost_tracker import CostTracker; print('C-6 OK')"`
- [ ] Run `pytest tests/unit/test_circuit_breaker.py -v` (C-7)
- [ ] Run `pytest tests/integration/test_phase0_fixes.py -v`

### Success Criteria
- [ ] All imports work without errors
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] No runtime crashes on basic operations
- [ ] Code passes `ruff check` with no errors
- [ ] Code passes `mypy --strict` with no errors

---

## Phase 0 Exit Criteria

Before proceeding to Phase 1, the following MUST be true:

1. **All 7 critical defects fixed and tested**
2. **Test coverage > 90% for modified files**
3. **Zero linting errors**
4. **Zero type checking errors**
5. **Integration test passes**
6. **Code review completed** (self-review checklist below)

### Self-Review Checklist
- [ ] Each fix addresses exactly one defect
- [ ] No new defects introduced
- [ ] Backward compatibility maintained (where applicable)
- [ ] Documentation updated (docstrings, comments)
- [ ] Error messages are user-friendly
- [ ] Edge cases handled

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Fix introduces new bug | Comprehensive unit tests, integration test |
| Database schema changes | Migration scripts, backward compatibility |
| Circuit breaker behavior changes | Extensive state transition tests |
| Event bus pattern changes | Verify all subscribers still work |

---

## Time Estimates by Defect

| Defect | Estimated Time | Actual Time |
|--------|----------------|-------------|
| C-1: TokenUsage.cost | 15 min | ___ |
| C-2: Foreign keys | 20 min | ___ |
| C-3: Session leak | 30 min | ___ |
| C-4: Async streaming | 30 min | ___ |
| C-5: JSON import | 5 min | ___ |
| C-6: EventBus injection | 20 min | ___ |
| C-7: Test fix | 10 min | ___ |
| Integration test | 20 min | ___ |
| **Total** | **~2.5 hours** | ___ |

---

## Next Phase Trigger

**Proceed to Phase 1 when:**
- [ ] This sub-plan is validated and approved
- [ ] All Phase 0 exit criteria met
- [ ] Time estimates filled in and acceptable

---

*Sub-Plan Version: 1.0*  
*Created: 2026-02-15*  
*Status: Ready for Review*
