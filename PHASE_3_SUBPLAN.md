# Phase 3: Application Layer - Orchestration & State Management
## Sub-Plan with Integrated Checklist

**Phase Objective:** Build the application layer that coordinates all infrastructure components into a cohesive chat system with session management.  
**Estimated Duration:** 6-8 hours  
**Success Criteria:** End-to-end chat flow works, streaming displays correctly, state persists across rerenders.  
**Dependencies:** Phase 0, 1, 2 complete (✅)  
**Methodology:** Test-Driven Development (TDD) - RED | GREEN | REFACTOR

---

## Phase 3 Architecture Overview

```
Phase 3 Deliverables:
├── Application Layer
│   ├── state.py              [NEW] Session state manager
│   ├── orchestrator.py       [NEW] ChatOrchestrator service
│   └── cost_tracker.py       [COMPLETE] Already done in Phase 2
│
├── Components Layer (NEW)
│   └── stream_bridge.py      [NEW] Async-to-sync bridge
│
└── Tests
    ├── unit/test_state.py              [NEW]
    ├── unit/test_orchestrator.py       [NEW]
    ├── unit/test_stream_bridge.py      [NEW]
    └── integration/test_chat_flow.py   [NEW]
```

---

## 3.1 Session State Manager

### Design Specification

Streamlit's session state is volatile (lost on page refresh) and type-unsafe (dictionary of Any). The StateManager provides:
- Type-safe wrapper around st.session_state
- Namespace isolation to prevent key collisions
- Conversation caching with serialization
- Stream buffer management for streaming state

### Files to Create

#### File: `src/neural_terminal/application/state.py`

**Interface Specification:**
```python
"""Session state management for Streamlit.

Provides type-safe abstraction over st.session_state with namespace
isolation to prevent key collisions.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import streamlit as st

from neural_terminal.domain.models import Conversation


@dataclass
class AppState:
    """Immutable application state container.
    
    Attributes:
        current_conversation_id: Active conversation ID (as string for JSON serialization)
        accumulated_cost: Accumulated cost as string (Decimal doesn't JSON serialize)
        selected_model: Currently selected model ID
        stream_buffer: Partial SSE data buffer
        is_streaming: Whether currently streaming
        error_message: Current error message if any
    """
    current_conversation_id: Optional[str] = None
    accumulated_cost: str = "0.00"
    selected_model: str = "openai/gpt-3.5-turbo"
    stream_buffer: str = ""
    is_streaming: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for session storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppState":
        """Create from dictionary."""
        return cls(**data)


class StateManager:
    """Type-safe wrapper around Streamlit's session state.
    
    Provides:
    - Namespace isolation (prevents key collisions)
    - Atomic updates
    - Conversation caching with proper serialization
    
    Phase 0 Defect H-3 Note:
        This manages synchronous state only. For async operations,
        use the StreamlitStreamBridge which handles the async-to-sync bridge.
    """
    
    _NAMESPACE = "neural_terminal_"
    
    def __init__(self):
        """Initialize state manager with namespace."""
        self._ensure_initialized()
    
    def _ensure_initialized(self) -> None:
        """Idempotent initialization of session state."""
        init_key = f"{self._NAMESPACE}initialized"
        if init_key not in st.session_state:
            st.session_state[init_key] = True
            st.session_state[f"{self._NAMESPACE}state"] = AppState().to_dict()
            st.session_state[f"{self._NAMESPACE}conversation_cache"] = {}
    
    @property
    def state(self) -> AppState:
        """Get current application state.
        
        Returns:
            AppState instance
        """
        raw = st.session_state.get(f"{self._NAMESPACE}state", {})
        return AppState.from_dict(raw)
    
    def update(self, **kwargs) -> None:
        """Atomic state update.
        
        Args:
            **kwargs: Key-value pairs to update
        """
        current = self.state
        new_state = AppState(**{**current.to_dict(), **kwargs})
        st.session_state[f"{self._NAMESPACE}state"] = new_state.to_dict()
    
    def set_conversation(self, conversation: Conversation) -> None:
        """Cache conversation in session state.
        
        Phase 0 Defect H-4 Fix:
            Uses conversation.to_dict() for proper serialization.
        
        Args:
            conversation: Conversation to cache
        """
        cache_key = f"{self._NAMESPACE}conversation_cache"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = {}
        
        # Serialize conversation properly
        conv_data = conversation.to_dict()
        st.session_state[cache_key][str(conversation.id)] = conv_data
        
        # Update current conversation ID
        self.update(current_conversation_id=str(conversation.id))
    
    def get_cached_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Retrieve conversation from cache.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation or None if not cached
        """
        cache_key = f"{self._NAMESPACE}conversation_cache"
        cache = st.session_state.get(cache_key, {})
        data = cache.get(conversation_id)
        
        if data:
            return Conversation(**data)
        return None
    
    def clear_stream_buffer(self) -> None:
        """Clear streaming buffer and reset streaming flag."""
        self.update(stream_buffer="", is_streaming=False)
    
    def append_stream_buffer(self, text: str) -> None:
        """Append text to streaming buffer.
        
        Args:
            text: Text to append
        """
        current = self.state.stream_buffer
        self.update(stream_buffer=current + text, is_streaming=True)
    
    def set_error(self, message: str) -> None:
        """Set error message.
        
        Args:
            message: Error message
        """
        self.update(error_message=message, is_streaming=False)
    
    def clear_error(self) -> None:
        """Clear error message."""
        self.update(error_message=None)
```

**TDD Checklist - 3.1:**
- [ ] **RED**: Write test `test_state_initialization()`
- [ ] **RED**: Write test `test_state_update()`
- [ ] **RED**: Write test `test_set_conversation_serializes_correctly()`
- [ ] **RED**: Write test `test_get_cached_conversation_deserializes()`
- [ ] **RED**: Write test `test_stream_buffer_operations()`
- [ ] **RED**: Write test `test_error_handling()`
- [ ] **GREEN**: Implement StateManager
- [ ] **VALIDATE**: All tests pass

---

## 3.2 Chat Orchestrator

### Design Specification

Central service managing conversation lifecycle:
- Dependency injection of all infrastructure
- Event emission during conversation flow
- Context window management with truncation
- Circuit breaker integration (manual checks for streaming)
- Error handling with partial message persistence
- Input validation (empty, too long)

### Files to Create

#### File: `src/neural_terminal/application/orchestrator.py`

**Interface Specification:**
```python
"""Chat orchestrator - central service for conversation management.

Coordinates between repositories, external APIs, and event system.
"""
import time
from decimal import Decimal
from typing import AsyncGenerator, List, Optional, Tuple
from uuid import UUID, uuid4

from neural_terminal.application.events import DomainEvent, EventBus, Events
from neural_terminal.config import settings
from neural_terminal.domain.exceptions import ValidationError
from neural_terminal.domain.models import Conversation, Message, MessageRole, TokenUsage
from neural_terminal.infrastructure.circuit_breaker import CircuitBreaker
from neural_terminal.infrastructure.openrouter import OpenRouterClient, OpenRouterModel
from neural_terminal.infrastructure.repositories import ConversationRepository
from neural_terminal.infrastructure.token_counter import TokenCounter


class ChatOrchestrator:
    """Domain service managing conversation lifecycle.
    
    Coordinates between repositories, external APIs, and event system.
    
    Phase 0 Defect C-4 Implementation:
        For streaming, we manually check circuit state, then stream directly,
        then manually record success/failure. This avoids trying to await
        an async generator.
    """
    
    def __init__(
        self,
        repository: ConversationRepository,
        openrouter: OpenRouterClient,
        event_bus: EventBus,
        token_counter: TokenCounter,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """Initialize orchestrator with dependencies.
        
        Args:
            repository: Conversation repository
            openrouter: OpenRouter API client
            event_bus: Event bus for domain events
            token_counter: Token counter for context management
            circuit_breaker: Optional circuit breaker for resilience
        """
        self._repo = repository
        self._openrouter = openrouter
        self._event_bus = event_bus
        self._tokenizer = token_counter
        self._circuit = circuit_breaker or CircuitBreaker()
        self._available_models: List[OpenRouterModel] = []
    
    async def load_models(self) -> List[OpenRouterModel]:
        """Fetch and cache available models.
        
        Returns:
            List of available models
        """
        self._available_models = await self._openrouter.get_available_models()
        return self._available_models
    
    def get_model_config(self, model_id: str) -> Optional[OpenRouterModel]:
        """Get pricing and capabilities for model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model configuration or None if not found
        """
        return next((m for m in self._available_models if m.id == model_id), None)
    
    async def create_conversation(
        self,
        title: Optional[str] = None,
        model_id: str = "openai/gpt-3.5-turbo",
        system_prompt: Optional[str] = None
    ) -> Conversation:
        """Initialize new conversation with optional system context.
        
        Args:
            title: Optional conversation title
            model_id: Model to use
            system_prompt: Optional system prompt
            
        Returns:
            Created conversation
        """
        conv = Conversation(title=title, model_id=model_id)
        
        if system_prompt:
            system_msg = Message(
                id=uuid4(),
                conversation_id=conv.id,
                role=MessageRole.SYSTEM,
                content=system_prompt
            )
            self._repo.add_message(system_msg)
        
        self._repo.save(conv)
        return conv
    
    async def send_message(
        self,
        conversation_id: UUID,
        content: str,
        temperature: float = 0.7
    ) -> AsyncGenerator[Tuple[str, Optional[dict]], None]:
        """Send message and stream response.
        
        Phase 0 Defect C-4 Implementation:
            We manually check circuit state before streaming, then iterate
            the async generator directly, then manually record success/failure.
        
        Args:
            conversation_id: Conversation ID
            content: Message content
            temperature: Sampling temperature
            
        Yields:
            Tuple of (delta_text, metadata_dict)
            metadata is None for deltas, dict with usage for final
            
        Raises:
            ValidationError: If conversation not found or input invalid
        """
        # Validate input
        if not content or not content.strip():
            raise ValidationError("Message content cannot be empty", code="EMPTY_INPUT")
        
        if len(content) > 32000:  # Max input length
            raise ValidationError(
                "Message exceeds maximum length (32000 chars)",
                code="INPUT_TOO_LONG"
            )
        
        # Load conversation
        conv = self._repo.get_by_id(conversation_id)
        if not conv:
            raise ValidationError(f"Conversation {conversation_id} not found")
        
        # Get model config for pricing
        model_config = self.get_model_config(conv.model_id)
        
        # Create user message
        user_msg = Message(
            id=uuid4(),
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content
        )
        
        # Check context length and truncate if necessary
        history = self._repo.get_messages(conversation_id)
        history.append(user_msg)
        
        max_context = model_config.context_length if model_config else 4096
        truncated = self._tokenizer.truncate_context(
            history,
            conv.model_id,
            max_context
        )
        
        if len(truncated) < len(history):
            self._event_bus.emit(DomainEvent(
                event_type=Events.CONTEXT_TRUNCATED,
                conversation_id=conversation_id,
                payload={"original_count": len(history), "new_count": len(truncated)}
            ))
        
        # Save user message
        self._repo.add_message(user_msg)
        
        # Prepare API messages
        api_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in truncated
        ]
        
        # Emit start event
        self._event_bus.emit(DomainEvent(
            event_type=Events.MESSAGE_STARTED,
            conversation_id=conversation_id,
            payload={"model": conv.model_id}
        ))
        
        # Streaming
        assistant_content = ""
        final_usage: Optional[TokenUsage] = None
        latency_ms = 0
        
        # Phase 0 Defect C-4: Manual circuit breaker check
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
                    final_usage = chunk.get("usage")
                    latency_ms = chunk.get("latency_ms", 0)
            
            # Record success
            self._circuit._on_success()
            
            # Calculate cost
            cost = Decimal("0")
            if final_usage and model_config:
                cost = self._calculate_cost(final_usage, model_config)
            
            # Save assistant message
            assistant_msg = Message(
                id=uuid4(),
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=assistant_content,
                token_usage=final_usage,
                cost=cost,
                latency_ms=latency_ms,
                model_id=conv.model_id
            )
            self._repo.add_message(assistant_msg)
            
            # Update conversation
            conv.update_cost(cost)
            self._repo.save(conv)
            
            # Emit completion
            self._event_bus.emit(DomainEvent(
                event_type=Events.MESSAGE_COMPLETED,
                conversation_id=conversation_id,
                payload={
                    "usage": final_usage,
                    "cost": str(cost),
                    "latency_ms": latency_ms
                }
            ))
            
            # Yield final metadata
            yield ("", {
                "usage": final_usage,
                "cost": cost,
                "latency": latency_ms,
                "message_id": assistant_msg.id
            })
        
        except Exception as e:
            # Record failure
            self._circuit._on_failure()
            
            # Save partial message on error
            if assistant_content:
                partial_msg = Message(
                    id=uuid4(),
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=assistant_content + "\n[Error: Stream interrupted]",
                    model_id=conv.model_id
                )
                self._repo.add_message(partial_msg)
            raise e
    
    def _calculate_cost(self, usage: TokenUsage, model: OpenRouterModel) -> Decimal:
        """Calculate cost from usage and pricing."""
        return usage.calculate_cost(
            model.prompt_price or Decimal("0"),
            model.completion_price or Decimal("0")
        )
```

**TDD Checklist - 3.2:**
- [ ] **RED**: Write test `test_create_conversation()`
- [ ] **RED**: Write test `test_create_conversation_with_system_prompt()`
- [ ] **RED**: Write test `test_send_message_validates_empty_input()`
- [ ] **RED**: Write test `test_send_message_validates_long_input()`
- [ ] **RED**: Write test `test_send_message_truncates_context()`
- [ ] **RED**: Write test `test_send_message_emits_events()`
- [ ] **RED**: Write test `test_send_message_saves_partial_on_error()`
- [ ] **GREEN**: Implement ChatOrchestrator
- [ ] **VALIDATE**: All tests pass

---

## 3.3 Streamlit Streaming Bridge

### Design Specification

Bridge async generators to Streamlit's synchronous world:
- Producer-consumer pattern with threading
- Queue-based communication
- Error propagation from async to sync
- Run async coroutines safely in Streamlit context

### Files to Create

#### File: `src/neural_terminal/components/stream_bridge.py`

**Interface Specification:**
```python
"""Async-to-sync bridge for Streamlit.

Bridges async generators to Streamlit's synchronous world using
threading and queues. Handles the nested event loop problem in Streamlit.

Phase 0 Defect H-3 Fix:
    Uses threading-based execution to avoid nested event loop issues
    that occur when using asyncio.run() inside Streamlit.
"""
import asyncio
import queue
import threading
from typing import Any, Callable, Coroutine, Optional, TypeVar

import streamlit as st

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Execute async coroutine safely within Streamlit.
    
    Phase 0 Defect H-3 Fix:
        Streamlit runs its own event loop. Calling asyncio.run() creates
        a new loop which conflicts. This function detects if we're in a
        running loop and uses threading to execute the coroutine if so.
    
    Args:
        coro: Async coroutine to execute
        
    Returns:
        Coroutine result
        
    Raises:
        Exception: Any exception from the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # We're inside an existing loop - run in thread
        result: list = [None]
        exception: list = [None]
        
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
        # No loop running - can use asyncio.run directly
        return asyncio.run(coro)


class StreamlitStreamBridge:
    """Bridges async generators to Streamlit's synchronous world.
    
    Uses producer-consumer pattern with threading to prevent blocking
    Streamlit's execution while waiting for async operations.
    
    Example:
        bridge = StreamlitStreamBridge(placeholder)
        metadata = bridge.stream(orchestrator.send_message(...))
    """
    
    def __init__(self, placeholder: Any):
        """Initialize bridge.
        
        Args:
            placeholder: Streamlit placeholder for updates
        """
        self.placeholder = placeholder
        self._buffer = ""
        self._queue: queue.Queue = queue.Queue()
        self._is_running = False
        self._error: Optional[str] = None
        self._final_metadata: Optional[dict] = None
    
    def stream(
        self,
        async_generator,
        on_delta: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[dict], None]] = None
    ) -> Optional[dict]:
        """Consume async generator and update Streamlit UI.
        
        Args:
            async_generator: Async generator yielding (delta, metadata)
            on_delta: Optional callback for each delta
            on_complete: Optional callback on completion
            
        Returns:
            Final metadata dict or None
            
        Raises:
            Exception: If error occurs during streaming
        """
        self._is_running = True
        
        # Start producer thread
        def run_async_in_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def consume():
                    async for delta, meta in async_generator:
                        if delta:
                            self._queue.put(("delta", delta))
                        if meta:
                            self._queue.put(("meta", meta))
                    self._queue.put(("done", None))
                
                loop.run_until_complete(consume())
            except Exception as e:
                self._queue.put(("error", str(e)))
        
        thread = threading.Thread(target=run_async_in_thread)
        thread.start()
        
        # Consume queue in main thread (Streamlit-safe)
        while self._is_running:
            try:
                msg_type, data = self._queue.get(timeout=0.1)
                
                if msg_type == "delta":
                    self._buffer += data
                    if on_delta:
                        on_delta(data)
                
                elif msg_type == "meta":
                    self._final_metadata = data
                    if on_complete:
                        on_complete(data)
                
                elif msg_type == "done":
                    self._is_running = False
                
                elif msg_type == "error":
                    self._error = data
                    self._is_running = False
                    raise Exception(data)
            
            except queue.Empty:
                continue
        
        thread.join()
        return self._final_metadata
    
    @property
    def content(self) -> str:
        """Get accumulated content."""
        return self._buffer
```

**TDD Checklist - 3.3:**
- [ ] **RED**: Write test `test_run_async_in_thread()`
- [ ] **RED**: Write test `test_stream_consumes_generator()`
- [ ] **RED**: Write test `test_stream_calls_on_delta()`
- [ ] **RED**: Write test `test_stream_calls_on_complete()`
- [ ] **RED**: Write test `test_stream_propagates_errors()`
- [ ] **RED**: Write test `test_stream_content_property()`
- [ ] **GREEN**: Implement StreamlitStreamBridge
- [ ] **VALIDATE**: All tests pass

---

## Phase 3 Integration Test

#### File: `tests/integration/test_chat_flow.py`

```python
"""Integration tests for complete chat flow.

Tests end-to-end conversation flow with mocked OpenRouter.
"""
import pytest
import respx
from httpx import Response

from neural_terminal.application.events import EventBus
from neural_terminal.application.orchestrator import ChatOrchestrator
from neural_terminal.infrastructure.openrouter import OpenRouterClient
from neural_terminal.infrastructure.repositories import SQLiteConversationRepository
from neural_terminal.infrastructure.token_counter import TokenCounter


class TestChatFlow:
    """End-to-end chat flow tests."""

    @pytest.fixture
    def setup(self):
        """Create orchestrator with all dependencies."""
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
        
        return {
            "orchestrator": orchestrator,
            "event_bus": event_bus,
            "repo": repo,
        }
    
    @respx.mock
    async def test_create_and_send_message(self, setup):
        """Test creating conversation and sending message."""
        orchestrator = setup["orchestrator"]
        repo = setup["repo"]
        
        # Mock OpenRouter models endpoint
        respx.get("https://openrouter.ai/api/v1/models").mock(
            return_value=Response(200, json={
                "data": [{
                    "id": "openai/gpt-3.5-turbo",
                    "name": "GPT-3.5 Turbo",
                    "pricing": {"prompt": "0.0015", "completion": "0.002"},
                    "context_length": 4096
                }]
            })
        )
        
        # Load models
        models = await orchestrator.load_models()
        assert len(models) == 1
        
        # Create conversation
        conv = await orchestrator.create_conversation(
            title="Test",
            model_id="openai/gpt-3.5-turbo",
            system_prompt="You are helpful."
        )
        
        # Verify conversation created
        assert conv.title == "Test"
        assert conv.model_id == "openai/gpt-3.5-turbo"
        
        # Mock chat completion endpoint
        respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=Response(200, text="""
data: {"choices": [{"delta": {"content": "Hello"}}]}

data: {"choices": [{"delta": {"content": " there"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}

data: [DONE]
""")
        )
        
        # Send message
        chunks = []
        async for delta, meta in orchestrator.send_message(
            conversation_id=conv.id,
            content="Hi!"
        ):
            if delta:
                chunks.append(delta)
            if meta:
                # Final metadata
                assert "cost" in meta
        
        # Verify response
        assert "".join(chunks) == "Hello there"
        
        # Verify message saved
        messages = repo.get_messages(conv.id)
        assert len(messages) == 3  # system + user + assistant
```

---

## Phase 3 Validation Criteria

### Pre-Validation Checklist
- [ ] StateManager with namespace isolation
- [ ] ChatOrchestrator with full flow
- [ ] StreamlitStreamBridge for async
- [ ] Integration tests for chat flow
- [ ] All error cases handled

### Success Criteria
- [ ] `make test` passes (100+ tests)
- [ ] End-to-end chat flow works
- [ ] Streaming tested with mocks
- [ ] State persistence verified
- [ ] Cost tracking accurate

---

## Time Estimates

| Task | Estimated | Actual |
|------|-----------|--------|
| 3.1 State Manager | 45 min | ___ |
| 3.2 Chat Orchestrator | 90 min | ___ |
| 3.3 Stream Bridge | 45 min | ___ |
| 3.4 Integration Tests | 45 min | ___ |
| **Total** | **~4 hours** | ___ |

---

*Sub-Plan Version: 1.0*  
*Created: 2026-02-15*  
*Status: Ready for Review*
