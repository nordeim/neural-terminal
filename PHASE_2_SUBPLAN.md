# Phase 2: Infrastructure - Database & External APIs
## Sub-Plan with Integrated Checklist

**Phase Objective:** Build resilient external API client, token counting, and the complete service layer with event-driven architecture.  
**Estimated Duration:** 6-8 hours  
**Success Criteria:** All integration tests pass, streaming works end-to-end, cost tracking accurate.  
**Dependencies:** Phase 0 (✅) and Phase 1 (✅) complete  
**Methodology:** Test-Driven Development (TDD) - RED | GREEN | REFACTOR

---

## Phase 2 Architecture Overview

```
Phase 2 Deliverables:
├── Infrastructure Layer
│   ├── openrouter.py         [C-4, C-5 COMPLETION] Streaming HTTP client
│   ├── token_counter.py      [NEW] Tiktoken integration
│   └── database.py           [COMPLETE] Already done
│
├── Application Layer (NEW)
│   ├── events.py             [NEW] Event bus with typed observers
│   ├── cost_tracker.py       [C-6 FIX] Budget tracking with injected EventBus
│   ├── orchestrator.py       [NEW] ChatOrchestrator service
│   └── state.py              [NEW] Session state manager
│
├── Components (NEW)
│   └── stream_bridge.py      [NEW] Async-to-sync bridge for Streamlit
│
└── Tests
    ├── integration/test_openrouter.py      [NEW]
    ├── integration/test_streaming.py       [NEW]
    ├── unit/test_token_counter.py          [NEW]
    ├── unit/test_events.py                 [NEW]
    ├── unit/test_cost_tracker.py           [NEW]
    └── unit/test_orchestrator.py           [NEW]
```

---

## 2.1 Token Counter (Tiktoken Integration)

### Design Specification

The token counter provides:
- Model-aware token encoding selection
- Token counting for messages and conversations
- Context window truncation strategy
- Caching of encoders for performance

### Files to Create

#### File: `src/neural_terminal/infrastructure/token_counter.py`

**Interface Specification:**
```python
"""Token counting infrastructure using tiktoken.

Provides model-aware token counting with encoding caching.
"""
import tiktoken
from typing import List, Optional

from neural_terminal.domain.models import Message, MessageRole


class TokenCounter:
    """Model-aware token counting with encoding caching.
    
    Uses tiktoken for accurate OpenAI-compatible token counting.
    Falls back to cl100k_base for unknown models (Claude approximation).
    """
    
    # Mapping of model name patterns to tiktoken encoding names
    ENCODING_MAP = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "claude": "cl100k_base",  # Approximation - Claude uses different tokenizer
        "default": "cl100k_base"
    }
    
    def __init__(self):
        """Initialize with empty encoder cache."""
        self._encoders: dict = {}
    
    def _get_encoder(self, model_id: str) -> tiktoken.Encoding:
        """Get or create encoder for model.
        
        Args:
            model_id: Model identifier (e.g., 'openai/gpt-3.5-turbo')
            
        Returns:
            Tiktoken encoding instance
        """
        # Extract base model name
        base = model_id.split("/")[-1].lower()
        
        # Find matching encoding
        encoding_name = "default"
        for key, enc in self.ENCODING_MAP.items():
            if key in base:
                encoding_name = enc
                break
        
        # Cache encoder instances
        if encoding_name not in self._encoders:
            self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
        
        return self._encoders[encoding_name]
    
    def count_tokens(self, text: str, model_id: str) -> int:
        """Count tokens in plain text.
        
        Args:
            text: Text to count
            model_id: Model identifier
            
        Returns:
            Number of tokens
        """
        encoder = self._get_encoder(model_id)
        return len(encoder.encode(text))
    
    def count_message(self, message: Message, model_id: str) -> int:
        """Count tokens in a single message.
        
        Uses tiktoken's message format: <|start|>{role}\n{content}<|end|>
        
        Args:
            message: Message to count
            model_id: Model identifier
            
        Returns:
            Number of tokens
        """
        encoder = self._get_encoder(model_id)
        
        # Tiktoken format: <|start|>{role}\n{content}<|end|>
        # Every message follows <|start|>{role/name}\n{content}<|end|>
        # See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        tokens = 4  # Base overhead for message formatting
        tokens += len(encoder.encode(message.role.value))
        tokens += len(encoder.encode(message.content))
        
        return tokens
    
    def count_messages(self, messages: List[Message], model_id: str) -> int:
        """Count total tokens for conversation history.
        
        Args:
            messages: List of messages
            model_id: Model identifier
            
        Returns:
            Total number of tokens
        """
        total = 0
        for msg in messages:
            total += self.count_message(msg, model_id)
        total += 2  # Reply primer
        return total
    
    def truncate_context(
        self,
        messages: List[Message],
        model_id: str,
        max_tokens: int,
        reserve_tokens: int = 500
    ) -> List[Message]:
        """Truncate messages to fit context window.
        
        Strategy: Keep system message (if first), keep recent messages,
        drop middle messages with summarization marker.
        
        Args:
            messages: Full message list
            model_id: Model identifier
            max_tokens: Maximum tokens allowed
            reserve_tokens: Tokens to reserve for response
            
        Returns:
            Truncated message list
        """
        if not messages:
            return messages
        
        target_tokens = max_tokens - reserve_tokens
        
        # Always keep system message if present
        system_messages = []
        conversation_messages = []
        
        if messages[0].role == MessageRole.SYSTEM:
            system_messages = [messages[0]]
            conversation_messages = messages[1:]
        else:
            conversation_messages = messages
        
        # Count system messages
        current_tokens = self.count_messages(system_messages, model_id)
        truncated = list(system_messages)
        
        # Add messages from the end until limit reached
        for msg in reversed(conversation_messages):
            msg_tokens = self.count_message(msg, model_id)
            if current_tokens + msg_tokens > target_tokens:
                break
            truncated.insert(len(system_messages), msg)
            current_tokens += msg_tokens
        
        # Add truncation marker if we dropped messages
        if len(truncated) < len(messages):
            marker = Message(
                role=MessageRole.SYSTEM,
                content="[Earlier conversation context truncated due to length]",
                conversation_id=messages[0].conversation_id if messages else None
            )
            truncated.insert(len(system_messages), marker)
        
        return truncated
```

**TDD Checklist - 2.1:**
- [ ] **RED**: Write test `test_count_tokens_with_known_values()`
  - Input: "Hello world" with gpt-3.5-turbo
  - Expected: 2 tokens
- [ ] **RED**: Write test `test_count_message_with_role()`
  - Input: Message(role=USER, content="Hello")
  - Expected: role tokens + content tokens + overhead
- [ ] **RED**: Write test `test_count_messages_total()`
  - Input: 3 messages
  - Expected: sum of individual counts + reply primer
- [ ] **RED**: Write test `test_encoder_caching()`
  - Call twice with same model
  - Verify same encoder instance returned
- [ ] **RED**: Write test `test_truncate_context_keeps_system_message()`
  - Input: System + 10 conversation messages
  - Limit: Small enough to truncate
  - Expected: System message preserved, marker added
- [ ] **RED**: Write test `test_truncate_context_adds_marker()`
  - Verify marker message added when truncation occurs
- [ ] **GREEN**: Implement TokenCounter class
- [ ] **VALIDATE**: All tests pass

---

## 2.2 Event System

### Design Specification

Event-driven architecture for decoupled communication:
- Typed event bus with observer pattern
- Supports both specific and global subscribers
- Error isolation (one subscriber failure doesn't stop propagation)
- Standard event types for the domain

### Files to Create

#### File: `src/neural_terminal/application/events.py`

**Interface Specification:**
```python
"""Event system for decoupled communication.

Implements Observer pattern with typed event bus.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass(frozen=True)
class DomainEvent:
    """Immutable domain event.
    
    Attributes:
        event_type: Event type identifier
        conversation_id: Optional conversation context
        payload: Event data dictionary
    """
    event_type: str
    conversation_id: Optional[UUID] = None
    payload: Optional[Dict[str, Any]] = None


class EventObserver(ABC):
    """Abstract base class for event observers."""
    
    @abstractmethod
    def on_event(self, event: DomainEvent) -> None:
        """Handle a domain event.
        
        Args:
            event: The event to handle
        """
        raise NotImplementedError


class EventBus:
    """Thread-safe event bus for decoupled communication.
    
    Supports:
    - Typed subscribers (specific event types)
    - Global subscribers (all events)
    - Error isolation (subscriber failures don't stop propagation)
    """
    
    def __init__(self):
        """Initialize empty subscriber registry."""
        self._subscribers: Dict[str, List[EventObserver]] = {}
        self._global_subscribers: List[EventObserver] = []
    
    def subscribe(self, event_type: str, observer: EventObserver) -> None:
        """Subscribe to a specific event type.
        
        Args:
            event_type: Event type to subscribe to
            observer: Observer instance
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(observer)
    
    def subscribe_all(self, observer: EventObserver) -> None:
        """Subscribe to all events.
        
        Args:
            observer: Observer instance
        """
        self._global_subscribers.append(observer)
    
    def emit(self, event: DomainEvent) -> None:
        """Emit an event to all subscribers.
        
        Errors in subscribers are caught and logged but don't stop
        event propagation to other subscribers.
        
        Args:
            event: Event to emit
        """
        # Notify specific subscribers
        for observer in self._subscribers.get(event.event_type, []):
            try:
                observer.on_event(event)
            except Exception as e:
                # Log but don't stop propagation
                print(f"Event handler error: {e}")
        
        # Notify global subscribers
        for observer in self._global_subscribers:
            try:
                observer.on_event(event)
            except Exception as e:
                print(f"Global handler error: {e}")


class Events:
    """Standard event type constants."""
    
    # Message lifecycle
    MESSAGE_STARTED = "message.started"
    TOKEN_GENERATED = "token.generated"
    MESSAGE_COMPLETED = "message.completed"
    
    # Budget
    BUDGET_THRESHOLD = "budget.threshold"
    BUDGET_EXCEEDED = "budget.exceeded"
    
    # Context
    CONTEXT_TRUNCATED = "context.truncated"
```

**TDD Checklist - 2.2:**
- [ ] **RED**: Write test `test_subscribe_and_emit()`
  - Subscribe observer to event type
  - Emit event
  - Verify observer received event
- [ ] **RED**: Write test `test_subscribe_all_receives_all_events()`
  - Subscribe global observer
  - Emit multiple event types
  - Verify observer received all
- [ ] **RED**: Write test `test_error_isolation()`
  - Subscribe observer that raises exception
  - Subscribe second observer
  - Emit event
  - Verify second observer still received event
- [ ] **RED**: Write test `test_event_immutability()`
  - Attempt to modify frozen DomainEvent
  - Verify FrozenInstanceError raised
- [ ] **GREEN**: Implement EventBus and DomainEvent
- [ ] **VALIDATE**: All tests pass

---

## 2.3 Cost Tracker (C-6 Fix Completion)

### Design Specification

Real-time cost tracking with budget enforcement:
- Implements EventObserver for event-driven updates
- Injected EventBus (not creating orphan instances)
- Estimates cost during streaming
- Reconciles with actual usage at completion
- Emits budget threshold and exceeded events

### Files to Create

#### File: `src/neural_terminal/application/cost_tracker.py`

**Interface Specification:**
```python
"""Cost tracking with budget enforcement.

Phase 0 Defect C-6 Fix:
    EventBus is injected in constructor, not created as orphan instance.
    This ensures budget events are emitted to the shared bus.
"""
from decimal import Decimal
from typing import Optional

from neural_terminal.application.events import DomainEvent, EventBus, EventObserver, Events
from neural_terminal.domain.models import TokenUsage
from neural_terminal.infrastructure.openrouter import OpenRouterModel


class CostTracker(EventObserver):
    """Real-time cost accumulator with budget enforcement.
    
    Implements Observer pattern for decoupled economic tracking.
    
    Phase 0 Defect C-6 Fix:
        EventBus is injected via constructor. Never create new EventBus
        instances within methods - that sends events to nowhere.
    
    Args:
        event_bus: Shared EventBus instance for emitting budget events
        budget_limit: Optional budget limit in USD
    """
    
    def __init__(self, event_bus: EventBus, budget_limit: Optional[Decimal] = None):
        """Initialize cost tracker.
        
        Args:
            event_bus: Shared event bus (injected, not created)
            budget_limit: Optional budget limit in USD
        """
        self._bus = event_bus  # Injected singleton - use this!
        self._accumulated = Decimal("0.00")
        self._budget_limit = budget_limit
        self._current_model_price: Optional[OpenRouterModel] = None
        self._estimated_tokens = 0
        self._is_tracking = False
    
    def set_model(self, model: OpenRouterModel) -> None:
        """Set current pricing model for estimation.
        
        Args:
            model: OpenRouter model with pricing
        """
        self._current_model_price = model
    
    def on_event(self, event: DomainEvent) -> None:
        """Handle domain events for cost tracking.
        
        Args:
            event: Domain event to process
        """
        if event.event_type == Events.MESSAGE_STARTED:
            self._is_tracking = True
            self._estimated_tokens = 0
        
        elif event.event_type == Events.TOKEN_GENERATED:
            # Estimate cost during streaming (rough approximation)
            self._estimated_tokens += 1
            # Check budget every 100 tokens
            if self._estimated_tokens % 100 == 0:
                self._check_budget(self._estimate_current_cost())
        
        elif event.event_type == Events.MESSAGE_COMPLETED:
            # Reconcile with actual usage from API
            usage = event.payload.get("usage") if event.payload else None
            if usage and isinstance(usage, TokenUsage):
                actual_cost = self._calculate_actual_cost(usage)
                self._accumulated += actual_cost
                self._is_tracking = False
                
                # Final budget check
                if self._budget_limit and self._accumulated > self._budget_limit:
                    self._emit_budget_exceeded()
    
    def _estimate_current_cost(self) -> Decimal:
        """Estimate cost during streaming.
        
        Returns:
            Estimated cost based on current token count
        """
        if not self._current_model_price or not self._current_model_price.completion_price:
            return Decimal("0")
        
        return (Decimal(self._estimated_tokens) / 1000) * self._current_model_price.completion_price
    
    def _calculate_actual_cost(self, usage: TokenUsage) -> Decimal:
        """Calculate precise cost from usage.
        
        Args:
            usage: Token usage from API
            
        Returns:
            Actual cost in USD
        """
        if not self._current_model_price:
            return Decimal("0")
        
        prompt_price = self._current_model_price.prompt_price or Decimal("0")
        completion_price = self._current_model_price.completion_price or Decimal("0")
        
        return usage.calculate_cost(prompt_price, completion_price)
    
    def _check_budget(self, estimated_cost: Decimal) -> None:
        """Check if approaching budget limit and emit events.
        
        Args:
            estimated_cost: Current estimated cost
        """
        if not self._budget_limit:
            return
        
        projected = self._accumulated + estimated_cost
        
        if projected > self._budget_limit:
            self._emit_budget_exceeded()
        elif projected > self._budget_limit * Decimal("0.8"):
            # Emit warning at 80%
            self._bus.emit(DomainEvent(
                event_type=Events.BUDGET_THRESHOLD,
                payload={
                    "accumulated": str(self._accumulated),
                    "limit": str(self._budget_limit),
                }
            ))
    
    def _emit_budget_exceeded(self) -> None:
        """Emit budget exceeded event using injected bus."""
        self._bus.emit(DomainEvent(
            event_type=Events.BUDGET_EXCEEDED,
            payload={"accumulated": str(self._accumulated)}
        ))
    
    @property
    def accumulated_cost(self) -> Decimal:
        """Get accumulated cost."""
        return self._accumulated
    
    def reset(self) -> None:
        """Reset accumulated cost."""
        self._accumulated = Decimal("0.00")
```

**TDD Checklist - 2.3:**
- [ ] **RED**: Write test `test_cost_tracker_requires_event_bus()`
  - Verify EventBus is required in constructor
- [ ] **RED**: Write test `test_message_started_resets_tracking()`
  - Emit MESSAGE_STARTED
  - Verify internal state reset
- [ ] **RED**: Write test `test_token_generated_estimates_cost()`
  - Set model with pricing
  - Emit 100 TOKEN_GENERATED events
  - Verify budget check triggered
- [ ] **RED**: Write test `test_message_completed_calculates_actual_cost()`
  - Set model with known pricing
  - Emit MESSAGE_COMPLETED with TokenUsage
  - Verify accumulated_cost matches expected
- [ ] **RED**: Write test `test_budget_threshold_emits_event_at_80_percent()`
  - Set budget limit
  - Accumulate cost to 80% threshold
  - Verify BUDGET_THRESHOLD event emitted to shared bus
- [ ] **RED**: Write test `test_budget_exceeded_emits_event()`
  - Exceed budget limit
  - Verify BUDGET_EXCEEDED event emitted
- [ ] **RED**: Write test `test_no_orphan_event_bus()`
  - Verify tracker uses injected bus, never creates new one
- [ ] **GREEN**: Implement CostTracker with injected EventBus
- [ ] **VALIDATE**: All tests pass

---

## 2.4 OpenRouter Client (C-4, C-5 Completion)

### Design Specification

Resilient HTTP client for OpenRouter API:
- Async streaming with SSE parsing
- Phase 0 C-4 Fix: Don't wrap async generators in circuit breaker
- Phase 0 C-5 Fix: Include `import json`
- Error translation to domain exceptions
- Model listing with caching consideration

### Files to Create

#### File: `src/neural_terminal/infrastructure/openrouter.py`

**Interface Specification:**
```python
"""OpenRouter API client with streaming support.

Phase 0 Defect C-4 Fix:
    chat_completion_stream yields directly. Connection errors surface
    before first yield. Success/failure recorded manually by caller.

Phase 0 Defect C-5 Fix:
    json module is imported at top of file.
"""
import json  # C-5 FIX: Must be imported for SSE parsing
import time
from decimal import Decimal
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from neural_terminal.config import settings
from neural_terminal.domain.exceptions import (
    ModelUnavailableError,
    OpenRouterAPIError,
    RateLimitError,
    TokenLimitError,
)
from neural_terminal.domain.models import TokenUsage


class OpenRouterModel(BaseModel):
    """OpenRouter model information.
    
    Attributes:
        id: Model identifier (e.g., 'openai/gpt-4')
        name: Human-readable name
        description: Model description
        pricing: Pricing dictionary with 'prompt' and 'completion' keys
        context_length: Maximum context length in tokens
    """
    id: str
    name: str
    description: Optional[str] = None
    pricing: Dict[str, Optional[str]] = Field(default_factory=dict)
    context_length: Optional[int] = None
    
    @property
    def prompt_price(self) -> Optional[Decimal]:
        """Get prompt price per 1K tokens."""
        if "prompt" in self.pricing and self.pricing["prompt"]:
            return Decimal(self.pricing["prompt"])
        return None
    
    @property
    def completion_price(self) -> Optional[Decimal]:
        """Get completion price per 1K tokens."""
        if "completion" in self.pricing and self.pricing["completion"]:
            return Decimal(self.pricing["completion"])
        return None


class OpenRouterClient:
    """Resilient OpenRouter API client with circuit breaker support.
    
    Phase 0 Defect C-4 Fix:
        chat_completion_stream yields chunks directly. Callers must:
        1. Check circuit state manually before calling
        2. Record success/failure manually after streaming
    
    Usage:
        client = OpenRouterClient()
        
        # Check circuit state first
        circuit._check_state()
        
        try:
            async for chunk in client.chat_completion_stream(...):
                # Process chunk
                pass
            circuit._on_success()
        except Exception:
            circuit._on_failure()
            raise
    """
    
    def __init__(self):
        """Initialize client with configuration."""
        self.base_url = settings.openrouter_base_url
        self.api_key = settings.openrouter_api_key.get_secret_value()
        self.timeout = settings.openrouter_timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.
        
        Returns:
            Configured httpx AsyncClient
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://neural-terminal.local",
                    "X-Title": "Neural Terminal",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    
    async def get_available_models(self) -> List[OpenRouterModel]:
        """Fetch available models from OpenRouter.
        
        Returns:
            List of available models
            
        Raises:
            OpenRouterAPIError: If API request fails
        """
        client = await self._get_client()
        
        response = await client.get("/models")
        
        if response.status_code != 200:
            raise OpenRouterAPIError(
                message=f"Failed to fetch models: {response.text}",
                status_code=response.status_code,
                response_body=response.text,
            )
        
        data = response.json()
        return [OpenRouterModel(**m) for m in data.get("data", [])]
    
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming chat completion with SSE parsing.
        
        Phase 0 Defect C-4 Fix:
            This method yields directly. It should NOT be wrapped in the
            circuit breaker's call_async method (which would try to await
            an async generator, causing TypeError).
        
        Connection errors surface before first yield.
        Success/failure should be recorded manually by caller.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model ID (e.g., 'openai/gpt-3.5-turbo')
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Dict with keys:
            - 'type': 'delta' | 'final'
            - 'content': str (for delta type)
            - 'usage': TokenUsage (for final type)
            - 'latency_ms': int (for final type)
            
        Raises:
            RateLimitError: On 429 response
            TokenLimitError: On 400 context too long
            ModelUnavailableError: On 503 response
            OpenRouterAPIError: On other errors
        """
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
        
        async with client.stream(
            "POST",
            "/chat/completions",
            json=payload,
        ) as response:
            # Check for errors before streaming
            if response.status_code == 429:
                body = await response.aread()
                raise RateLimitError(
                    retry_after=int(response.headers.get("retry-after", 60))
                )
            elif response.status_code == 503:
                body = await response.aread()
                raise ModelUnavailableError(model_id=model)
            elif response.status_code == 400:
                body = await response.aread()
                # Check if it's a context length error
                if "context" in body.decode().lower() or "token" in body.decode().lower():
                    raise TokenLimitError()
                raise OpenRouterAPIError(
                    message="Bad request",
                    status_code=400,
                    response_body=body.decode(),
                )
            elif response.status_code >= 400:
                body = await response.aread()
                raise OpenRouterAPIError(
                    message=f"OpenRouter API error: {body.decode()}",
                    status_code=response.status_code,
                    response_body=body.decode(),
                )
            
            # Stream SSE data
            full_content = ""
            usage: Optional[TokenUsage] = None
            
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                
                data = line[6:]
                if data == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data)  # C-5 FIX: json imported at top
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content:
                        full_content += content
                        yield {
                            "type": "delta",
                            "content": content,
                            "accumulated": full_content,
                        }
                    
                    # Check for usage in final chunk
                    if "usage" in chunk:
                        u = chunk["usage"]
                        usage = TokenUsage(
                            prompt_tokens=u.get("prompt_tokens", 0),
                            completion_tokens=u.get("completion_tokens", 0),
                            total_tokens=u.get("total_tokens", 0),
                        )
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
            
            # Final yield with metadata
            latency = int((time.time() - start_time) * 1000)
            yield {
                "type": "final",
                "content": full_content,
                "usage": usage,
                "latency_ms": latency,
                "model": model,
            }
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
```

**TDD Checklist - 2.4:**
- [ ] **RED**: Write test `test_json_import_available()`
  - Verify json module imported (C-5 regression test)
- [ ] **RED**: Write test `test_get_available_models_with_mock()`
  - Mock httpx response
  - Verify models parsed correctly
- [ ] **RED**: Write test `test_chat_completion_stream_yields_deltas()`
  - Mock SSE stream
  - Verify delta chunks yielded
- [ ] **RED**: Write test `test_chat_completion_stream_yields_final_with_usage()`
  - Mock stream with usage in final chunk
  - Verify final chunk has usage data
- [ ] **RED**: Write test `test_rate_limit_error_raised_on_429()`
  - Mock 429 response
  - Verify RateLimitError raised with retry_after
- [ ] **RED**: Write test `test_model_unavailable_error_on_503()`
  - Mock 503 response
  - Verify ModelUnavailableError raised
- [ ] **RED**: Write test `test_token_limit_error_on_400_context()`
  - Mock 400 with context length message
  - Verify TokenLimitError raised
- [ ] **RED**: Write test `test_streaming_not_wrapped_in_circuit()`
  - Document that streaming yields directly (C-4)
- [ ] **GREEN**: Implement OpenRouterClient
- [ ] **VALIDATE**: All tests pass

---

## 2.5 Chat Orchestrator

### Design Specification

Central service managing conversation lifecycle:
- Dependency injection of all infrastructure
- Event emission during conversation flow
- Context window management with truncation
- Circuit breaker integration (manual checks for streaming)
- Error handling with partial message persistence

### Files to Create

#### File: `src/neural_terminal/application/orchestrator.py`

**Interface Specification:**
```python
"""Chat orchestrator - central service for conversation management.

Coordinates between repositories, external APIs, and event system.
"""
import asyncio
import time
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
            raise ValidationError("Message content cannot be empty")
        
        if len(content) > 32000:  # Max input length
            raise ValidationError(
                f"Message exceeds maximum length (32000 chars)",
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
        """Calculate cost from usage and pricing.
        
        Args:
            usage: Token usage
            model: Model with pricing
            
        Returns:
            Cost in USD
        """
        return usage.calculate_cost(
            model.prompt_price or Decimal("0"),
            model.completion_price or Decimal("0")
        )
```

**TDD Checklist - 2.5:**
- [ ] **RED**: Write test `test_create_conversation()`
- [ ] **RED**: Write test `test_create_conversation_with_system_prompt()`
- [ ] **RED**: Write test `test_send_message_validates_empty_input()`
- [ ] **RED**: Write test `test_send_message_validates_long_input()`
- [ ] **RED**: Write test `test_send_message_validates_conversation_exists()`
- [ ] **RED**: Write test `test_send_message_truncates_context()`
- [ ] **RED**: Write test `test_send_message_emits_events()`
- [ ] **RED**: Write test `test_send_message_handles_streaming_errors()`
- [ ] **RED**: Write test `test_send_message_saves_partial_on_error()`
- [ ] **GREEN**: Implement ChatOrchestrator
- [ ] **VALIDATE**: All tests pass

---

## Phase 2 Validation Criteria

### Pre-Validation Checklist
- [ ] All infrastructure files created
- [ ] All application layer files created
- [ ] Event system implemented
- [ ] Cost tracker with injected EventBus
- [ ] OpenRouter client with streaming
- [ ] Chat orchestrator with circuit breaker integration
- [ ] Comprehensive tests for all components

### Integration Test
```python
# tests/integration/test_phase2_infrastructure.py
async def test_end_to_end_streaming_flow():
    """Test complete flow with mocked OpenRouter."""
    # Arrange
    event_bus = EventBus()
    repo = SQLiteConversationRepository()
    client = OpenRouterClient()
    counter = TokenCounter()
    orchestrator = ChatOrchestrator(repo, client, event_bus, counter)
    
    # Act - Create conversation and send message
    conv = await orchestrator.create_conversation()
    
    # Mock the streaming response
    # ... mock setup ...
    
    # Assert
    # - Message persisted
    # - Events emitted
    # - Cost calculated correctly
```

### Success Criteria
- [ ] `make test` passes all tests
- [ ] Integration tests pass
- [ ] Cost tracking accurate to $0.0001
- [ ] Streaming works with mocked responses
- [ ] Event flow verified

---

## Time Estimates

| Task | Estimated | Actual |
|------|-----------|--------|
| 2.1 Token Counter | 45 min | ___ |
| 2.2 Event System | 30 min | ___ |
| 2.3 Cost Tracker | 45 min | ___ |
| 2.4 OpenRouter Client | 60 min | ___ |
| 2.5 Chat Orchestrator | 60 min | ___ |
| 2.6 Integration Tests | 30 min | ___ |
| **Total** | **~5.5 hours** | ___ |

---

*Sub-Plan Version: 1.0*  
*Created: 2026-02-15*  
*Status: Ready for Review*
