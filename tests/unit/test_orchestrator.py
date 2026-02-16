"""Unit tests for chat orchestrator.

Tests for Phase 3: ChatOrchestrator conversation management.
"""
from decimal import Decimal
from uuid import uuid4

import pytest

from neural_terminal.application.events import EventBus, Events
from neural_terminal.application.orchestrator import ChatOrchestrator
from neural_terminal.domain.exceptions import ValidationError
from neural_terminal.domain.models import ConversationStatus, MessageRole, TokenUsage
from neural_terminal.infrastructure.circuit_breaker import CircuitBreaker
from neural_terminal.infrastructure.openrouter import OpenRouterClient
from neural_terminal.infrastructure.repositories import SQLiteConversationRepository
from neural_terminal.infrastructure.token_counter import TokenCounter


class MockAsyncGenerator:
    """Mock async generator for testing streaming."""
    
    def __init__(self, chunks):
        self.chunks = chunks
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


class TestChatOrchestrator:
    """Tests for ChatOrchestrator."""

    @pytest.fixture
    def setup(self):
        """Create orchestrator with all dependencies."""
        event_bus = EventBus()
        repo = SQLiteConversationRepository()
        
        # We'll mock the OpenRouterClient methods
        class MockOpenRouterClient:
            async def get_available_models(self):
                from neural_terminal.infrastructure.openrouter import OpenRouterModel
                return [
                    OpenRouterModel(
                        id="meta/llama-3.1-8b-instruct",
                        name="Llama 3.1 8B",
                        pricing={"prompt": "0.0005", "completion": "0.001"},
                        context_length=8192
                    )
                ]
            
            async def chat_completion_stream(self, **kwargs):
                # Mock streaming response
                yield {"type": "delta", "content": "Hello", "accumulated": "Hello"}
                yield {"type": "delta", "content": " there", "accumulated": "Hello there"}
                yield {
                    "type": "final",
                    "content": "Hello there",
                    "usage": TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                    "latency_ms": 500,
                    "model": "meta/llama-3.1-8b-instruct"
                }
        
        client = MockOpenRouterClient()
        counter = TokenCounter()
        circuit = CircuitBreaker()
        
        orchestrator = ChatOrchestrator(
            repository=repo,
            openrouter=client,
            event_bus=event_bus,
            token_counter=counter,
            circuit_breaker=circuit
        )
        
        return {
            "orchestrator": orchestrator,
            "event_bus": event_bus,
            "repo": repo,
            "circuit": circuit,
        }
    
    def test_create_conversation(self, setup):
        """Test creating a conversation."""
        orchestrator = setup["orchestrator"]
        
        conv = orchestrator.create_conversation(
            title="Test Conversation",
            model_id="openai/gpt-3.5-turbo"
        )
        
        assert conv.title == "Test Conversation"
        assert conv.model_id == "openai/gpt-3.5-turbo"
        assert conv.status == ConversationStatus.ACTIVE
    
    def test_create_conversation_with_system_prompt(self, setup):
        """Test creating a conversation with system prompt."""
        orchestrator = setup["orchestrator"]
        repo = setup["repo"]
        
        conv = orchestrator.create_conversation(
            title="Test",
            system_prompt="You are helpful."
        )
        
        # Verify system message was saved
        messages = repo.get_messages(conv.id)
        assert len(messages) == 1
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[0].content == "You are helpful."
    
    @pytest.mark.asyncio
    async def test_load_models(self, setup):
        """Test loading available models."""
        orchestrator = setup["orchestrator"]
        
        models = await orchestrator.load_models()
        
        assert len(models) == 1
        assert models[0].id == "meta/llama-3.1-8b-instruct"
    
    @pytest.mark.asyncio
    async def test_get_model_config(self, setup):
        """Test getting model configuration."""
        orchestrator = setup["orchestrator"]
        await orchestrator.load_models()
        
        config = orchestrator.get_model_config("meta/llama-3.1-8b-instruct")
        
        assert config is not None
        assert config.prompt_price == Decimal("0.0005")
        assert config.completion_price == Decimal("0.001")
    
    @pytest.mark.asyncio
    async def test_send_message_validates_empty_input(self, setup):
        """Test that empty input raises ValidationError."""
        orchestrator = setup["orchestrator"]
        
        conv = orchestrator.create_conversation()
        
        with pytest.raises(ValidationError, match="cannot be empty"):
            async for _ in orchestrator.send_message(conv.id, ""):
                pass
        
        with pytest.raises(ValidationError, match="cannot be empty"):
            async for _ in orchestrator.send_message(conv.id, "   "):
                pass
    
    @pytest.mark.asyncio
    async def test_send_message_validates_long_input(self, setup):
        """Test that long input raises ValidationError."""
        orchestrator = setup["orchestrator"]
        
        conv = orchestrator.create_conversation()
        
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            async for _ in orchestrator.send_message(conv.id, "x" * 32001):
                pass
    
    @pytest.mark.asyncio
    async def test_send_message_validates_conversation_exists(self, setup):
        """Test that non-existent conversation raises ValidationError."""
        orchestrator = setup["orchestrator"]
        
        with pytest.raises(ValidationError, match="not found"):
            async for _ in orchestrator.send_message(uuid4(), "Hello"):
                pass
    
    @pytest.mark.asyncio
    async def test_send_message_emits_events(self, setup):
        """Test that send_message emits domain events."""
        orchestrator = setup["orchestrator"]
        event_bus = setup["event_bus"]
        
        # Track events
        events_received = []
        class EventTracker:
            def on_event(self, event):
                events_received.append(event)
        
        tracker = EventTracker()
        event_bus.subscribe_all(tracker)
        
        # Load models first
        await orchestrator.load_models()
        
        conv = orchestrator.create_conversation()
        
        # Send message
        async for delta, meta in orchestrator.send_message(conv.id, "Hi"):
            pass
        
        # Check events were emitted
        event_types = [e.event_type for e in events_received]
        assert Events.MESSAGE_STARTED in event_types
        assert Events.TOKEN_GENERATED in event_types
        assert Events.MESSAGE_COMPLETED in event_types
    
    @pytest.mark.asyncio
    async def test_send_message_saves_messages(self, setup):
        """Test that send_message saves user and assistant messages."""
        orchestrator = setup["orchestrator"]
        repo = setup["repo"]
        
        await orchestrator.load_models()
        conv = orchestrator.create_conversation()
        
        # Send message
        async for delta, meta in orchestrator.send_message(conv.id, "Hi"):
            pass
        
        # Check messages saved
        messages = repo.get_messages(conv.id)
        assert len(messages) == 2  # user + assistant
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "Hi"
        assert messages[1].role == MessageRole.ASSISTANT
        assert messages[1].content == "Hello there"
    
    @pytest.mark.asyncio
    async def test_send_message_updates_conversation_cost(self, setup):
        """Test that send_message updates conversation cost."""
        orchestrator = setup["orchestrator"]
        repo = setup["repo"]
        
        await orchestrator.load_models()
        conv = orchestrator.create_conversation()
        
        initial_cost = conv.total_cost
        
        # Send message
        async for delta, meta in orchestrator.send_message(conv.id, "Hi"):
            pass
        
        # Reload conversation and check cost updated
        updated_conv = repo.get_by_id(conv.id)
        assert updated_conv.total_cost > initial_cost
    
    @pytest.mark.asyncio
    async def test_send_message_returns_metadata(self, setup):
        """Test that send_message returns metadata in final chunk."""
        orchestrator = setup["orchestrator"]
        
        await orchestrator.load_models()
        conv = orchestrator.create_conversation()
        
        final_meta = None
        async for delta, meta in orchestrator.send_message(conv.id, "Hi"):
            if meta:
                final_meta = meta
        
        assert final_meta is not None
        assert "cost" in final_meta
        assert "latency" in final_meta
        assert "message_id" in final_meta
