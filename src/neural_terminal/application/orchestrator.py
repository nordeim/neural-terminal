"""Chat orchestrator - central service for conversation management.

Coordinates between repositories, external APIs, and event system.
"""
import sys
import time
from decimal import Decimal
from typing import AsyncGenerator, Dict, List, Optional, Tuple
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
    
    def update_api_key(self, api_key: str) -> None:
        """Update OpenRouter API key.
        
        Args:
            api_key: New API key
        """
        self._openrouter.update_api_key(api_key)
    
    def create_conversation(
        self,
        title: Optional[str] = None,
        model_id: str = "meta/llama-3.1-8b-instruct",
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
        
        # Save conversation first (required for foreign key constraint)
        self._repo.save(conv)
        
        if system_prompt:
            system_msg = Message(
                id=uuid4(),
                conversation_id=conv.id,
                role=MessageRole.SYSTEM,
                content=system_prompt
            )
            self._repo.add_message(system_msg)
        
        return conv
    
    async def send_message(
        self,
        conversation_id: UUID,
        content: str,
        temperature: float = 0.7,
    ) -> AsyncGenerator[Tuple[str, Optional[Dict]], None]:
        """Send a message and stream the response.
        
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
        print(f"[DEBUG] Loading conversation {conversation_id}", file=sys.stderr)
        conv = self._repo.get_by_id(conversation_id)
        if not conv:
            print(f"[DEBUG] Conversation not found!", file=sys.stderr)
            raise ValidationError(f"Conversation {conversation_id} not found")
        
        print(f"[DEBUG] Loaded conversation: model={conv.model_id}", file=sys.stderr)
        
        # Get model config for pricing
        model_config = self.get_model_config(conv.model_id)
        print(f"[DEBUG] Model config: {model_config}", file=sys.stderr)
        
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
            print(f"[DEBUG] Starting stream from openrouter", file=sys.stderr)
            async for chunk in self._openrouter.chat_completion_stream(
                messages=api_messages,
                model=conv.model_id,
                temperature=temperature,
            ):
                print(f"[DEBUG] Received chunk from openrouter: {chunk}", file=sys.stderr)
                
                if chunk["type"] == "delta":
                    delta = chunk["content"]
                    assistant_content += delta
                    
                    # Emit token event for cost tracking
                    self._event_bus.emit(DomainEvent(
                        event_type=Events.TOKEN_GENERATED,
                        conversation_id=conversation_id,
                        payload={"delta": delta}
                    ))
                    
                    print(f"[DEBUG] Yielding delta: '{delta}'", file=sys.stderr)
                    yield (delta, None)
                
                elif chunk["type"] == "final":
                    final_usage = chunk.get("usage")
                    latency_ms = chunk.get("latency_ms", 0)
                    print(f"[DEBUG] Final chunk received, usage: {final_usage}", file=sys.stderr)
            
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
    
    def get_conversation(self, conversation_id: UUID) -> Optional[Conversation]:
        """Get conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation or None if not found
        """
        return self._repo.get_by_id(conversation_id)
    
    def get_conversation_messages(self, conversation_id: UUID) -> List[Message]:
        """Get all messages for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of messages
        """
        return self._repo.get_messages(conversation_id)
    
    def get_conversation_history(self, limit: int = 100) -> List[Conversation]:
        """Get conversation history.
        
        Args:
            limit: Maximum number of conversations
            
        Returns:
            List of conversations ordered by most recent
        """
        return self._repo.list_active(limit=limit)
