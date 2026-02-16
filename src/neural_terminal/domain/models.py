"""Domain models for Neural Terminal.

Phase 0 Defect C-1 Fix: TokenUsage.cost converted from property to method.
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationStatus(str, Enum):
    """Conversation status enumeration."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    FORKED = "forked"


@dataclass(frozen=True)
class TokenUsage:
    """Immutable token consumption metrics.
    
    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens (prompt + completion)
    
    Phase 0 Defect C-1 Fix:
        calculate_cost() is a REGULAR METHOD, not a property.
        Properties cannot accept arguments in Python.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    def calculate_cost(
        self, 
        price_per_1k_prompt: Decimal, 
        price_per_1k_completion: Decimal
    ) -> Decimal:
        """Calculate cost based on pricing.
        
        Args:
            price_per_1k_prompt: Price per 1000 prompt tokens
            price_per_1k_completion: Price per 1000 completion tokens
            
        Returns:
            Total cost as Decimal with full precision
            
        Example:
            >>> usage = TokenUsage(1000, 500, 1500)
            >>> usage.calculate_cost(Decimal("0.0015"), Decimal("0.002"))
            Decimal("0.0025")
        """
        prompt_cost = (Decimal(self.prompt_tokens) / 1000) * price_per_1k_prompt
        completion_cost = (Decimal(self.completion_tokens) / 1000) * price_per_1k_completion
        return prompt_cost + completion_cost


@dataclass
class Message:
    """Domain entity for chat messages.
    
    Attributes:
        id: Unique message identifier
        conversation_id: Parent conversation ID
        role: Message role (user/assistant/system)
        content: Message text content
        token_usage: Token consumption metrics (optional)
        cost: Calculated cost in USD (optional)
        latency_ms: Response latency in milliseconds (optional)
        model_id: Model used for generation (optional)
        created_at: Timestamp of creation
        metadata: Additional metadata dict
    """
    id: UUID = field(default_factory=uuid4)
    conversation_id: Optional[UUID] = None
    role: MessageRole = MessageRole.USER
    content: str = ""
    token_usage: Optional[TokenUsage] = None
    cost: Optional[Decimal] = None
    latency_ms: Optional[int] = None
    model_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Aggregate root for conversations.
    
    Phase 0 Defect H-1 Fix:
        update_cost() uses simple assignment (dataclass is not frozen).
        object.__setattr__ bypass is unnecessary and misleading.
    
    Phase 0 Defect H-4 Fix:
        to_dict() method added for session state serialization.
    
    Attributes:
        id: Unique conversation identifier
        title: Optional conversation title
        model_id: Default model for this conversation
        status: Conversation status
        created_at: Timestamp of creation
        updated_at: Timestamp of last update
        total_cost: Accumulated cost in USD
        total_tokens: Accumulated token count
        parent_conversation_id: ID of parent conversation (for forking)
        tags: List of string tags
    """
    id: UUID = field(default_factory=uuid4)
    title: Optional[str] = None
    model_id: str = "meta/llama-3.1-8b-instruct"  # Default
    status: ConversationStatus = ConversationStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    total_cost: Decimal = field(default_factory=lambda: Decimal("0.00"))
    total_tokens: int = 0
    parent_conversation_id: Optional[UUID] = None
    tags: List[str] = field(default_factory=list)
    
    def update_cost(self, message_cost: Decimal) -> None:
        """Atomic cost update.
        
        Args:
            message_cost: Cost to add to total
            
        Note:
            Simple assignment works because this dataclass is NOT frozen.
            object.__setattr__ bypass is unnecessary.
        """
        self.total_cost += message_cost
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation for session state storage.
        
        Phase 0 Defect H-4 Fix:
            Properly serializes UUID and Decimal types that don't JSON serialize natively.
        
        Returns:
            Dict with string-serialized values
        """
        return {
            "id": str(self.id),
            "title": self.title,
            "model_id": self.model_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_cost": str(self.total_cost),
            "total_tokens": self.total_tokens,
            "parent_conversation_id": str(self.parent_conversation_id) if self.parent_conversation_id else None,
            "tags": self.tags,
        }
