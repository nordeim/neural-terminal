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
        
        # Find matching encoding key in model name
        encoding_key = "default"
        for key in self.ENCODING_MAP:
            if key in base and key != "default":
                encoding_key = key
                break
        
        # Get actual encoding name from map
        encoding_name = self.ENCODING_MAP[encoding_key]
        
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
