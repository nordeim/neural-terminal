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
    """OpenRouter model information."""
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
    """Resilient OpenRouter API client."""
    
    def __init__(self):
        """Initialize client with configuration."""
        self.base_url = settings.openrouter_base_url
        self.api_key = settings.openrouter_api_key.get_secret_value()
        self.timeout = settings.openrouter_timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    def update_api_key(self, api_key: str) -> None:
        """Update API key and reset client connection.
        
        Args:
            api_key: New OpenRouter API key
        """
        self.api_key = api_key
        # Close existing client so new one is created with updated key
        if self._client is not None and not self._client.is_closed:
            import asyncio
            try:
                asyncio.get_running_loop()
                # If in async context, schedule close
                asyncio.create_task(self._client.aclose())
            except RuntimeError:
                # No running loop, client will be recreated on next use
                pass
        self._client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
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
        """Fetch available models from OpenRouter."""
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
        """Streaming chat completion with SSE parsing."""
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
                raise RateLimitError(
                    retry_after=int(response.headers.get("retry-after", 60))
                )
            elif response.status_code == 503:
                raise ModelUnavailableError(model_id=model)
            elif response.status_code == 400:
                body = await response.aread()
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
                    chunk = json.loads(data)
                    
                    # Safely extract content with null checks
                    if chunk is None:
                        continue
                    
                    choices = chunk.get("choices", [])
                    if not choices or not isinstance(choices, list):
                        continue
                    
                    first_choice = choices[0] if choices else {}
                    if first_choice is None:
                        continue
                    
                    delta = first_choice.get("delta", {}) or {}
                    content = delta.get("content", "") if delta else ""
                    
                    if content:
                        full_content += content
                        yield {
                            "type": "delta",
                            "content": content,
                            "accumulated": full_content,
                        }
                    
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
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
