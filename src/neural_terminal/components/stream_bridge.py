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
                # Reset any clients that may have been bound to old event loop
                _reset_async_clients()
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


def _reset_async_clients():
    """Reset async clients that may be bound to a different event loop.
    
    This is called before running async code in a new thread to ensure
    clients are recreated for the new event loop.
    """
    try:
        # Import here to avoid circular imports
        from neural_terminal.app_state import get_app_state
        
        app_state = get_app_state()
        if app_state._orchestrator and hasattr(app_state._orchestrator, '_openrouter'):
            openrouter = app_state._orchestrator._openrouter
            if hasattr(openrouter, '_client') and openrouter._client is not None:
                # Close and reset the client so it's recreated with new event loop
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Schedule close if loop is running
                        asyncio.create_task(openrouter._client.aclose())
                except Exception:
                    pass
                openrouter._client = None
    except Exception:
        # Ignore errors during reset - client will be recreated anyway
        pass


class StreamlitStreamBridge:
    """Bridges async generators to Streamlit's synchronous world.
    
    Uses producer-consumer pattern with threading to prevent blocking
    Streamlit's execution while waiting for async operations.
    
    Example:
        bridge = StreamlitStreamBridge(placeholder)
        metadata = bridge.stream(orchestrator.send_message(...))
    """
    
    def __init__(self):
        """Initialize bridge."""
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
