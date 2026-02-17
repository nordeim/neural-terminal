"""Main application orchestration for Neural Terminal.

Provides the high-level application coordination between
UI components and business logic.
"""

import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from decimal import Decimal

import streamlit as st

from neural_terminal.app_state import ApplicationState, get_app_state, AppConfig
from neural_terminal.components.styles import StyleManager, inject_css
from neural_terminal.components.themes import ThemeRegistry
from neural_terminal.components.chat_container import ChatContainer, MessageViewModel
from neural_terminal.components.header import Header, HeaderConfig, Sidebar
from neural_terminal.components.status_bar import StatusBar, StatusInfo, CostDisplay
from neural_terminal.components.message_renderer import MessageRenderer
from neural_terminal.components.error_handler import ErrorHandler, ErrorSeverity


class NeuralTerminalApp:
    """Main Neural Terminal application class.
    
    Coordinates all UI components and business logic to provide
    a complete chat interface.
    """
    
    # Available models (NVIDIA API compatible)
    AVAILABLE_MODELS = [
        ("z-ai/glm5", "GLM 5 (NVIDIA Recommended)"),
        ("meta/llama-3.1-8b-instruct", "Llama 3.1 8B"),
        ("meta/llama-3.1-70b-instruct", "Llama 3.1 70B"),
        ("meta/llama-3.1-405b-instruct", "Llama 3.1 405B"),
        ("meta/llama-3.3-70b-instruct", "Llama 3.3 70B"),
        ("meta/llama-4-scout-17b-16e-instruct", "Llama 4 Scout"),
        ("meta/llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick"),
        ("mistralai/mistral-7b-instruct-v0.2", "Mistral 7B"),
        ("mistralai/mixtral-8x7b-instruct-v0.1", "Mixtral 8x7B"),
        ("mistralai/codestral-22b-instruct-v0.1", "Codestral 22B"),
        ("qwen/qwen2.5-7b-instruct", "Qwen 2.5 7B"),
        ("qwen/qwen2.5-72b-instruct", "Qwen 2.5 72B"),
        ("01-ai/yi-large", "Yi Large"),
        ("google/gemma-2-9b-it", "Gemma 2 9B"),
        ("google/gemma-2-27b-it", "Gemma 2 27B"),
    ]
    
    def __init__(self):
        """Initialize application."""
        self._app_state = get_app_state()
        self._style_manager = StyleManager()
        self._chat_container = ChatContainer()
        self._header = Header(HeaderConfig())
        self._sidebar = Sidebar()
        self._status_bar = StatusBar()
        self._cost_display = CostDisplay()
        self._error_handler = ErrorHandler()
        self._message_renderer = MessageRenderer()
    
    def setup(self) -> None:
        """Setup application on first run."""
        # Page configuration
        st.set_page_config(
            page_title="Neural Terminal",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        
        # Initialize app state
        try:
            self._app_state.initialize()
        except Exception as e:
            self._error_handler.show_startup_error(str(e))
            return
        
        # Apply theme
        theme_name = self._app_state.config.theme
        try:
            theme = ThemeRegistry.get_theme(theme_name)
            inject_css(theme)
        except Exception:
            inject_css()  # Use default
    
    def run(self) -> None:
        """Run the main application loop."""
        # Setup
        self.setup()
        
        # Check for initialization errors
        if self._app_state.session.error_message:
            self._error_handler.show_error_message(
                f"⚠️ {self._app_state.session.error_message}",
                ErrorSeverity.ERROR
            )
            return
        
        # Render sidebar
        self._render_sidebar()
        
        # Render main content based on current page
        if self._app_state.session.current_page == "chat":
            self._render_chat_page()
        elif self._app_state.session.current_page == "settings":
            self._render_settings_page()
        else:
            self._render_chat_page()
    
    def _render_sidebar(self) -> None:
        """Render sidebar with conversation list and settings."""
        with st.sidebar:
            st.title("⚡ Neural Terminal")
            
            # System Prompt configuration
            with st.expander("📝 System Prompt", expanded=False):
                system_prompt = st.text_area(
                    "System Prompt",
                    value=self._app_state.config.system_prompt,
                    placeholder="Enter a system prompt to guide the AI's behavior...",
                    label_visibility="collapsed",
                    height=100,
                    key="sidebar_system_prompt",
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save", use_container_width=True, key="save_sys_prompt"):
                        self._app_state.update_config(system_prompt=system_prompt)
                        st.success("Saved!")
                with col2:
                    if st.button("🗑️ Clear", use_container_width=True, key="clear_sys_prompt"):
                        self._app_state.update_config(system_prompt="")
                        st.rerun()
                
                st.caption("Applied to new conversations only")
            
            # New conversation button - uses system prompt from config
            if st.button("➕ New Conversation", use_container_width=True):
                self._app_state.create_conversation(
                    system_prompt=self._app_state.config.system_prompt or None
                )
                st.rerun()
            
            st.divider()
            
            # Conversation list
            st.subheader("Conversations")
            
            conversations = self._app_state.session.conversations
            
            if not conversations:
                st.caption("No conversations yet")
            else:
                for conv in conversations:
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        title = conv.get("title", "Untitled")[:30]
                        is_current = conv["id"] == self._app_state.session.current_conversation_id
                        
                        button_type = "primary" if is_current else "secondary"
                        if st.button(
                            title,
                            key=f"conv_{conv['id']}",
                            use_container_width=True,
                            type=button_type,
                        ):
                            self._app_state.set_current_conversation(conv["id"])
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️", key=f"del_{conv['id']}", help="Delete"):
                            self._app_state.delete_conversation(conv["id"])
                            st.rerun()
            
            st.divider()
            
            # Cost summary
            st.subheader("Session Cost")
            total_cost = self._app_state.session.total_cost
            budget = self._app_state.config.budget_limit
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Spent", f"${float(total_cost):.4f}")
            with col2:
                if budget:
                    remaining = budget - total_cost
                    st.metric("Remaining", f"${float(remaining):.4f}")
                else:
                    st.metric("Budget", "∞")
            
            if budget and budget > 0:
                progress = float(total_cost / budget)
                st.progress(min(progress, 1.0), text=f"{progress*100:.1f}% used")
            
            st.divider()
            
            # Navigation
            st.subheader("Navigation")
            
            if st.button("💬 Chat", use_container_width=True):
                self._app_state.session.current_page = "chat"
                st.rerun()
            
            if st.button("⚙️ Settings", use_container_width=True):
                self._app_state.session.current_page = "settings"
                st.rerun()
            
            # Help expander
            with st.expander("❓ Help"):
                st.markdown("""
                **Keyboard Shortcuts:**
                - Enter: Send message
                - Shift+Enter: New line
                
                **Tips:**
                - Use code blocks with ```language
                - Set a budget to track costs
                - Switch models anytime
                """)
    
    def _render_chat_page(self) -> None:
        """Render main chat interface."""
        # Header
        is_connected = self._app_state.orchestrator is not None
        
        self._header.render(
            is_connected=is_connected,
            available_models=self.AVAILABLE_MODELS,
            selected_model=self._app_state.config.default_model,
            on_model_change=self._on_model_change,
        )
        
        # Status bar
        status = StatusInfo(
            total_cost=self._app_state.session.total_cost,
            budget_limit=self._app_state.config.budget_limit,
            total_tokens=self._app_state.session.total_tokens,
            message_count=self._app_state.session.message_count,
            current_model=self._app_state.config.default_model,
            is_streaming=self._app_state.session.is_streaming,
            connection_status="connected" if is_connected else "disconnected",
        )
        
        self._status_bar.render_compact(status)
        self._status_bar.render_budget_warning(status)
        
        st.divider()
        
        # Chat messages area
        messages = self._app_state.get_conversation_messages()
        
        if not messages and not self._app_state.session.current_conversation_id:
            # Welcome message
            self._render_welcome()
        else:
            # Display conversation
            self._render_messages(messages)
        
        # Show streaming content if active
        if self._app_state.session.is_streaming:
            self._chat_container.render_streaming_message(
                self._app_state.session.streaming_content,
            )
        
        # Input area
        self._render_input()
    
    def _render_welcome(self) -> None:
        """Render welcome message for new users."""
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <h1 style="color: var(--nt-accent-primary);">⚡ Welcome to Neural Terminal</h1>
            <p style="color: var(--nt-text-secondary); font-size: 1.2rem;">
                Your production-grade AI chat interface
            </p>
            <br>
            <p>
                Start a new conversation from the sidebar or just type a message below.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Render message history.
        
        Args:
            messages: List of message dictionaries
        """
        for msg in messages:
            # Safety check for None messages
            if msg is None:
                continue
            
            view_model = MessageViewModel(
                role=msg.get("role", "assistant"),
                content=msg.get("content", ""),
                cost=Decimal(msg.get("cost", "0")),
                tokens=msg.get("tokens", 0),
            )
            self._chat_container.render_message(view_model)
    
    def _render_input(self) -> None:
        """Render message input area."""
        st.divider()
        
        # Check if we need to clear input from previous message
        if st.session_state.get("_clear_input_on_next_render", False):
            # Clear the input by setting it to empty in session state
            # This works because it happens BEFORE the widget is created
            st.session_state["message_input"] = ""
            st.session_state["_clear_input_on_next_render"] = False
        
        with st.container():
            col1, col2 = st.columns([6, 1])
            
            with col1:
                prompt = st.text_area(
                    "Message",
                    placeholder="Type your message here... (Shift+Enter for new line)",
                    label_visibility="collapsed",
                    height=80,
                    key="message_input",
                )
            
            with col2:
                st.write("")  # Spacer
                st.write("")
                
                # Simple Save button approach - always active, grab content on click
                if st.button(
                    "Send",
                    use_container_width=True,
                    disabled=False,  # Always active - no state management
                    type="primary",
                ):
                    # Debug: Log button click
                    import sys
                    print(f"[DEBUG] Send button clicked!", file=sys.stderr)
                    
                    # Grab whatever is in the input box right now
                    current_content = st.session_state.get("message_input", "")
                    print(f"[DEBUG] Current input content: '{current_content}'", file=sys.stderr)
                    
                    # Only send if there's actual content
                    if current_content.strip():
                        print(f"[DEBUG] Sending message: '{current_content}'", file=sys.stderr)
                        self._handle_send_message(current_content)
                    else:
                        print(f"[DEBUG] No content to send, ignoring click", file=sys.stderr)
    
    def _handle_send_message(self, content: str) -> None:
        """Handle sending a message.
        
        Args:
            content: Message content
        """
        import sys
        print(f"[DEBUG] _handle_send_message called with: '{content}'", file=sys.stderr)
        
        if not content.strip():
            print(f"[DEBUG] Early return - content is empty after strip", file=sys.stderr)
            return
        
        print(f"[DEBUG] Current session state:", file=sys.stderr)
        print(f"  - is_streaming: {self._app_state.session.is_streaming}", file=sys.stderr)
        print(f"  - current_conversation_id: {self._app_state.session.current_conversation_id}", file=sys.stderr)
        print(f"  - error_message: {self._app_state.session.error_message}", file=sys.stderr)
        
        # Run async message sending
        try:
            print(f"[DEBUG] About to call _run_async_send", file=sys.stderr)
            self._run_async_send(content)
        except Exception as e:
            print(f"[DEBUG] Exception in _handle_send_message: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self._error_handler.show_error(f"Failed to send message: {e}")
    
    def _run_async_send(self, content: str) -> None:
        """Run async message sending in sync context.
        
        Args:
            content: Message content
        """
        import sys
        print(f"[DEBUG] _run_async_send called with: '{content}'", file=sys.stderr)
        
        from neural_terminal.components.stream_bridge import run_async
        
        async def send():
            print(f"[DEBUG] Async send function starting", file=sys.stderr)
            chunks = []
            try:
                async for chunk in self._app_state.send_message(content):
                    print(f"[DEBUG] Received chunk: '{chunk}'", file=sys.stderr)
                    chunks.append(chunk)
                result = "".join(chunks)
                print(f"[DEBUG] Final result: '{result}'", file=sys.stderr)
                return result
            except Exception as e:
                print(f"[DEBUG] Exception in async send: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                raise
        
        try:
            print(f"[DEBUG] About to call run_async", file=sys.stderr)
            result = run_async(send())
            print(f"[DEBUG] run_async completed, result: '{result}'", file=sys.stderr)
            
            # Clear input after successful send
            if result.strip():
                st.session_state["_clear_input_on_next_render"] = True
            
            st.rerun()
        except Exception as e:
            print(f"[DEBUG] Exception in _run_async_send: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise e
    
    def _on_model_change(self, model: str) -> None:
        """Handle model selection change.
        
        Args:
            model: Selected model ID
        """
        self._app_state.update_config(default_model=model)
        st.toast(f"Switched to {model}")
    
    def _render_settings_page(self) -> None:
        """Render settings configuration page."""
        st.title("⚙️ Settings")
        
        # API Configuration (Read-only from .env file)
        st.subheader("API Configuration (from .env)")
        
        # Import settings to display current env values
        from neural_terminal.config import settings
        
        # Display masked API key
        api_key_value = settings.openrouter_api_key.get_secret_value()
        masked_key = api_key_value[:8] + "..." + api_key_value[-4:] if len(api_key_value) > 12 else "***"
        
        st.text_input(
            "OpenRouter API Key",
            value=masked_key,
            disabled=True,
            help="Configured via OPENROUTER_API_KEY in .env file",
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input(
                "Base URL",
                value=settings.openrouter_base_url,
                disabled=True,
                help="Configured via OPENROUTER_BASE_URL in .env file",
            )
        with col2:
            st.number_input(
                "Timeout (seconds)",
                value=settings.openrouter_timeout,
                disabled=True,
                help="Configured via OPENROUTER_TIMEOUT in .env file",
            )
        
        st.caption("🔒 API settings are loaded from environment variables (.env file) and cannot be changed via UI.")
        
        # Model Settings
        st.subheader("Model Settings")
        
        model = st.selectbox(
            "Default Model",
            options=[m[0] for m in self.AVAILABLE_MODELS],
            format_func=lambda x: next((m[1] for m in self.AVAILABLE_MODELS if m[0] == x), x),
            index=[m[0] for m in self.AVAILABLE_MODELS].index(
                self._app_state.config.default_model
            ) if self._app_state.config.default_model in [m[0] for m in self.AVAILABLE_MODELS] else 0,
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=self._app_state.config.temperature,
            step=0.1,
            help="Higher = more creative, Lower = more deterministic",
        )
        
        max_tokens = st.number_input(
            "Max Tokens per Message",
            min_value=100,
            max_value=65535,
            value=self._app_state.config.max_tokens_per_message,
            step=100,
        )
        
        # System Prompt (Default)
        st.subheader("Default System Prompt")
        
        default_system_prompt = st.text_area(
            "Default System Prompt",
            value=self._app_state.config.system_prompt,
            placeholder="Enter a default system prompt for new conversations...",
            label_visibility="collapsed",
            height=100,
            help="This system prompt will be applied to all new conversations",
        )
        
        # Budget Settings
        st.subheader("Budget Settings")
        
        enable_budget = st.checkbox(
            "Enable Budget Limit",
            value=self._app_state.config.budget_limit is not None,
        )
        
        if enable_budget:
            budget = st.number_input(
                "Budget Limit ($)",
                min_value=0.1,
                max_value=1000.0,
                value=float(self._app_state.config.budget_limit or 10.0),
                step=0.5,
            )
            budget_decimal = Decimal(str(budget))
        else:
            budget_decimal = None
        
        # Theme Settings
        st.subheader("Appearance")
        
        theme = st.selectbox(
            "Theme",
            options=["terminal", "amber", "minimal"],
            format_func=lambda x: {
                "terminal": "Terminal Green",
                "amber": "Cyberpunk Amber",
                "minimal": "Minimal Dark",
            }.get(x, x),
            index=["terminal", "amber", "minimal"].index(self._app_state.config.theme),
        )
        
        # Save button
        st.divider()
        
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            self._app_state.update_config(
                default_model=model,
                temperature=temperature,
                max_tokens_per_message=max_tokens,
                budget_limit=budget_decimal,
                theme=theme,
                system_prompt=default_system_prompt,
            )
            
            st.success("Settings saved!")
            
            # Apply theme immediately
            if theme != self._app_state.config.theme:
                st.rerun()


def main() -> None:
    """Main application entry point."""
    app = NeuralTerminalApp()
    app.run()


if __name__ == "__main__":
    main()
