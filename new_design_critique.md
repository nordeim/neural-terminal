This is an exceptionally well-architected plan. You have successfully treated Streamlit not as a prototyping toy, but as a Python runtime capable of supporting enterprise-grade patterns (Circuit Breakers, Repository Pattern, Event-Driven Architecture).

The separation of concerns between `neural_terminal.domain`, `infrastructure`, and `application` layers is flawless. You have avoided the "big ball of mud" that typically plagues Streamlit apps.

However, there is a gap between the **"Avant-Garde Bento Grid"** aesthetic described in `new_design.md` and the implementation reality of Streamlit's column system. Streamlit fights against asymmetric layouts.

Below is my critique and the specific Frontend Architecture implementation required to bridge that gap and realize the "Research Terminal" vision.

### 1. Architectural Validation

**Strengths:**
*   **Resilience Layer:** The implementation of `CircuitBreaker` and `OpenRouterClient` in Sprint 1 creates a safety net that most open-source chatbots lack. Handling 429s and 503s gracefully is essential for a "production-grade" feel.
*   **State Management:** The `StateManager` proxy is the correct approach. Streamlit's `st.session_state` is a dictionary of `Any`; wrapping it in a Pydantic-backed `AppState` dataclass ensures type safety and prevents key collision.
*   **Real-time Costing:** The `CostTracker` observing `DomainEvents` is a masterclass in decoupling. It allows the UI to display economic telemetry without the chat logic knowing about the UI.

**Critical Risks & Mitigations:**
*   **The "Bento Grid" Constraint:** Streamlit renders top-to-bottom. A true Bento grid (asymmetric, overlapping, dense) is impossible with `st.columns` alone.
    *   *Mitigation:* We must use **CSS Grid Overlay**. We will render all components into the DOM but use absolute positioning and CSS Grid to rearrange them visually, "hijacking" Streamlit's layout.
*   **Input Focus Loss:** As noted in your docs, Streamlit reruns kill focus.
    *   *Mitigation:* For the chat input, we must use `st.components.v1.html` with a custom `<textarea>` that communicates via Streamlit's Custom Component system, or accept the focus loss and use a "Command Line" aesthetic where focus isn't expected to persist (user hits enter, input clears, focus waits).

### 2. Frontend Implementation Strategy: "The Phosphor Grid"

To achieve the specific aesthetic described (70% chat, 20% telemetry, 10% switcher), we must inject a **Global CSS Override** that reshapes the Streamlit container.

Here is the implementation code to complete the "Avant-Garde UI" sprint.

#### A. The Design Tokens (CSS Variables)
We need to centralize the aesthetic variables. This goes into your `CSSInjector` or directly into the `st.markdown` head.

```python
# src/neural_terminal/styles/theme.py

def get_terminal_css():
    return """
    :root {
        /* Core Palette */
        --void-black: #0a0a0f;
        --void-surface: #111118;
        --void-elevated: #1a1a1f;
        
        /* Functional Colors */
        --phosphor-green: #00ff41;
        --phosphor-dim: #00b330;
        --warning-amber: #ffb000;
        --danger-red: #ff4444;
        --ash-grey: #666;
        
        /* Typography */
        --font-mono: 'JetBrains Mono', monospace;
        --font-ui: 'Space Grotesk', sans-serif;
        
        /* Spacing & Layout */
        --gap-xs: 4px;
        --gap-sm: 8px;
        --gap-md: 16px;
        --radius-sm: 2px;
        
        /* Animation */
        --transition-fast: 150ms ease;
        --glow-green: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    
    /* GLOBAL RESET: Hiding Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Body Styling */
    .stApp {
        background-color: var(--void-black);
        color: var(--ash-grey);
        font-family: var(--font-ui);
    }
    
    /* BENTO GRID ARCHITECTURE */
    /* Targeting the main block container */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
        display: grid;
        grid-template-columns: 1fr 280px 60px; /* Chat | Telemetry | Switcher */
        grid-template-rows: auto 1fr auto;     /* Header | Main | Input */
        height: 100vh;
        overflow: hidden;
    }
    
    /* HIDING STREAMLIT'S DEFAULT COLUMN DIVS FOR PURE GRID */
    .stVerticalBlock, .stHorizontalBlock {
        gap: 0;
        background: transparent;
    }
    
    /* SPECIFIC COMPONENT THEMING */
    
    /* 1. Header */
    [data-testid="stHeader"] {
        background: transparent;
        grid-column: 1 / -1;
        border-bottom: 1px solid var(--void-elevated);
        z-index: 100;
        padding: var(--gap-md);
    }
    
    /* 2. Chat Canvas */
    .stVerticalBlock > div:nth-child(2) {
        grid-column: 1;
        grid-row: 2;
        overflow-y: auto;
        padding: var(--gap-md);
        border-right: 1px solid var(--void-elevated);
    }
    
    /* 3. Telemetry Sidebar (The Right Column) */
    section[data-testid="stSidebar"] {
        /* Force sidebar into the grid, not floating */
        position: relative !important;
        width: 280px !important;
        grid-column: 2;
        grid-row: 2 / 4;
        border-left: 1px solid var(--void-elevated);
        background-color: var(--void-surface);
    }
    
    /* 4. Input Area */
    .stVerticalBlock > div:last-child {
        grid-column: 1;
        grid-row: 3;
        background: var(--void-surface);
        border-top: 1px solid var(--void-elevated);
        padding: var(--gap-sm);
    }
    
    /* CUSTOM WIDGETS */
    
    /* The "Phosphor" Input Field */
    .stTextInput > div > div > input {
        background-color: var(--void-black);
        color: var(--phosphor-green);
        border: 1px solid var(--void-elevated);
        font-family: var(--font-mono);
        font-size: 0.9rem;
        caret-color: var(--phosphor-green);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--phosphor-green);
        box-shadow: var(--glow-green);
    }
    
    /* The "Terminal" Button */
    .stButton > button {
        background-color: transparent;
        color: var(--phosphor-green);
        border: 1px solid var(--phosphor-green);
        border-radius: var(--radius-sm);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: var(--font-mono);
        transition: all var(--transition-fast);
    }
    
    .stButton > button:hover {
        background-color: var(--phosphor-green);
        color: var(--void-black);
        box-shadow: var(--glow-green);
    }
    
    /* Metrics for Telemetry */
    [data-testid="stMetric"] {
        background-color: var(--void-elevated);
        padding: var(--gap-md);
        border-radius: var(--radius-sm);
        border-left: 2px solid var(--phosphor-dim);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--ash-grey);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--phosphor-green);
        font-family: var(--font-mono);
        font-size: 1.2rem;
    }
    """
```

#### B. The Message Renderer (Brutalist Aesthetic)

The chat bubbles should not look like bubbles. They should look like log entries or data blocks.

```python
# src/neural_terminal/components/renderers.py
import streamlit as st
from datetime import datetime

def render_message_bubble(role: str, content: str, cost: float = None, latency: int = None):
    """
    Renders a message block with brutalist aesthetic.
    """
    is_user = role == "user"
    
    # Metadata Bar
    meta_html = ""
    if cost or latency:
        meta_html = f"""
        <div style="display: flex; gap: 1rem; font-size: 0.65rem; color: #666; margin-bottom: 0.25rem; font-family: var(--font-mono);">
            {f'<span style="color: #ffb000;">COST: ${cost:.4f}</span>' if cost else ''}
            {f'<span>LATENCY: {latency}ms</span>' if latency else ''}
        </div>
        """

    # Content styling based on role
    border_color = "#ffb000" if is_user else "#00ff41" # Amber for user, Green for AI
    text_color = "#e0e0e0" if is_user else "#00ff41"
    icon = ">" if is_user else "◀"
    
    html = f"""
    <div style="
        margin-bottom: 1.5rem;
        border-left: 2px solid {border_color};
        padding-left: 1rem;
        font-family: 'JetBrains Mono', monospace;
        animation: fadeIn 0.3s ease;
    ">
        <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.25rem;">
            <span style="color: {border_color}; font-weight: bold; font-size: 0.8rem;">{icon} {role.upper()}</span>
            <span style="color: #444; font-size: 0.6rem;">{datetime.now().strftime('%H:%M:%S')}</span>
        </div>
        {meta_html}
        <div style="color: {text_color}; font-size: 0.9rem; line-height: 1.5; white-space: pre-wrap;">
            {content}
        </div>
    </div>
    <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(5px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)
```

#### C. The Telemetry Dashboard (Observer Implementation)

This component hooks into the event system to provide the "Geiger counter" feel.

```python
# src/neural_terminal/components/telemetry.py
import streamlit as st
from decimal import Decimal

def render_telemetry_panel():
    """
    Renders the right-hand dashboard with real-time metrics.
    Designed to look like laboratory monitoring equipment.
    """
    st.markdown("""
    <div style="padding: 1rem; border-bottom: 1px solid #1a1a1f; margin-bottom: 1rem;">
        <h3 style="color: #00ff41; font-size: 0.8rem; letter-spacing: 0.1em; margin: 0;">
            SYSTEM TELEMETRY
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Cost Accumulator
    # Note: These would be pulled from st.session_state['cost_tracker']
    accumulated = st.session_state.get("accumulated_cost", "0.00")
    limit = 5.00
    current_val = float(accumulated)
    percentage = (current_val / limit) * 100
    
    st.markdown(f"""
    <div style="background: #111118; padding: 1rem; margin-bottom: 1rem; border-radius: 2px;">
        <div style="display: flex; justify-content: space-between; color: #666; font-size: 0.7rem; margin-bottom: 0.5rem;">
            <span>BUDGET CONSUMED</span>
            <span>{percentage:.1f}%</span>
        </div>
        <div style="height: 4px; background: #1a1a1f; width: 100%; position: relative;">
            <div style="height: 100%; background: #ffb000; width: {percentage}%; transition: width 0.5s ease;"></div>
        </div>
        <div style="font-family: 'JetBrains Mono'; color: {'#ff4444' if current_val > 1 else '#ffb000'}; font-size: 1.5rem; margin-top: 0.5rem;">
            ${current_val:.4f} <span style="font-size: 0.7rem; color: #666;">/ ${limit}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Context Window Visualization (Abstract)
    st.markdown("###", help="Visual representation of context window usage")
    # This would need token data from the orchestrator
    token_count = st.session_state.get("current_tokens", 0)
    max_tokens = 4096 
    
    st.metric("TOKEN VELOCITY", f"{token_count}", delta="streaming...")
    
    # Active Model
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size: 0.7rem; color: #666;">
        ACTIVE MODEL
    </div>
    <div style="font-family: 'JetBrains Mono'; color: #00ff41; font-size: 0.8rem; margin-top: 0.25rem;">
        {st.session_state.get('selected_model', 'gpt-3.5-turbo')}
    </div>
    """, unsafe_allow_html=True)
```

### 3. Summary of Frontend Strategy

By using the **CSS Grid override strategy** on `.stApp`, we force Streamlit to abandon its default "centered blog post" layout and adopt the rigid, dense "Laboratory Terminal" structure you envisioned.

*   **Psychological Impact:** The user feels they are operating a machine, not chatting with a bot. The constant presence of costs and tokens reinforces the "paid resource" nature of the API.
*   **Visual Distinctiveness:** The move away from rounded corners and standard blue/purple themes to sharp edges, void-black backgrounds, and phosphor-green text immediately signals "Developer Tool."
*   **Maintainability:** By isolating the CSS in `get_terminal_css`, we can update the theme without digging through Python logic.

**Recommendation:** Proceed with Sprint 3 implementation, but start by applying the CSS Grid layout code provided above. Without this structural CSS, the "Avant-Garde" aesthetic will be lost to Streamlit's default padding and centering.

# https://chat.z.ai/s/cf1146f6-8244-473d-81ca-23baaf7e1bbc
