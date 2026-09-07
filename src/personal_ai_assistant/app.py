"""Streamlit dashboard + chat UI for the Student Study & Productivity Assistant.

Run with:  uv run streamlit run src/personal_ai_assistant/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
os.environ.setdefault("PYTHONUTF8", "1")

# On Streamlit Community Cloud, config comes from the Secrets panel. Mirror it
# into environment variables so CrewAI / LiteLLM pick it up.
for _key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_NAME"):
    try:
        if _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
    except Exception:  # noqa: BLE001 - no secrets file locally is fine
        pass

from personal_ai_assistant.crew import PersonalAiAssistant  # noqa: E402
from personal_ai_assistant.rag.store import DOCS_DIR, get_knowledge_base  # noqa: E402
from personal_ai_assistant.tools.personal_memory_tool import memory_facts  # noqa: E402

MAX_TURNS_IN_CONTEXT = 6

TOOLS_CATALOG = [
    {
        "icon": "📚",
        "name": "Study Material Search",
        "category": "Local RAG",
        "desc": "Semantic vector search over your indexed lecture notes, summaries, and course PDFs.",
        "example": "What does chapter 3 say about gradient descent?",
    },
    {
        "icon": "📄",
        "name": "Read Document",
        "category": "File Analysis",
        "desc": "Directly parses and inspects any specific PDF, CSV, or TXT file path in depth.",
        "example": "Summarise C:/files/syllabus.pdf",
    },
    {
        "icon": "🌐",
        "name": "Web Search",
        "category": "Live Web",
        "desc": "Fetches current articles, documentation, and real-time facts using DuckDuckGo.",
        "example": "Who won the 2024 Nobel Prize in Physics?",
    },
    {
        "icon": "📖",
        "name": "Wikipedia",
        "category": "Reference",
        "desc": "Pulls authoritative encyclopedic summaries, definitions, and historical context.",
        "example": "Explain reinforcement learning from human feedback",
    },
    {
        "icon": "🌦️",
        "name": "Live Weather",
        "category": "Utility",
        "desc": "Provides current temperature, wind speed, and weather forecasts via Open-Meteo.",
        "example": "Weather in Hyderabad right now",
    },
    {
        "icon": "📅",
        "name": "Date & Time",
        "category": "Productivity",
        "desc": "Computes relative deadlines, upcoming dates, weekday schedules, and time offsets.",
        "example": "What date is 10 days from today?",
    },
    {
        "icon": "🧮",
        "name": "Math Calculator",
        "category": "Computation",
        "desc": "Safe AST-evaluated arithmetic engine supporting powers, logarithms, and roots.",
        "example": "15% of 3200 plus sqrt(169)",
    },
    {
        "icon": "🧠",
        "name": "Personal Memory",
        "category": "Profile Memory",
        "desc": "Recalls your stored student profile, courses, preferences, and adds new durable facts.",
        "example": "Remember that my machine learning exam is on 15 October",
    },
]

SUGGESTIONS = [
    ("📚", "What topics are on my mid-term exam?"),
    ("📊", "Explain the bias-variance tradeoff from my notes"),
    ("🧮", "What is 15% of 3200 plus the square root of 169?"),
    ("🧠", "Remember that my ML project is due next Friday"),
]

st.set_page_config(
    page_title="Student Study & Productivity Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

      html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0f172a;
      }

      header { visibility: hidden !important; height: 0 !important; }
      #MainMenu, footer { visibility: hidden !important; }

      .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1140px;
      }

      /* Top Header */
      .dashboard-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.15rem;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid #e2e8f0;
      }

      .brand-title-wrap h1 {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin: 0 0 0.2rem 0;
      }

      .brand-subtitle {
        color: #64748b;
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.4;
      }

      .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        color: #334155;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.32rem 0.75rem;
        border-radius: 9999px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      }

      .badge-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 0 2px #d1fae5;
      }

      /* KPI Cards */
      .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
        margin-bottom: 1.25rem;
      }

      @media (max-width: 900px) {
        .kpi-grid {
          grid-template-columns: repeat(2, 1fr);
        }
      }

      .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.95rem 1.05rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        position: relative;
        overflow: hidden;
      }

      .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
        border-color: #cbd5e1;
      }

      .kpi-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.35rem;
      }

      .kpi-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #64748b;
      }

      .kpi-icon-badge {
        width: 28px;
        height: 28px;
        border-radius: 7px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
      }

      .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
      }

      .kpi-sub {
        font-size: 0.8rem;
        color: #64748b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }

      /* Modern Pill Segmented Tabs */
      .stTabs [data-baseweb="tab-list"],
      div[data-testid="stTabs"] [data-baseweb="tab-list"],
      div[role="tablist"] {
        gap: 6px !important;
        background-color: #f1f5f9 !important;
        padding: 5px !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        margin-bottom: 1.1rem !important;
      }

      .stTabs [data-baseweb="tab"],
      div[data-testid="stTabs"] [data-baseweb="tab"],
      button[role="tab"] {
        height: auto !important;
        white-space: pre-wrap !important;
        background-color: transparent !important;
        border-radius: 8px !important;
        color: #64748b !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        border: none !important;
        transition: all 0.15s ease !important;
      }

      .stTabs [data-baseweb="tab"]:hover,
      button[role="tab"]:hover {
        color: #0f172a !important;
        background-color: rgba(255, 255, 255, 0.6) !important;
      }

      .stTabs [aria-selected="true"],
      button[role="tab"][aria-selected="true"] {
        background-color: #ffffff !important;
        color: #4f46e5 !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08) !important;
      }

      .stTabs [data-baseweb="tab-border"],
      .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
      }

      /* Empty State */
      .empty-state-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.8rem 1.4rem;
        text-align: center;
        margin-bottom: 1.1rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
      }

      .empty-avatar-circle {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: #eef2ff;
        color: #4f46e5;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 0 0 5px #f5f3ff;
      }

      .empty-state-title {
        font-size: 1.15rem;
        font-weight: 750;
        color: #0f172a;
        margin: 0 0 0.3rem 0;
      }

      .empty-state-desc {
        color: #64748b;
        font-size: 0.88rem;
        max-width: 540px;
        margin: 0 auto;
        line-height: 1.45;
      }

      /* Suggestion Buttons */
      div[data-testid="stButton"] > button {
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        background: #ffffff !important;
        color: #1e293b !important;
        font-weight: 500 !important;
        font-size: 0.86rem !important;
        padding: 0.7rem 1rem !important;
        text-align: left !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02) !important;
        width: 100% !important;
      }

      div[data-testid="stButton"] > button:hover {
        border-color: #4f46e5 !important;
        background: #faf5ff !important;
        color: #4f46e5 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.08) !important;
      }

      /* Chat Messages */
      div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.1rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02) !important;
      }

      /* Tool Cards */
      .tool-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem;
        height: 100%;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: all 0.2s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }

      .tool-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
        transform: translateY(-2px);
      }

      .tool-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.4rem;
      }

      .tool-title-group {
        display: flex;
        align-items: center;
        gap: 0.55rem;
      }

      .tool-icon {
        font-size: 1.25rem;
        line-height: 1;
      }

      .tool-name {
        font-size: 0.96rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
      }

      .tool-tag {
        font-size: 0.72rem;
        font-weight: 600;
        color: #4f46e5;
        background: #eef2ff;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
      }

      .tool-desc {
        color: #475569;
        font-size: 0.85rem;
        line-height: 1.45;
        margin: 0.35rem 0 0.75rem 0;
        flex-grow: 1;
      }

      .tool-example-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 7px;
        padding: 0.45rem 0.65rem;
        font-size: 0.78rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        color: #334155;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      /* Knowledge Base styles */
      .kb-row {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
      }

      .kb-file-info {
        display: flex;
        align-items: center;
        gap: 0.65rem;
      }

      .kb-filename {
        font-weight: 600;
        font-size: 0.9rem;
        color: #0f172a;
      }

      .retrieval-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.95rem 1.05rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
      }

      .retrieval-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.45rem;
      }

      .retrieval-source {
        font-weight: 600;
        font-size: 0.85rem;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 0.4rem;
      }

      .similarity-badge {
        font-size: 0.75rem;
        font-weight: 700;
        color: #047857;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
      }

      .retrieval-snippet {
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.45;
        background: #f8fafc;
        border-left: 3px solid #4f46e5;
        padding: 0.55rem 0.75rem;
        border-radius: 0 6px 6px 0;
      }

      /* Architecture Step Card */
      .arch-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
      }

      .arch-card h4 {
        margin: 0 0 0.35rem 0;
        font-size: 0.98rem;
        font-weight: 700;
        color: #0f172a;
      }

      .arch-card p {
        margin: 0;
        font-size: 0.86rem;
        color: #475569;
        line-height: 1.5;
      }

      /* Sidebar specs */
      .spec-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        padding: 0.35rem 0;
        border-bottom: 1px solid #f1f5f9;
      }

      .spec-name {
        color: #64748b;
      }

      .spec-val {
        font-weight: 600;
        color: #0f172a;
        font-family: ui-monospace, monospace;
        font-size: 0.76rem;
      }

      /* Chat input adjustments */
      div[data-testid="stChatInput"] {
        border-radius: 12px;
      }

      div[data-testid="stChatInput"] textarea {
        font-size: 0.92rem;
      }

      /* Footer */
      .app-footer {
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        font-size: 0.8rem;
        color: #94a3b8;
      }

      .app-footer a {
        color: #4f46e5;
        text-decoration: none;
        font-weight: 600;
      }

      .app-footer a:hover {
        text-decoration: underline;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "No `OPENAI_API_KEY` found. Add it to `.env` locally, or to the app's "
        "Secrets on Streamlit Cloud (see `.streamlit/secrets.toml.example`)."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


def history_text() -> str:
    msgs = st.session_state.messages[-MAX_TURNS_IN_CONTEXT * 2 :]
    if not msgs:
        return "(no previous messages)"
    role = {"user": "User", "assistant": "Assistant"}
    return "\n".join(f"{role[m['role']]}: {m['content']}" for m in msgs)


def answer(prompt: str, style: str) -> str:
    request = prompt
    if style == "Concise":
        request += "\n\n(Answer briefly — 1-3 sentences or short bullets.)"
    elif style == "Detailed":
        request += "\n\n(Give a thorough, well-structured explanation.)"
    try:
        result = PersonalAiAssistant().crew().kickoff(
            inputs={"user_request": request, "conversation_history": history_text()}
        )
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Something went wrong: {exc}"


kb = get_knowledge_base()
kb.ingest()
kb_stats = kb.stats()
facts = memory_facts()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.35rem;">
          <span style="font-size: 1.45rem;">🎓</span>
          <div>
            <div style="font-weight: 800; font-size: 1.05rem; color: #0f172a; line-height: 1.1;">Study Assistant</div>
            <div style="font-size: 0.72rem; color: #64748b;">Autonomous CrewAI Agent</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Personal academic study copilot with local RAG, memory, and real-time tools.")

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("###### ⚙️ ANSWER STYLE")

    style = st.radio(
        "Response Detail",
        ["Auto", "Concise", "Detailed"],
        index=0,
        label_visibility="collapsed",
        help="Adjusts output verbosity sent to the reasoning agent.",
    )

    st.markdown("<div style='margin-top: 1.1rem;'></div>", unsafe_allow_html=True)
    st.markdown("###### 📊 AGENT SPECIFICATIONS")

    model_name = os.environ.get("OPENAI_MODEL_NAME", "default")
    st.markdown(
        f"""
        <div class="spec-item"><span class="spec-name">LLM Engine</span><span class="spec-val">{model_name}</span></div>
        <div class="spec-item"><span class="spec-name">Framework</span><span class="spec-val">CrewAI 1.x</span></div>
        <div class="spec-item"><span class="spec-name">Embeddings</span><span class="spec-val">model2vec (32M)</span></div>
        <div class="spec-item"><span class="spec-name">Vector Store</span><span class="spec-val">NumPy Offline</span></div>
        <div class="spec-item"><span class="spec-name">Tool Count</span><span class="spec-val">8 Tools</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 1.4rem;'></div>", unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.toast("Conversation cleared")
        st.rerun()

# ------------------------------------------------------------------ header
st.markdown(
    """
    <div class="dashboard-header">
      <div class="brand-title-wrap">
        <h1>🎓 Student Study &amp; Productivity Assistant</h1>
        <p class="brand-subtitle">
          One CrewAI agent with 8 specialized tools · Local semantic RAG · Web intelligence · Long-term profile memory
        </p>
      </div>
      <div class="badge-pill">
        <div class="badge-dot"></div>
        <span>Agent Online</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ KPI row
turns_count = len(st.session_state.messages) // 2
docs_count = len(kb_stats["sources"])
chunks_count = kb_stats["chunks"]
facts_count = len(facts)

st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="border-top: 3px solid #4f46e5;">
        <div class="kpi-top">
          <span class="kpi-label">Knowledge Base</span>
          <div class="kpi-icon-badge">📚</div>
        </div>
        <div class="kpi-value">{chunks_count}</div>
        <div class="kpi-sub">{docs_count} document(s) indexed</div>
      </div>

      <div class="kpi-card" style="border-top: 3px solid #0284c7;">
        <div class="kpi-top">
          <span class="kpi-label">Active Tools</span>
          <div class="kpi-icon-badge">🛠️</div>
        </div>
        <div class="kpi-value">8</div>
        <div class="kpi-sub">RAG · Web · Memory · Math</div>
      </div>

      <div class="kpi-card" style="border-top: 3px solid #8b5cf6;">
        <div class="kpi-top">
          <span class="kpi-label">Profile Memory</span>
          <div class="kpi-icon-badge">🧠</div>
        </div>
        <div class="kpi-value">{facts_count}</div>
        <div class="kpi-sub">Facts remembered about you</div>
      </div>

      <div class="kpi-card" style="border-top: 3px solid #10b981;">
        <div class="kpi-top">
          <span class="kpi-label">Session Activity</span>
          <div class="kpi-icon-badge">💬</div>
        </div>
        <div class="kpi-value">{turns_count}</div>
        <div class="kpi-sub">Conversation turns completed</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ navigation tabs
tab_chat, tab_kb, tab_tools, tab_about = st.tabs(
    ["💬  Assistant", "📚  Knowledge Base", "🛠️  Tools", "ℹ️  How It Works"]
)

# ------------------------------------------------------------------ Assistant tab
with tab_chat:
    if not st.session_state.messages and st.session_state.pending is None:
        st.markdown(
            """
            <div class="empty-state-card">
              <div class="empty-avatar-circle">🎓</div>
              <h3 class="empty-state-title">What would you like to explore today?</h3>
              <p class="empty-state-desc">
                Ask questions about your course materials, research topics on the web,
                calculate equations, or schedule deadlines. Select a prompt below or type your own.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='font-size:0.75rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:#94a3b8;margin-bottom:0.55rem;'>SUGGESTED PROMPTS</p>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        for i, (icon, prompt_text) in enumerate(SUGGESTIONS):
            col = col1 if i % 2 == 0 else col2
            btn_label = f"{icon}  {prompt_text}  →"
            if col.button(btn_label, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending = prompt_text
                st.rerun()

    # Conversation history
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "🧑‍🎓" if role == "user" else "🎓"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

    typed = st.chat_input("Ask about your lecture notes, live web, or study tasks...")
    prompt = typed or st.session_state.pending
    st.session_state.pending = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Agent thinking — selecting tools and reasoning..."):
                reply = answer(prompt, style)
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# ------------------------------------------------------------------ Knowledge Base tab
with tab_kb:
    st.markdown(
        """
        <div style="margin-bottom: 1.1rem;">
          <h3 style="font-size: 1.25rem; font-weight: 700; margin: 0 0 0.25rem 0;">📚 Document Knowledge Base</h3>
          <p style="color: #64748b; font-size: 0.88rem; margin: 0;">
            Local semantic vector store powered by <code>model2vec</code> (32M parameters).
            Documents are parsed, chunked, and embedded entirely on your machine.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_docs, col_actions = st.columns([3, 2], gap="large")

    with col_docs:
        st.markdown("##### Indexed Documents")
        if kb_stats["sources"]:
            for source_name in kb_stats["sources"]:
                st.markdown(
                    f"""
                    <div class="kb-row">
                      <div class="kb-file-info">
                        <span style="font-size: 1.2rem;">📄</span>
                        <div>
                          <div class="kb-filename">{source_name}</div>
                          <div style="font-size: 0.75rem; color: #64748b;">Ready for RAG semantic search</div>
                        </div>
                      </div>
                      <span class="badge-pill" style="font-size: 0.72rem; color: #4f46e5; background: #eef2ff;">Indexed</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption(f"Total: {kb_stats['chunks']} text chunks indexed across {len(kb_stats['sources'])} file(s).")
        else:
            st.info("No documents indexed yet. Upload course materials to begin.")

    with col_actions:
        st.markdown("##### Manage & Upload")
        uploaded_files = st.file_uploader(
            "Upload Course Notes (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Files will be saved to knowledge/docs/ and indexed immediately.",
        )

        if uploaded_files:
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            for f in uploaded_files:
                (DOCS_DIR / f.name).write_bytes(f.getbuffer())
            info = get_knowledge_base().ingest()
            st.toast(f"Indexed {info['documents_indexed']} file(s) ({info['total_chunks']} chunks)")
            st.success(f"Indexed {info['total_chunks']} chunks successfully.")
            st.rerun()

        st.markdown("<div style='margin-top: 0.6rem;'></div>", unsafe_allow_html=True)
        if st.button("↻  Rebuild Vector Index", use_container_width=True):
            with st.spinner("Re-embedding documents locally..."):
                info = get_knowledge_base().ingest(force=True)
            st.toast(f"Index rebuilt: {info['total_chunks']} chunks")
            st.success(f"Index rebuilt: {info['total_chunks']} chunks.")

    st.divider()

    st.markdown("##### 🔍 Semantic Retrieval Inspector")
    st.caption("Verify how chunks match student queries with cosine similarity scores before asking the assistant.")

    search_query = st.text_input(
        "Test Query",
        placeholder="e.g., What is gradient descent? or When is the mid-term exam?",
        label_visibility="collapsed",
    )

    if search_query:
        hits = get_knowledge_base().search(search_query, k=4)
        if hits:
            st.markdown(f"<p style='font-size: 0.8rem; color: #64748b; margin: 0.4rem 0;'>Found {len(hits)} matching passages:</p>", unsafe_allow_html=True)
            for hit in hits:
                score_pct = int(hit["score"] * 100)
                st.markdown(
                    f"""
                    <div class="retrieval-card">
                      <div class="retrieval-header">
                        <div class="retrieval-source">
                          <span>📄</span> <span>{hit['source']}</span>
                        </div>
                        <span class="similarity-badge">{score_pct}% Match · Score {hit['score']}</span>
                      </div>
                      <div class="retrieval-snippet">{hit['text']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No passages found matching this query. Try a different query or index more documents.")

# ------------------------------------------------------------------ Tools tab
with tab_tools:
    st.markdown(
        """
        <div style="margin-bottom: 1.1rem;">
          <h3 style="font-size: 1.25rem; font-weight: 700; margin: 0 0 0.25rem 0;">🛠️ Agent Tool Integrations</h3>
          <p style="color: #64748b; font-size: 0.88rem; margin: 0;">
            The autonomous CrewAI agent evaluates your prompt and selects the right tool(s) dynamically in a ReAct loop.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i in range(0, len(TOOLS_CATALOG), 2):
        col1, col2 = st.columns(2, gap="medium")
        for col, tool in zip((col1, col2), TOOLS_CATALOG[i : i + 2]):
            with col:
                st.markdown(
                    f"""
                    <div class="tool-card">
                      <div>
                        <div class="tool-header">
                          <div class="tool-title-group">
                            <span class="tool-icon">{tool['icon']}</span>
                            <span class="tool-name">{tool['name']}</span>
                          </div>
                          <span class="tool-tag">{tool['category']}</span>
                        </div>
                        <p class="tool-desc">{tool['desc']}</p>
                      </div>
                      <div>
                        <div style="font-size: 0.7rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 0.25rem;">Example Query</div>
                        <div class="tool-example-box">{tool['example']}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='margin-bottom: 0.85rem;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ How It Works tab
with tab_about:
    st.markdown(
        """
        <div style="margin-bottom: 1.1rem;">
          <h3 style="font-size: 1.25rem; font-weight: 700; margin: 0 0 0.25rem 0;">ℹ️ Architecture &amp; System Design</h3>
          <p style="color: #64748b; font-size: 0.88rem; margin: 0;">
            An overview of how the single-agent ReAct loop, offline vector search, and long-term memory cooperate.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="arch-card">
          <h4>1. Single ReAct Agent Architecture</h4>
          <p>
            Powered by <strong>CrewAI</strong>. When a student prompt arrives, the agent analyzes the intent,
            determines which tools are required, executes tool actions, observes results, and synthesizes a final response.
            A built-in guardrail validates the response before returning it to the user.
          </p>
        </div>

        <div class="arch-card">
          <h4>2. Zero-Cost Local Vector RAG</h4>
          <p>
            Course materials placed in <code>knowledge/docs/</code> (PDF, TXT, MD) are broken into ~70-word chunks.
            They are embedded using <strong>model2vec</strong> (<code>potion-retrieval-32M</code>) — ultra-fast static embeddings
            that run locally without GPU or API calls. Embeddings are stored in compressed NumPy arrays and retrieved via cosine similarity.
          </p>
        </div>

        <div class="arch-card">
          <h4>3. Persistent Student Memory</h4>
          <p>
            Long-term preferences and facts are preserved across sessions in <code>knowledge/user_preference.txt</code>.
            When you mention exam dates, courses, or study habits, the agent records durable notes that persist even after the server restarts.
            Recent conversation turns are retained in session state for contextual multi-turn dialogue.
          </p>
        </div>

        <div class="arch-card">
          <h4>4. Multi-Modal Tool Ecosystem</h4>
          <p>
            Eight distinct toolkits extend agent capabilities beyond raw LLM knowledge: live DuckDuckGo web search,
            Wikipedia summary lookups, Open-Meteo weather forecasts, AST-based arithmetic calculation, calendar computation,
            and direct file readers.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("💡 **Verification & Tests:** The entire offline test suite runs with `uv run pytest -q` (13 passing tests without requiring LLM calls).")

# ------------------------------------------------------------------ footer
st.markdown(
    """
    <div class="app-footer">
      Student Study &amp; Productivity Assistant · Built with CrewAI &amp; Streamlit ·
      <a href="https://github.com/farheenfatima110/personal_ai_assistant" target="_blank">GitHub Repository</a>
    </div>
    """,
    unsafe_allow_html=True,
)
