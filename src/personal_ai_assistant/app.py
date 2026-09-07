"""Streamlit chat UI for the Student Study & Productivity Assistant.

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

from personal_ai_assistant.rag.store import DOCS_DIR, get_knowledge_base  # noqa: E402
from personal_ai_assistant.crew import PersonalAiAssistant  # noqa: E402

MAX_TURNS_IN_CONTEXT = 6

TOOLS = [
    ("📚", "Study Material Search", "semantic RAG over your notes"),
    ("📄", "Read Document", "open a PDF / CSV / TXT by path"),
    ("🌐", "Web Search", "current information from the web"),
    ("📖", "Wikipedia", "encyclopaedic facts & definitions"),
    ("🌦️", "Weather", "live conditions for any city"),
    ("📅", "Date & Time", "deadlines and 'in N days' math"),
    ("🧮", "Calculator", "safe arithmetic"),
    ("🧠", "Personal Memory", "read + save what you tell it"),
]

EXAMPLES = [
    "What topics are on my mid-term exam?",
    "Explain the bias-variance tradeoff from my notes",
    "What is 15% of 3200 plus the square root of 169?",
    "Remember that my project is due next Friday",
]

st.set_page_config(
    page_title="Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 850px; }
      #MainMenu, footer { visibility: hidden; }

      .hero { text-align: center; margin: 0.5rem 0 1.6rem; }
      .hero h1 { font-size: 2rem; font-weight: 750; margin: 0 0 .35rem; line-height: 1.2; }
      .hero p  { color: var(--text-color-secondary, #9aa0aa); margin: 0; font-size: .95rem; }

      .chip {
        display:inline-block; padding:.28rem .7rem; margin:.2rem;
        border:1px solid rgba(140,150,180,.28); border-radius:999px;
        font-size:.82rem; color:#c9cdd6;
      }
      .kb-card {
        border:1px solid rgba(140,150,180,.22); border-radius:12px;
        padding:.7rem .85rem; margin-bottom:.6rem; font-size:.85rem;
      }
      .tool-row { font-size:.85rem; line-height:1.5; margin:.15rem 0; }
      .tool-row b { color:#e6e8ec; }
      .tool-row span { color:#9aa0aa; }
      div[data-testid="stChatInput"] textarea { font-size:.95rem; }
      .stButton button {
        border-radius:10px; border:1px solid rgba(140,150,180,.3);
        text-align:left; font-size:.86rem; padding:.5rem .8rem; width:100%;
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


def answer(prompt: str) -> str:
    try:
        result = PersonalAiAssistant().crew().kickoff(
            inputs={
                "user_request": prompt,
                "conversation_history": history_text(),
            }
        )
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Something went wrong: {exc}"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🎓 Study Assistant")
    st.caption(f"Model: `{os.environ.get('OPENAI_MODEL_NAME', 'default')}`")

    st.markdown("#### 📚 Knowledge base")
    kb = get_knowledge_base()
    kb.ingest()
    stats = kb.stats()
    if stats["sources"]:
        st.markdown(
            f'<div class="kb-card">{stats["chunks"]} chunks · '
            f'{len(stats["sources"])} document(s)<br>'
            + "<br>".join(f"• {s}" for s in stats["sources"])
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No documents yet — upload some below.")

    uploaded = st.file_uploader(
        "Add PDF / TXT / MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            (DOCS_DIR / f.name).write_bytes(f.getbuffer())
        info = get_knowledge_base().ingest()
        st.success(f"Indexed {info['total_chunks']} chunks.")
        st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("↻ Reindex"):
        info = get_knowledge_base().ingest(force=True)
        st.toast(f"Reindexed: {info['total_chunks']} chunks")
    if c2.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("#### 🛠️ Tools")
    for icon, name, desc in TOOLS:
        st.markdown(
            f'<div class="tool-row">{icon} <b>{name}</b> — <span>{desc}</span></div>',
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------------ header
st.markdown(
    """
    <div class="hero">
      <h1>🎓 Student Study &amp; Productivity Assistant</h1>
      <p>One CrewAI agent · 8 tools · RAG over your notes · web research ·
         calculator · memory that remembers you</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ empty state
if not st.session_state.messages and st.session_state.pending is None:
    st.markdown("<p style='text-align:center;color:#9aa0aa;'>Try one of these 👇</p>",
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}"):
            st.session_state.pending = ex
            st.rerun()

# ------------------------------------------------------------------ transcript
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🎓"):
        st.markdown(m["content"])

# ------------------------------------------------------------------ input
typed = st.chat_input("Ask about your notes, the web, or your profile...")
prompt = typed or st.session_state.pending
st.session_state.pending = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking — picking the right tools…"):
            reply = answer(prompt)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
