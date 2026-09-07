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

TOOLS = [
    ("📚", "Study Material Search", "Semantic RAG search over your indexed notes and PDFs.",
     "What does chapter 3 say about gradient descent?"),
    ("📄", "Read Document", "Open a specific PDF / CSV / TXT file by path and analyse it.",
     "Summarise C:/files/syllabus.pdf"),
    ("🌐", "Web Search", "Current information and recent events from the web (DuckDuckGo).",
     "Who won the 2024 Nobel Prize in Physics?"),
    ("📖", "Wikipedia", "Concise encyclopaedic definitions and background.",
     "Explain reinforcement learning"),
    ("🌦️", "Weather", "Live conditions for any city (Open-Meteo, no key).",
     "Weather in Hyderabad right now"),
    ("📅", "Date & Time", "Current date and 'in N days' deadline math.",
     "What date is 10 days from today?"),
    ("🧮", "Calculator", "Safe arithmetic — no eval, functions like sqrt / log.",
     "15% of 3200 plus sqrt(169)"),
    ("🧠", "Personal Memory", "Reads your profile and saves durable new facts.",
     "Remember that my exam is on 15 October"),
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
      .block-container { padding-top: 2rem; max-width: 1050px; }
      #MainMenu, footer { visibility: hidden; }

      .app-title { font-size: 1.7rem; font-weight: 780; margin: 0; letter-spacing:-.01em; }
      .app-stack { color:#6b7280; font-size:.82rem; margin:.15rem 0 .1rem; }
      .app-desc  { color:#4b5563; font-size:.92rem; margin:.1rem 0 0; }

      .metric-card {
        border:1px solid #e6e8ec; border-radius:14px; padding:.85rem 1rem;
        background:#fff; height:100%;
      }
      .metric-label { font-size:.68rem; font-weight:700; letter-spacing:.06em;
        text-transform:uppercase; color:#8a92a0; }
      .metric-value { font-size:1.5rem; font-weight:750; margin:.15rem 0 .1rem;
        color:#1f2430; line-height:1.1; }
      .metric-sub { font-size:.8rem; color:#6b7280; }

      .tool-card {
        border:1px solid #e6e8ec; border-radius:14px; padding:1rem 1.1rem;
        background:#fff; height:100%;
      }
      .tool-card h4 { margin:.1rem 0 .3rem; font-size:1rem; }
      .tool-card p  { margin:0 0 .5rem; font-size:.86rem; color:#4b5563; }
      .tool-card code { font-size:.78rem; background:#f1f3f7; padding:.12rem .4rem;
        border-radius:6px; color:#334155; }

      .fact-row { font-size:.88rem; padding:.35rem 0; border-bottom:1px solid #eef0f3; }
      .stButton button { border-radius:10px; text-align:left; font-size:.88rem; width:100%; }
      div[data-testid="stChatInput"] textarea { font-size:.95rem; }
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

# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🎓 Study Assistant")
    st.caption("A CrewAI agent that answers from your notes, the web, and memory.")
    st.divider()
    style = st.radio("Answer style", ["Auto", "Concise", "Detailed"], horizontal=False)
    st.caption(f"Model  ·  `{os.environ.get('OPENAI_MODEL_NAME', 'default')}`")
    st.caption(f"Embeddings  ·  `local (model2vec)`")
    st.divider()
    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------- header
st.markdown(
    """
    <div>
      <p class="app-title">🎓 Student Study &amp; Productivity Assistant</p>
      <p class="app-stack">CrewAI · Python · local RAG (model2vec) · DuckDuckGo · Open-Meteo · Streamlit</p>
      <p class="app-desc">Ask questions about your own lecture notes, research the web,
         do maths, and keep track of your deadlines — one agent, eight tools.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


def metric(label: str, value: str, sub: str) -> str:
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div></div>'
    )


m1, m2, m3, m4 = st.columns(4)
m1.markdown(
    metric("Knowledge base", str(kb_stats["chunks"]),
           f'{len(kb_stats["sources"])} document(s) indexed'),
    unsafe_allow_html=True,
)
m2.markdown(metric("Tools", "8", "RAG · web · memory · calc · more"), unsafe_allow_html=True)
m3.markdown(metric("Memory", str(len(facts)), "facts remembered about you"),
           unsafe_allow_html=True)
m4.markdown(
    metric("Conversation", str(len(st.session_state.messages) // 2), "turns this session"),
    unsafe_allow_html=True,
)
st.write("")

tab_chat, tab_kb, tab_tools, tab_about = st.tabs(
    ["💬  Assistant", "📚  Knowledge base", "🛠️  Tools", "ℹ️  How it works"]
)

# --------------------------------------------------------------------- chat tab
with tab_chat:
    if not st.session_state.messages and st.session_state.pending is None:
        st.caption("Try one of these, or type your own below:")
        c = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if c[i % 2].button(ex, key=f"ex_{i}"):
                st.session_state.pending = ex
                st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
            st.markdown(msg["content"])

    typed = st.chat_input("Ask about your notes, the web, or your profile…")
    prompt = typed or st.session_state.pending
    st.session_state.pending = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Thinking — picking the right tools…"):
                reply = answer(prompt, style)
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# --------------------------------------------------------------------- KB tab
with tab_kb:
    st.markdown("#### Your study material")
    st.caption(
        "Files in `knowledge/docs/` are chunked and embedded locally. "
        "Retrieval uses cosine similarity — no API key, works offline."
    )
    if kb_stats["sources"]:
        for s in kb_stats["sources"]:
            st.markdown(f'<div class="fact-row">📄 {s}</div>', unsafe_allow_html=True)
        st.caption(f'{kb_stats["chunks"]} chunks indexed in total.')
    else:
        st.info("No documents indexed yet. Upload one below.")

    up = st.file_uploader("Add PDF / TXT / MD", type=["pdf", "txt", "md"],
                          accept_multiple_files=True)
    if up:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for f in up:
            (DOCS_DIR / f.name).write_bytes(f.getbuffer())
        info = get_knowledge_base().ingest()
        st.success(f"Indexed — {info['total_chunks']} chunks from "
                   f"{info['documents_indexed']} document(s).")
        st.rerun()

    if st.button("↻  Rebuild index"):
        info = get_knowledge_base().ingest(force=True)
        st.toast(f"Rebuilt: {info['total_chunks']} chunks")

    st.divider()
    q = st.text_input("Preview retrieval", placeholder="Type a query to see matching passages…")
    if q:
        for h in get_knowledge_base().search(q, k=3):
            st.markdown(f"**{h['source']}** · similarity {h['score']}")
            st.caption(h["text"][:400] + ("…" if len(h["text"]) > 400 else ""))

# --------------------------------------------------------------------- tools tab
with tab_tools:
    st.markdown("#### The agent chooses from these eight tools")
    st.caption("It reads your request, calls the tool(s) it needs, then answers.")
    st.write("")
    for row in range(0, len(TOOLS), 2):
        cols = st.columns(2)
        for col, (icon, name, desc, ex) in zip(cols, TOOLS[row:row + 2]):
            col.markdown(
                f'<div class="tool-card"><h4>{icon}  {name}</h4>'
                f'<p>{desc}</p><code>{ex}</code></div>',
                unsafe_allow_html=True,
            )
        st.write("")

# --------------------------------------------------------------------- about tab
with tab_about:
    st.markdown(
        """
        #### How it works

        **1. One agent, sequential process.** A single CrewAI `Agent` receives your
        request plus the recent conversation. It runs a ReAct loop: think → pick a
        tool → observe the result → repeat → final answer. A guardrail rejects empty
        or refusal answers and retries.

        **2. Retrieval-Augmented Generation (RAG).** Documents in `knowledge/docs/`
        are split into ~70-word chunks and embedded with **model2vec**
        (`potion-retrieval-32M`) — static embeddings that run locally with no API
        key. The index is cached in `knowledge/.rag_index.npz` and only rebuilt when
        a file changes. Retrieval is cosine similarity over the chunk vectors.

        **3. Persistent memory.** `Personal Memory` reads `knowledge/user_preference.txt`;
        `Save Memory` appends durable facts you share, so the assistant remembers
        them in later sessions. Recent chat turns are passed back as context for
        multi-turn conversations.

        **4. Tools / APIs.** Web search (DuckDuckGo), Wikipedia REST API, weather
        (Open-Meteo — no key), a safe AST-based calculator, and date maths.

        &nbsp;

        *Code: `src/personal_ai_assistant/` — `crew.py`, `config/*.yaml`, `tools/`, `rag/store.py`.
        Tests: `uv run pytest -q` (13 passing, no LLM calls).*
        """
    )
