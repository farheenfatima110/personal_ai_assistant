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
REPO_URL = "https://github.com/farheenfatima110/personal_ai_assistant"

TOOLS = [
    ("📚", "Study Material Search", "Semantic RAG search across your indexed lecture notes and PDFs.",
     "What do my notes say about gradient descent?"),
    ("📄", "Read Document", "Open and analyse a specific PDF / CSV / TXT file by its path.",
     "Summarise C:/files/syllabus.pdf"),
    ("🌐", "Web Search", "Current information and recent events from the web (DuckDuckGo).",
     "Who won the 2024 Nobel Prize in Physics?"),
    ("📖", "Wikipedia", "Concise encyclopaedic definitions and background.",
     "Explain reinforcement learning"),
    ("🌦️", "Weather", "Live conditions for any city — Open-Meteo, no API key.",
     "Weather in Hyderabad right now"),
    ("📅", "Date & Time", "Today's date and 'in N days' deadline arithmetic.",
     "What date is 10 days from today?"),
    ("🧮", "Calculator", "Safe arithmetic — powers, roots, logs; no eval.",
     "15% of 3200 plus sqrt(169)"),
    ("🧠", "Personal Memory", "Reads your profile and saves durable new facts you share.",
     "Remember my exam is on 15 October"),
]

CAPABILITIES = [
    "🔍 Answer from your notes", "🌐 Research the web", "🧮 Do the maths",
    "📅 Track deadlines", "🧠 Remember your details", "📄 Read your documents",
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
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"], .stMarkdown, .stButton button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }
      .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1080px;
        min-height: 92vh; }
      #MainMenu, footer { visibility: hidden; }
      section[data-testid="stSidebar"] { border-right: 1px solid #e9edf2; }

      .hdr-title { font-size: 2.1rem; font-weight: 800; letter-spacing: -.03em;
        margin: 0; line-height: 1.15; }
      .hdr-sub   { color:#64748b; font-size:.95rem; margin:.4rem 0 0; }

      .kpi-label { font-size:.68rem; font-weight:700; letter-spacing:.06em;
        text-transform:uppercase; color:#64748b; margin:0; white-space:nowrap; }
      .kpi-value { font-size:1.85rem; font-weight:800; color:#0f172a; line-height:1.15;
        margin:.15rem 0 .1rem; letter-spacing:-.02em; }
      .kpi-sub   { font-size:.78rem; color:#64748b; margin:0; }

      .hero { text-align:center; padding:1.6rem 1rem .4rem; }
      .hero-badge { width:56px; height:56px; border-radius:16px; margin:0 auto .8rem;
        display:flex; align-items:center; justify-content:center; font-size:1.7rem;
        background:#eef2ff; border:1px solid #e0e7ff; }
      .hero h2 { font-size:1.25rem; font-weight:700; margin:0 0 .3rem; letter-spacing:-.01em; }
      .hero p  { color:#64748b; font-size:.92rem; margin:0 auto; max-width:520px; }

      .section-label { font-size:.72rem; font-weight:700; letter-spacing:.08em;
        text-transform:uppercase; color:#94a3b8; margin:1.4rem 0 .5rem; }

      .chip-row { display:flex; flex-wrap:wrap; gap:.4rem; }
      .chip { background:#f1f5f9; border:1px solid #e2e8f0; color:#334155;
        font-size:.82rem; font-weight:500; padding:.32rem .7rem; border-radius:999px; }
      .chip-accent { background:#eef2ff; border-color:#e0e7ff; color:#4338ca; }

      .spec-row { display:flex; justify-content:space-between; font-size:.82rem;
        padding:.32rem 0; border-bottom:1px solid #eef0f3; }
      .spec-row:last-child { border-bottom:none; }
      .spec-row .k { color:#64748b; }
      .spec-row .v { color:#0f172a; font-weight:600; }

      .stButton button { border-radius:10px; border:1px solid #e2e8f0; background:#fff;
        text-align:left; font-size:.88rem; font-weight:500; padding:.6rem .85rem;
        width:100%; color:#1e293b; transition:all .15s ease; }
      .stButton button:hover { border-color:#a5b4fc; color:#4338ca; background:#fafaff; }

      .tool-name { font-weight:700; font-size:.95rem; margin:0; }
      .tool-desc { color:#475569; font-size:.84rem; margin:.3rem 0 .5rem; }
      .tool-ex   { font-family:ui-monospace,Menlo,monospace; font-size:.76rem;
        background:#f8fafc; border:1px solid #eef0f3; border-radius:6px;
        padding:.2rem .45rem; color:#475569; }

      div[data-testid="stChatInput"] textarea { font-size:.95rem; }
      .stTabs [data-baseweb="tab"] { font-weight:600; }
      .footer { text-align:center; color:#94a3b8; font-size:.8rem; margin-top:2.5rem;
        padding-top:1rem; border-top:1px solid #eef0f3; }
      .footer a { color:#6366f1; text-decoration:none; }
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
model_name = os.environ.get("OPENAI_MODEL_NAME", "default")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🎓 Study Assistant")
    st.caption("A CrewAI agent that answers from your notes, the web, and memory.")
    st.divider()

    style = st.radio("Answer style", ["Auto", "Concise", "Detailed"])

    st.markdown('<p class="section-label">System</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="spec-row"><span class="k">LLM</span>'
        f'<span class="v">{model_name}</span></div>'
        f'<div class="spec-row"><span class="k">Framework</span>'
        f'<span class="v">CrewAI</span></div>'
        f'<div class="spec-row"><span class="k">Embeddings</span>'
        f'<span class="v">model2vec · local</span></div>'
        f'<div class="spec-row"><span class="k">Vector store</span>'
        f'<span class="v">NumPy · offline</span></div>'
        f'<div class="spec-row"><span class="k">Tools</span>'
        f'<span class="v">{len(TOOLS)}</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------ header
st.markdown(
    '<p class="hdr-title">🎓 Student Study &amp; Productivity Assistant</p>'
    '<p class="hdr-sub">One CrewAI agent · 8 tools · local RAG over your notes · '
    'web research · calculator · long-term memory</p>',
    unsafe_allow_html=True,
)
st.write("")


def kpi(col, label: str, value: str, sub: str) -> None:
    with col.container(border=True):
        st.markdown(
            f'<p class="kpi-label">{label}</p><p class="kpi-value">{value}</p>'
            f'<p class="kpi-sub">{sub}</p>',
            unsafe_allow_html=True,
        )


r1a, r1b = st.columns(2, gap="small")
kpi(r1a, "Notes indexed", str(kb_stats["chunks"]),
    f'{len(kb_stats["sources"])} file · {kb_stats["chunks"]} chunks embedded locally')
kpi(r1b, "Tools ready", str(len(TOOLS)),
    "RAG · web search · Wikipedia · weather · maths · memory")
r2a, r2b = st.columns(2, gap="small")
kpi(r2a, "Memory", str(len(facts)), "facts saved about you across sessions")
kpi(r2b, "This session", str(len(st.session_state.messages) // 2),
    "questions answered so far")

st.write("")
tab_chat, tab_kb, tab_tools, tab_about = st.tabs(
    ["💬  Assistant", "📚  Knowledge base", "🛠️  Tools", "ℹ️  How it works"]
)

# ------------------------------------------------------------------ Assistant
with tab_chat:
    if not st.session_state.messages and st.session_state.pending is None:
        st.markdown(
            '<div class="hero"><div class="hero-badge">🎓</div>'
            "<h2>How can I help with your studies today?</h2>"
            "<p>Ask about your course material, research a topic, run a calculation, "
            "or note a deadline — then keep the conversation going.</p></div>",
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-label">Try one of these</p>', unsafe_allow_html=True)
        rows = [st.columns(2), st.columns(2)]
        for i, ex in enumerate(EXAMPLES):
            if rows[i // 2][i % 2].button(ex, key=f"ex_{i}"):
                st.session_state.pending = ex
                st.rerun()

        st.markdown('<p class="section-label">What it can do</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="chip-row">'
            + "".join(f'<span class="chip">{c}</span>' for c in CAPABILITIES)
            + "</div>",
            unsafe_allow_html=True,
        )

        if facts:
            st.markdown('<p class="section-label">What it remembers about you</p>',
                        unsafe_allow_html=True)
            shown = [f.split("] ")[-1].rstrip(".") for f in facts][:6]
            st.markdown(
                '<div class="chip-row">'
                + "".join(f'<span class="chip chip-accent">{s}</span>' for s in shown)
                + "</div>",
                unsafe_allow_html=True,
            )

        if kb_stats["sources"]:
            st.markdown('<p class="section-label">Study material indexed</p>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="chip-row">'
                + "".join(f'<span class="chip">📄 {s}</span>' for s in kb_stats["sources"])
                + "</div>",
                unsafe_allow_html=True,
            )

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

# ------------------------------------------------------------------ Knowledge base
with tab_kb:
    st.markdown("#### Your study material")
    st.caption(
        "Files in `knowledge/docs/` are chunked and embedded locally (model2vec). "
        "Retrieval is cosine similarity — no API key, fully offline."
    )
    if kb_stats["sources"]:
        for s in kb_stats["sources"]:
            with st.container(border=True):
                st.markdown(f"**📄 {s}**")
        st.caption(f'{kb_stats["chunks"]} chunks indexed in total.')
    else:
        st.info("No documents indexed yet — upload one below.")

    up = st.file_uploader("Add PDF / TXT / MD", type=["pdf", "txt", "md"],
                          accept_multiple_files=True)
    if up:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for f in up:
            (DOCS_DIR / f.name).write_bytes(f.getbuffer())
        info = get_knowledge_base().ingest()
        st.success(f"Indexed {info['total_chunks']} chunks from "
                   f"{info['documents_indexed']} document(s).")
        st.rerun()

    if st.button("↻  Rebuild index"):
        info = get_knowledge_base().ingest(force=True)
        st.toast(f"Rebuilt: {info['total_chunks']} chunks")

    st.divider()
    st.markdown("#### Test retrieval")
    q = st.text_input("Query", placeholder="e.g. how does gradient descent work",
                      label_visibility="collapsed")
    if q:
        for h in get_knowledge_base().search(q, k=3):
            with st.container(border=True):
                st.markdown(f"**{h['source']}**  ·  similarity `{h['score']}`")
                st.caption(h["text"][:400] + ("…" if len(h["text"]) > 400 else ""))

# ------------------------------------------------------------------ Tools
with tab_tools:
    st.markdown("#### The agent picks from these eight tools")
    st.caption("It reads your request, calls the tool(s) it needs, then answers.")
    st.write("")
    for i in range(0, len(TOOLS), 2):
        cols = st.columns(2)
        for col, (icon, name, desc, ex) in zip(cols, TOOLS[i:i + 2]):
            with col.container(border=True):
                st.markdown(f'<p class="tool-name">{icon}  {name}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="tool-desc">{desc}</p>', unsafe_allow_html=True)
                st.markdown(f'<span class="tool-ex">{ex}</span>', unsafe_allow_html=True)

# ------------------------------------------------------------------ How it works
with tab_about:
    st.markdown(
        """
        #### How it works

        **One agent, sequential process.** A single CrewAI `Agent` receives your
        request plus recent conversation, then runs a ReAct loop: think → pick a
        tool → read the result → repeat → final answer. A guardrail rejects empty
        or refusal answers and retries.

        **Retrieval-Augmented Generation.** Documents in `knowledge/docs/` are split
        into ~70-word chunks and embedded with **model2vec** (`potion-retrieval-32M`)
        — static embeddings that run locally with no API key. The index is cached in
        `knowledge/.rag_index.npz` and rebuilt only when a file changes. Retrieval is
        cosine similarity over the chunk vectors.

        **Persistent memory.** `Personal Memory` reads `knowledge/user_preference.txt`;
        `Save Memory` appends durable facts you share, so they are remembered in later
        sessions. Recent chat turns are passed back as context for multi-turn dialogue.

        **Tools / APIs.** DuckDuckGo web search, Wikipedia REST API, Open-Meteo weather
        (no key), a safe AST-based calculator, and date maths.
        """
    )
    st.caption(
        "Code: `src/personal_ai_assistant/` — `crew.py`, `config/*.yaml`, `tools/`, "
        "`rag/store.py`.  Tests: `uv run pytest -q` (13 passing, no LLM calls)."
    )

st.markdown(
    f'<div class="footer">Student Study &amp; Productivity Assistant · '
    f'built with CrewAI &amp; Streamlit · <a href="{REPO_URL}">GitHub</a></div>',
    unsafe_allow_html=True,
)
