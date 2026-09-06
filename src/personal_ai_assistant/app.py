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

from personal_ai_assistant.crew import PersonalAiAssistant  # noqa: E402
from personal_ai_assistant.rag.store import DOCS_DIR, get_knowledge_base  # noqa: E402

MAX_TURNS_IN_CONTEXT = 6

st.set_page_config(page_title="Study Assistant", page_icon="🎓", layout="centered")
st.title("🎓 Student Study & Productivity Assistant")
st.caption(
    "CrewAI agent with 8 tools · RAG over your notes · web research · "
    "persistent personal memory · calculator"
)

if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "No `OPENAI_API_KEY` found. Add it to `.env` locally, or to the app's "
        "Secrets on Streamlit Cloud (see `.streamlit/secrets.toml.example`)."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def history_text() -> str:
    msgs = st.session_state.messages[-MAX_TURNS_IN_CONTEXT * 2 :]
    if not msgs:
        return "(no previous messages)"
    role = {"user": "User", "assistant": "Assistant"}
    return "\n".join(f"{role[m['role']]}: {m['content']}" for m in msgs)


with st.sidebar:
    st.header("Study material (RAG)")
    st.write(f"Documents folder: `{DOCS_DIR}`")
    uploaded = st.file_uploader(
        "Add a PDF / TXT / MD to the knowledge base",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            (DOCS_DIR / f.name).write_bytes(f.getbuffer())
        info = get_knowledge_base().ingest()
        st.success(
            f"Indexed {info['total_chunks']} chunks from "
            f"{info['documents_indexed']} document(s)."
        )

    if st.button("Reindex now"):
        info = get_knowledge_base().ingest(force=True)
        st.info(f"Rebuilt index: {info['total_chunks']} chunks.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about your notes, the web, or your profile..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking (coordinating agents)..."):
            try:
                result = PersonalAiAssistant().crew().kickoff(
                    inputs={
                        "user_request": prompt,
                        "conversation_history": history_text(),
                    }
                )
                answer = str(result)
            except Exception as exc:  # noqa: BLE001
                answer = f"Something went wrong: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
