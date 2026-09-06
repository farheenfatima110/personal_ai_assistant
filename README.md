# 🎓 Student Study & Productivity Assistant

A **personal AI assistant that understands natural-language requests and executes
them using tools and APIs**, built as a coordinated **multi-agent CrewAI** system.

The assistant is aimed at an AI & Data Science student: it answers questions from
the student's own lecture notes (RAG), researches the open web, tracks personal
deadlines and preferences in persistent memory, reads documents, and does maths -
all through one chat interface.

---

## What it can do

| Capability | How | Tools |
|---|---|---|
| Answer questions from **your own study notes / PDFs** | Local RAG (semantic search over `knowledge/docs/`) | `Study Material Search` |
| **Research** current info & general knowledge | DuckDuckGo web search + Wikipedia | `Web Search`, `Wikipedia` |
| **Remember** your deadlines, goals, preferences across sessions | Read + append to a plain-text profile | `Personal Memory`, `Save Memory` |
| **Read / summarise** a specific PDF, CSV, TXT or MD file | Path-based document reader | `Read Document` |
| **Weather** and **date/deadline** math | Open-Meteo (no key) + date tool | `Weather`, `Date and Time` |
| **Arithmetic** | Safe AST evaluator (no `eval`) | `Calculator` |

---

## Architecture

```
   user request + chat history
              │
              ▼
   ┌───────────────────────────────────────────────┐
   │  assistant  (CrewAI Agent, ReAct loop)         │
   │  reads the request → picks tool(s) → answers   │
   │  sequential process · guardrail on the output  │
   └───────────────────────┬───────────────────────┘
                           │  8 tools
   ┌───────────────────────┼─────────────────────────────────┐
   │            │          │           │          │          │
   ▼            ▼          ▼           ▼          ▼          ▼
Study RAG   Web Search  Wikipedia   Weather   Date/Time  Calculator
Read Doc    Personal Memory (read)  Save Memory (write)
```

- **Config-driven:** agent + task live in
  `src/personal_ai_assistant/config/{agents,tasks}.yaml`
- **Guardrail:** `assist_task` rejects empty / pure-refusal outputs and retries
- **Logging:** every run is appended to `logs.txt`

> The agent chooses tools itself from the request (a single reliable ReAct loop).
> A multi-agent hierarchical variant was prototyped but a small manager model
> delegated inconsistently; the single-agent design is faster and more accurate
> for this workload.

### RAG layer (`src/personal_ai_assistant/rag/store.py`)
- Documents: `knowledge/docs/*.{pdf,txt,md}`
- Chunking: ~70 words with 20-word overlap
- Embeddings: **`minishlab/potion-retrieval-32M`** via `model2vec` -
  static embeddings, **no API key, runs fully offline**
- Store: cached vectors in `knowledge/.rag_index.npz` (+ `.json` metadata);
  re-embeds only when a file's hash changes
- Retrieval: cosine similarity, top-k passages with source file + score

### Memory
- **Long-term / cross-session:** `Save Memory` appends facts to
  `knowledge/user_preference.txt`; `Personal Memory` reads it back on every run.
- **Short-term / multi-turn:** the last few chat turns are passed into the task
  as `{conversation_history}` (handled by the CLI loop and the Streamlit app).

---

## Setup

Requires Python `>=3.10,<3.14`.

```bash
pip install uv
uv sync
```

Create `.env` in the project root:

```
OPENAI_MODEL_NAME=gpt-oss-120b
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-...        # your OpenRouter key
```

Any LiteLLM-supported provider works - e.g. `OPENAI_MODEL_NAME=gpt-4o-mini`
with a real OpenAI key. Embeddings are local, so no embedding key is needed.

---

## Running

### CLI (multi-turn chat)
```bash
uv run run_crew
# optional: attach a document for the session
uv run run_crew --file "C:/path/to/syllabus.pdf"
```

### Web UI (Streamlit)
```bash
uv run streamlit run src/personal_ai_assistant/app.py
```
Upload PDFs/notes from the sidebar to grow the knowledge base live.

### One-shot (for automation / triggers)
```bash
uv run run_with_trigger '{"user_request": "What is my exam date?"}'
```

### Tests
```bash
uv run pytest -q            # tool + RAG unit tests (no LLM calls)
uv run test 2 gpt-4o-mini   # CrewAI end-to-end eval (needs LLM)
```

---

## Deploy (Streamlit Community Cloud - free)

1. Push this repo to GitHub.
2. Go to <https://share.streamlit.io> → sign in with GitHub → **Create app** →
   **Deploy a public app from GitHub**.
3. Settings:
   - Repository: `<your-user>/personal_ai_assistant`
   - Branch: `main`
   - Main file path: `src/personal_ai_assistant/app.py`
   - Python version (Advanced): `3.11`
4. **Secrets** (Advanced settings → Secrets), paste:
   ```toml
   OPENAI_MODEL_NAME = "gpt-oss-120b"
   OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
   OPENAI_API_KEY = "sk-or-v1-..."
   ```
5. **Deploy**. First start downloads the ~130 MB embedding model (once).

`requirements.txt` is what Streamlit Cloud installs. Note: a public app uses your
API key for every visitor - keep the app unlisted or rotate the key after grading.

---

## Example interactions

```
You: My machine learning exam is on 15 October. Please remember that.
Assistant: Saved to your profile: ML exam on 15 October.

You: What topics should I focus on for it?
Assistant: From ml_lecture_notes.md - the mid-term covers the bias-variance
tradeoff, gradient-descent maths, precision vs recall, and L1 vs L2
regularisation. (source: ml_lecture_notes.md)

You: What's the weather in Hyderabad right now?
Assistant: Hyderabad, India - partly cloudy, 29°C (feels like 31°C),
humidity 62%, wind 12 km/h.  (source: Open-Meteo)
```

---

## Project layout

```
personal_ai_assistant/
├── knowledge/
│   ├── user_preference.txt        # persistent personal memory
│   ├── docs/                       # your study material -> RAG
│   └── .rag_index.npz / .json      # cached embeddings (auto-generated)
├── src/personal_ai_assistant/
│   ├── config/agents.yaml
│   ├── config/tasks.yaml
│   ├── rag/store.py                # local RAG knowledge base
│   ├── tools/                      # 8 tools (see table above)
│   ├── crew.py                     # 4 agents, hierarchical, guardrail, logging
│   ├── main.py                     # CLI entry points
│   └── app.py                      # Streamlit chat UI
├── tests/test_tools.py
└── logs.txt                        # per-run execution log (auto-generated)
```
