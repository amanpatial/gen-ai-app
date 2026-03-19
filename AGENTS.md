# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a Python-based Gen AI learning/experimentation repository with several standalone components — not a production service. There are no shared build systems, Docker containers, or databases.

### Key components

| Component | Path | Run command | Needs API key |
|---|---|---|---|
| Google ADK multi-tool agent | `multi_tool_agent/` | `adk web --port 8080` (from repo root) | `GOOGLE_API_KEY` |
| LangChain PDF chatbot (Streamlit) | `langchain/chatbot/app.py` | `python3 -m streamlit run langchain/chatbot/app.py` | `OPENAI_API_KEY` |
| Product review summarizer | `agent/product_review_summarizer/main.py` | `python agent/product_review_summarizer/main.py` | `OPENAI_API_KEY` |
| LangChain OpenAI hello | `langchain/openai/hello_langchain_openai.py` | `python langchain/openai/hello_langchain_openai.py` | `OPENAI_API_KEY` |
| Pinecone chatbot | `pinecone/chatbot.py` | `python pinecone/chatbot.py` | `OPENAI_API_KEY` + `PINE_CONE_API_KEY` |
| Evals dataset builder | `evals/openai-evals/build_dataset.py` | `python evals/openai-evals/build_dataset.py` | None |

### Virtual environment

The venv lives at `.venv/`. Always activate before running anything:
```
source .venv/bin/activate
```

### Gotchas

- The codebase uses **langchain 0.3.x** (not 1.x). Imports like `from langchain.text_splitter import ...` and `from langchain.chains import ...` require this version range. Do not upgrade to langchain >=1.0.
- The root `requirements.txt` and `requirements-dev.txt` are entirely commented out — they are scaffolds. Actual dependencies are installed via pip directly (see the update script).
- Sub-project requirements files (`agent/product_review_summarizer/requirements.txt`, `evals/openai-evals/requirements.txt`) list real dependencies.
- All test files under `tests/` are empty stubs (0 tests collected is expected).
- The `fetch_reviews` tool in the product review summarizer hardcodes `page=3`, which yields 0 results when called standalone. It is designed to be called by the LangGraph ReAct agent across multiple pages.
- The Google ADK `adk web` command must be run from the repo root so it can discover the `multi_tool_agent/` package.

### Lint / test

- `flake8 --max-line-length=120 multi_tool_agent/ langchain/ agent/ pinecone/ evals/`
- `black --check multi_tool_agent/ langchain/ agent/ pinecone/ evals/`
- `python -m pytest tests/ -v` (currently 0 tests)
