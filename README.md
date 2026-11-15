# Agent Coliseum

[![CI](https://github.com/aifriend/agent-coliseum/actions/workflows/ci.yml/badge.svg)](https://github.com/aifriend/agent-coliseum/actions/workflows/ci.yml)

Agent Coliseum holds three question-answering agents, ported from Colab notebooks, that can be compared side by side. They are a Condor RAG agent (retrieval plus memory), a LangChain-style agent (memory plus reasoning) and a naive baseline (no memory, no retrieval). A small local web app asks all three the same question about the same knowledge so you can watch the different reasoning styles.

## Setup

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Edit `.env` and fill in your values. `.env.example` is the template for it.

### Environment variables

The three agent modules read these variables:

| Variable | Purpose |
|---|---|
| `AZURE_API_KEY` | API key for the Azure OpenAI resource the agents call. |
| `AZURE_BASE_URL` | Base URL of that resource, for example `https://YOUR-RESOURCE.openai.azure.com/openai/v1/`. |
| `NGROK_TOKEN` | ngrok auth token, used only when an agent is exposed through a tunnel to join the live arena. |

## Run the showcase app

```bash
uv run python showcase_app.py
```

The app serves on port 8000 (`http://localhost:8000`). It calls the agents directly in-process, so it opens no ngrok tunnel and registers nothing with the live arena.

## Tests

```bash
uv run pytest -q
```

## Layout

- `agent_base.py` holds the base classes that agents import.
- `agent_server.py` serves an agent over HTTP and registers it with the arena.
- `condor_rag_agent.py`, `langchain_agent.py` and `naive_baseline.py` are the three agents.
- `latam_facts.jsonl` is the fact file the Condor agent retrieves from.
- `showcase_app.py`, `templates/` and `static/` make up the web UI.
- The `0*_*.ipynb` notebooks are the original Colab versions.
