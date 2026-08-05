# LuminaSQL-Agent

Self-hosted NL→SQL/PartiQL agent for **PostgreSQL** and **AWS DynamoDB**, with a **self-correcting execution loop**.

Ask a question in natural language → the agent prunes relevant schema context → generates SQL or PartiQL → executes it → on failure, feeds the raw database error back to the model and retries (default max 3).

## Features

- **Dual backends:** PostgreSQL (SQL) and DynamoDB (PartiQL via `execute_statement`)
- **Schema-aware generation:** live metadata extraction (tables, columns, PK/FK, types, comments) with Top-K pruning
- **Self-correcting loop:** capture execution errors → debug prompt → rewrite → retry
- **Streaming:** FastAPI SSE events for schema, LLM tokens, queries, errors, and results
- **LLM providers:** OpenAI or Anthropic (swappable clients)
- **Safety defaults:** mutating/DDL statements blocked unless explicitly enabled; result row capping

## Architecture

```
User (Streamlit)
  → POST /api/v1/query/stream (SSE)
    → SchemaManager (extract + prune)
    → LLM (generate SQL/PartiQL)
    → DatabaseExecutor (Postgres / DynamoDB)
         success → result rows
         error   → debug prompt + retry (≤ MAX_RETRY_ITERATIONS)
```

| Module | Path | Role |
|--------|------|------|
| Agent | `src/agent.py` | Orchestration, prompts, self-debug loop, streaming |
| Schema | `src/schema_manager.py` | Live schema extract + token-cosine Top-K pruning |
| Executor | `src/db_executor.py` | Query execution and safety checks |
| API | `api/main.py` | FastAPI health + sync/stream endpoints |
| UI | `app.py` | Streamlit chat client |

## Quick start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Fill in at least:

- One LLM key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- PostgreSQL and/or DynamoDB connection settings

### 3. Run API

```bash
python run_api.py
```

API listens on `http://localhost:8000` by default.

### 4. Run UI (optional)

```bash
streamlit run app.py
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Postgres + DynamoDB health probes |
| `POST` | `/api/v1/query` | Synchronous query → structured `AgentResult` |
| `POST` | `/api/v1/query/stream` | SSE stream (`status`, `schema`, `llm_token`, `query`, `error`, `result`) |

Example sync request:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many users signed up last week?",
    "backend": "postgres",
    "allow_mutations": false
  }'
```

## Configuration

See `.env.example`. Notable variables:

| Variable | Default | Notes |
|----------|---------|-------|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_MODEL` | `gpt-4o` | — |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | — |
| `MAX_RETRY_ITERATIONS` | `3` | Self-correct attempts (1–10) |
| `DYNAMODB_ENDPOINT_URL` | unset | Set for local DynamoDB |

## Tech stack

Python · FastAPI · Uvicorn · Streamlit · Pydantic · SQLAlchemy · PostgreSQL · boto3 / DynamoDB · OpenAI SDK · Anthropic SDK · SSE · httpx · Pandas

## Limitations

- Schema pruning uses lexical (bag-of-words) cosine similarity, not vector embeddings
- Mutation blocking is pattern-based and intended as a default guard, not a full security boundary
- No auth, multi-tenancy, or evaluation harness in this repository

## License

No license file is included yet. All rights reserved unless otherwise specified.
