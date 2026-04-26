# Coder Buddy

A self-correcting autonomous coding agent. Give it a natural language prompt, and it writes Python code, runs it in an isolated sandbox, and debugs itself until the code works — or it hits the retry limit.

```
prompt → write → execute → evaluate → (retry / refactor) → response
```

Built on [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration and [Pydantic AI](https://ai.pydantic.dev/) for type-safe structured LLM output.

---

## Features

- **Self-correcting loop** — automatically retries on execution errors, feeding logs back to the LLM
- **Pluggable sandbox backends** — `subprocess+venv` (default), Docker, E2B, or Pyodide
- **Multiple LLM backends** — Gemini 1.5 Pro (default), GPT-4o, Claude 3.5 Sonnet
- **Code refactoring** — cleans up and comments working code before delivery
- **Explanation mode** — plain-language description of what the code does
- **Automated test generation** — generates and runs a pytest suite for the final code
- **Confidence scoring** — LLM self-rates its solution from 1 (low) to 5 (high)
- **Refactor diff** — unified diff showing exactly what the refactor step changed
- **Token tracking** — per-node token counts and estimated cost per run
- **Session memory** — remembers the last 10 prompts/outputs within a session

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd coder-buddy

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

For the E2B cloud sandbox backend, install the optional extra:

```bash
uv sync --extra e2b
# or
pip install -e ".[e2b]"
```

---

## API Keys

The agent automatically loads a `.env` file from the project root on startup. Copy the provided template and fill in your keys:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
# .env — gitignored, never committed
GEMINI_API_KEY=your-gemini-key-here
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
E2B_API_KEY=your-e2b-key-here   # only needed for the e2b sandbox backend
```

| LLM backend | Environment variable |
|---|---|
| `gemini-1.5-pro` (default) | `GEMINI_API_KEY` |
| `gpt-4o` | `OPENAI_API_KEY` |
| `claude-3-5-sonnet` | `ANTHROPIC_API_KEY` |

You only need to set the key for the backend you're actually using. You can also pass it directly via `AgentConfig.llm_api_key` if you prefer not to use a file.

---

## Quick Start

```python
from coder_buddy import CoderBuddy
from coder_buddy.config import AgentConfig

# Create an agent with default settings
agent = CoderBuddy(AgentConfig())

# Run a prompt
response = agent.run("Write a script that prints the first 10 Fibonacci numbers")

if response.success:
    print(response.source_code)
else:
    print("Failed:", response.failure_reason)
```

---

## The Response Object

`agent.run()` always returns an `AgentResponse`, whether the run succeeded or failed.

```python
response = agent.run("Sort a list of integers using merge sort")

# Core fields — always present
response.success          # bool
response.source_code      # str  — final generated code
response.file_name        # str  — e.g. "main.py"
response.dependencies     # list[str] — pip packages used
response.execution_logs   # str  — stdout + stderr from the sandbox
response.retry_count      # int  — how many retries were needed
response.elapsed_seconds  # float

# Enrichment fields — present on success (when enabled)
response.explanation      # str | None  — plain-language explanation
response.test_code        # str | None  — generated pytest suite
response.confidence_score # int | None  — 1 (low) to 5 (high)
response.refactor_diff    # str | None  — unified diff from refactor step

# Token usage
response.token_usage.total_input_tokens       # int
response.token_usage.total_output_tokens      # int
response.token_usage.total_estimated_cost_usd # float | None

# Failure details — populated when success=False
response.failure_reason   # str | None
response.warning          # str | None
```

---

## Configuration

All options are set via `AgentConfig`:

```python
from coder_buddy.config import AgentConfig

config = AgentConfig(
    # LLM backend
    llm_backend="gemini-1.5-pro",   # "gemini-1.5-pro" | "gpt-4o" | "claude-3-5-sonnet"
    llm_api_key=None,               # or pass key directly; falls back to env var

    # Sandbox backend
    sandbox_backend="subprocess+venv",  # "subprocess+venv" | "docker" | "e2b" | "pyodide"
    sandbox_timeout_seconds=10.0,       # max seconds per execution

    # Retry behaviour
    max_retries=5,                  # 1–10 inclusive

    # Session memory
    session_history_context_n=5,    # how many past interactions to include in prompts
    session_history_max=10,         # hard cap on stored history entries

    # Feature flags
    explanation_enabled=True,       # generate plain-language explanation
    test_generation_enabled=True,   # generate and run a pytest suite
    diff_view_enabled=True,         # compute unified diff from refactor step
)

agent = CoderBuddy(config)
```

### Choosing a sandbox backend

| Backend | Requires | Best for |
|---|---|---|
| `subprocess+venv` | Python 3.11+ (default) | Local development, no extra setup |
| `docker` | Docker daemon running | Stronger isolation |
| `e2b` | `E2B_API_KEY` env var + `pip install -e ".[e2b]"` | Cloud execution |
| `pyodide` | Pyodide runtime (e.g. JupyterLite) | Browser / WASM environments |

---

## Session Memory

A single `CoderBuddy` instance remembers the last 10 runs. You can reference prior work in follow-up prompts:

```python
agent = CoderBuddy(AgentConfig())

r1 = agent.run("Write a function that reads a CSV file and returns a list of dicts")
r2 = agent.run("Now add error handling to the script you just wrote")
r3 = agent.run("Make the previous code faster using pandas")

# Clear history when you want a fresh start
agent.reset()
```

Reference keywords that trigger context injection: `"the script"`, `"the code"`, `"you just wrote"`, `"from before"`, `"previous"`, `"last one"`, `"above"`, etc.

---

## Examples

### Use GPT-4o with Docker isolation

```python
from coder_buddy import CoderBuddy
from coder_buddy.config import AgentConfig

agent = CoderBuddy(AgentConfig(
    llm_backend="gpt-4o",
    sandbox_backend="docker",
))

response = agent.run("Write a web scraper that fetches the title of https://example.com")
print(response.source_code)
```

### Disable explanation and tests for faster runs

```python
agent = CoderBuddy(AgentConfig(
    explanation_enabled=False,
    test_generation_enabled=False,
    diff_view_enabled=False,
))

response = agent.run("Generate a random password of length 16")
print(response.source_code)
print(f"Completed in {response.elapsed_seconds:.1f}s, {response.retry_count} retries")
```

### Check token usage and cost

```python
response = agent.run("Implement binary search on a sorted list")

usage = response.token_usage
print(f"Input tokens:  {usage.total_input_tokens}")
print(f"Output tokens: {usage.total_output_tokens}")
if usage.total_estimated_cost_usd is not None:
    print(f"Estimated cost: ${usage.total_estimated_cost_usd:.6f}")
```

### Handle failures gracefully

```python
agent = CoderBuddy(AgentConfig(max_retries=3))

response = agent.run("Write a script that does something complex")

if response.success:
    print("Code:")
    print(response.source_code)

    if response.confidence_score is not None and response.confidence_score <= 2:
        print(f"⚠ Low confidence ({response.confidence_score}/5): {response.warning}")

    if response.explanation:
        print("\nExplanation:")
        print(response.explanation)

    if response.test_code:
        print("\nGenerated tests:")
        print(response.test_code)
else:
    print(f"Failed after {response.retry_count} retries")
    print(f"Reason: {response.failure_reason}")
    print(f"Last logs:\n{response.execution_logs}")
```

---

## Structured Logging

The agent emits JSON log entries to stdout at each node transition. To capture them:

```python
import logging
import json

# Enable the coder_buddy logger
logging.basicConfig(level=logging.INFO)

agent = CoderBuddy(AgentConfig())
response = agent.run("Print hello world")
```

Each log line is a JSON object parseable by `json.loads()`:

```json
{"ts": 1700000000.0, "node": "write_node", "event": "start", "retry_count": 0, "outcome": null}
{"ts": 1700000001.2, "node": "write_node", "event": "end",   "retry_count": 0, "outcome": "success"}
{"ts": 1700000005.8, "event": "run_complete", "success": true, "retry_count": 0, "elapsed_seconds": 5.8, "confidence_score": 4, "token_usage": {...}}
```

---

## Running Tests

```bash
# All tests (unit + property-based; excludes slow integration tests)
uv run pytest tests/ --ignore=tests/test_integration.py -q

# Integration tests (requires a valid LLM API key; takes ~5 minutes)
uv run pytest tests/test_integration.py -v

# Full suite
uv run pytest
```

---

## Project Structure

```
coder_buddy/
├── agent.py           # CoderBuddy class — public API and session management
├── graph.py           # LangGraph StateGraph construction
├── state.py           # AgentState TypedDict
├── models.py          # Pydantic models: CodeArtifact, AgentResponse, TokenUsage
├── config.py          # AgentConfig dataclass and exception hierarchy
├── logging_utils.py   # Structured JSON logger
├── nodes/
│   ├── write_node.py      # Code generation
│   ├── execute_node.py    # Sandbox execution
│   ├── evaluator.py       # Conditional router (retry / refactor / fail)
│   ├── refactor_node.py   # Code cleanup and commenting
│   ├── test_node.py       # Test suite generation and execution
│   └── post_process.py    # Explanation and confidence scoring
├── sandbox/
│   ├── base.py                # SandboxBackend abstract interface
│   ├── subprocess_venv.py     # Default: temp venv + subprocess
│   ├── docker_backend.py      # Docker container
│   ├── e2b_backend.py         # E2B cloud sandbox
│   └── pyodide_backend.py     # Pyodide / WASM
└── llm/
    ├── client.py    # LLMClient wrapper around Pydantic AI
    └── pricing.py   # Per-token cost table
```

---

## Limitations (V1)

- **Python only** — only `"python"` is supported as the code language
- **Single-file scripts** — multi-file projects are not supported
- **In-memory history** — session history is not persisted to disk
- **Pyodide** — pure-Python packages only (no C extensions via micropip)

---

## License

See [LICENSE](LICENSE).
