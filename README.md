# Consulting Intelligence Agent (Agent SDK)

A multi-agent system that generates polished consulting deliverables using the Claude Agent SDK. The pipeline implements five agentic design patterns — **Router**, **Parallelization**, **Orchestrator-Worker**, **Evaluator-Optimizer**, and **Human-in-the-Loop** — with your code driving every phase rather than relying on the model to self-direct.

## Agentic patterns

| #   | Pattern                 | Where it lives                                                                                                                    |
| --- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Router**              | `classify_request()` — a zero-tool agent classifies the user's request into a workflow type; your code branches deterministically |
| 2   | **Parallelization**     | `parallel_research()` — three specialized researchers (industry, company, technology) run simultaneously via `asyncio.gather`     |
| 3   | **Orchestrator-Worker** | `workflow_meeting_prep()` — your code coordinates research → synthesis → evaluation → revision phases sequentially                |
| 4   | **Evaluator-Optimizer** | Eval/revision loop — an evaluator subagent scores the brief against a rubric; if it fails, the analyst revises (up to 2 rounds)   |
| 5   | **Human-in-the-Loop**   | `human_approval_gate()` — the workflow pauses for human approval before finalizing the brief                                      |

## How it works

```
User request
     │
     ▼
┌──────────┐
│  ROUTER  │  ← classifies into meeting_prep / competitive_analysis / technology_evaluation
└────┬─────┘
     │
     ▼
┌─────────────────────────────────────┐
│     PARALLEL RESEARCH               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Industry │ │ Company  │ │   Tech   │  │  ← 3 researchers run simultaneously
│  └──────────┘ └──────────┘ └──────────┘  │
└────┬────────────────────────────────┘
     │
     ▼
┌──────────┐
│ ANALYST  │  ← synthesizes research into a client-ready brief
└────┬─────┘
     │
     ▼
┌──────────────┐     ┌──────────┐
│  EVALUATOR   │────▶│ ANALYST  │  ← revision loop (max 2 rounds) if score < 20/30
└──────┬───────┘     └──────────┘
       │
       ▼
┌──────────────────┐
│ HUMAN APPROVAL   │  ← approve / revise with feedback / reject
└──────┬───────────┘
       │
       ▼
   output/brief-<topic>.md
```

### Subagents

| Agent                   | Role                                         | Tools                                           |
| ----------------------- | -------------------------------------------- | ----------------------------------------------- |
| **Router**              | Classifies requests into workflow types      | None (classification only)                      |
| **Industry Researcher** | Macro trends, market size, sector dynamics   | `search_industry` MCP, `WebSearch`, `WebFetch`  |
| **Company Researcher**  | Company news, strategy, financials           | `research_company` MCP, `WebSearch`, `WebFetch` |
| **Tech Researcher**     | Enterprise technology and AI adoption trends | `get_tech_trends` MCP, `WebSearch`, `WebFetch`  |
| **Analyst**             | Synthesizes research into a structured brief | `Read`, `Write`, `Bash`                         |
| **Evaluator**           | Scores briefs against a six-criterion rubric | `Read`, `Bash`                                  |

### Evaluation rubric (30-point scale)

Each criterion is scored 1–5: executive summary, research depth, talking points, risk awareness, strategic questions, overall readiness. A score ≥ 20 passes.

### Memory system

The project includes a file-based memory system (`src/memory.py`) with three memory types:

- **Episodic** — records each run (topic, score, feedback, what worked/failed) as JSONL. Relevant episodes are injected into analyst prompts so the agent learns from past mistakes.
- **Semantic** — accumulated facts and preferences (e.g. "talking points often score low on first draft") stored as JSON. Categories: `preference`, `domain_knowledge`, `tool_insight`, `quality_pattern`.
- **Session** — session IDs for Agent SDK resume capability, enabling long-running workflows to pick up mid-way.

### A2A (Agent-to-Agent) protocol

The project implements Google's A2A protocol for inter-agent communication:

- **`src/a2a_server.py`** — exposes the consulting agent as an A2A-compliant HTTP service with JSON-RPC 2.0. Any A2A client (regardless of framework) can discover, call, and poll your agent.
- **`src/a2a_client.py`** — client for calling remote A2A agents. Includes `A2AClient` for discovery, task submission, and polling, plus a `multi_agent_consulting_brief()` function that orchestrates multiple specialist agents in parallel over the network.
- **`.well-known/agent.json`** — the Agent Card describing capabilities, skills, and authentication.

## Project structure

```
src/
  agent.py        # Orchestration: router, parallel research, eval loop, human gate
  models.py       # Pydantic models (EvalResult, RouteClassification, Episode, SemanticFact)
  tools.py        # Tavily MCP tools (search_industry, research_company, get_tech_trends)
  memory.py       # Episodic, semantic, and session memory system
  a2a_server.py   # A2A server — exposes agent over HTTP/JSON-RPC
  a2a_client.py   # A2A client — discover and call remote agents
skills/
  consulting-brief-generator/SKILL.md   # Brief format the analyst follows
  eval-consulting-brief/SKILL.md        # Rubric the evaluator uses
.well-known/
  agent.json      # A2A Agent Card
output/            # Generated briefs are saved here
```

## Setup

**Prerequisites:** Python 3.12+, [`uv`](https://docs.astral.sh/uv/)

1. Clone the repo and install dependencies:

   ```bash
   uv sync
   ```

2. Create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   TAVILY_API_KEY=tvly-...
   ```

## Usage

### Run the agent directly

```bash
uv run python -m src.agent "AI agents in healthcare" "UnitedHealth Group"
```

Arguments:

- `topic` (required) — industry or technology focus
- `client_name` (optional) — specific company to research

The router auto-classifies the request into `meeting_prep`, `competitive_analysis`, or `technology_evaluation`. If confidence is low, you're prompted to confirm.

The brief is saved to `output/brief-<topic>.md`.

### Run as an A2A server

```bash
uv run python src/a2a_server.py
# Server starts at http://localhost:8000
# Agent Card: http://localhost:8000/.well-known/agent.json
# A2A endpoint: http://localhost:8000/a2a
```

### Call the agent via A2A client

```bash
# (with the A2A server running in another terminal)
uv run python src/a2a_client.py
```

## Dependencies

| Package            | Purpose                                     |
| ------------------ | ------------------------------------------- |
| `claude-agent-sdk` | Agent orchestration and subagent delegation |
| `mcp`              | In-process MCP server for custom tools      |
| `tavily-python`    | Web search and company research             |
| `pydantic`         | Structured validation of agent JSON outputs |
| `python-dotenv`    | API key loading from `.env`                 |
