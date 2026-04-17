# Consulting Intelligence Agent (SDK)

An autonomous multi-agent system that generates structured consulting meeting prep briefs — researching, writing, evaluating, and revising without human intervention.

Built to explore the full production agent stack: the Claude Agent SDK as the core runtime, with LangGraph and CrewAI implementations of the same workflow for direct comparison. Every major concept in modern agent engineering is implemented here — not as a demo, but as working code.

---

## What It Does

Given a topic and optional client name, the agent:

1. **Routes** the request to the right workflow (meeting prep / competitive analysis / tech evaluation)
2. **Researches** in parallel — three specialist researchers run simultaneously via `asyncio.gather`
3. **Writes** a structured consulting brief using a custom Skill
4. **Evaluates** the brief on a 30-point rubric
5. **Revises** automatically if score < 20 (up to 2 cycles)
6. **Waits for human approval** before saving the final output
7. **Remembers** what worked and what didn't — memory persists across runs

---

## Three Implementations — Same Workflow

One of the goals of this project is to compare orchestration approaches on identical logic:

| File | Runtime | Routing | State Management |
|---|---|---|---|
| `src/agent.py` | Claude Agent SDK | Python code — explicit | Python variables |
| `src/langgraph_agent.py` | LangGraph | Graph edges + routing functions | Typed `StateGraph` dict |
| `src/crewai_agent.py` | CrewAI | Sequential process | Implicit via task `context` |

The eval loop (researcher → analyst → evaluator → revise if score < 20) is the same pattern in all three. In the SDK you write the `if/else` yourself. In LangGraph it's `add_conditional_edges`. In CrewAI it requires hierarchical process. Same problem, very different implementation complexity.

---

## Architecture (SDK version)

```
User request
      │
      ▼
┌──────────┐
│  ROUTER  │  ← classifies: meeting_prep / competitive_analysis / tech_evaluation
└────┬─────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         PARALLEL RESEARCH               │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ Industry │ │ Company  │ │  Tech   │ │  ← 3 researchers run simultaneously
│  └──────────┘ └──────────┘ └─────────┘ │
└────┬────────────────────────────────────┘
     │
     ▼
┌──────────┐
│ ANALYST  │  ← synthesizes research into structured brief (uses consulting-brief Skill)
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

---

## Agentic Patterns Implemented

| Pattern | Where |
|---|---|
| Router | `classify_request()` — classifies request, code branches deterministically |
| Parallelization | `parallel_research()` — 3 researchers via `asyncio.gather` |
| Orchestrator-Worker | `workflow_meeting_prep()` — Python coordinates all phases sequentially |
| Evaluator-Optimizer | Eval/revision loop — evaluator scores, analyst revises if needed |
| Human-in-the-Loop | `human_approval_gate()` — workflow pauses before finalising |

---

## Subagents

| Agent | Role | Tools |
|---|---|---|
| Router | Classifies requests | None |
| Industry Researcher | Macro trends, market dynamics | `search_industry` MCP, WebSearch, WebFetch |
| Company Researcher | Company news, strategy, financials | `research_company` MCP, WebSearch, WebFetch |
| Tech Researcher | Technology and AI adoption trends | `get_tech_trends` MCP, WebSearch, WebFetch |
| Analyst | Synthesizes research into structured brief | Read, Write, Bash |
| Evaluator | Scores brief against 6-criterion rubric | Read, Bash |

**Evaluation rubric (30 points):** executive summary, research depth, talking points, risk awareness, strategic questions, overall readiness — scored 1-5 each. Score ≥ 20 passes.

---

## Project Structure

```
consulting-agent-with-sdk/
├── .well-known/
│   └── agent.json           ← A2A Agent Card (skills, capabilities, endpoint)
├── skills/
│   ├── consulting-brief-generator/SKILL.md  ← brief format the analyst follows
│   └── eval-consulting-brief/SKILL.md       ← rubric the evaluator uses
├── src/
│   ├── agent.py             ← main orchestrator (Claude Agent SDK)
│   ├── models.py            ← Pydantic models (EvalResult, RouteClassification, Episode)
│   ├── tools.py             ← custom MCP tools (FastMCP + Tavily)
│   ├── memory.py            ← episodic + semantic + session memory
│   ├── a2a_server.py        ← exposes agent as A2A-compliant HTTP service
│   ├── a2a_client.py        ← discovers and calls remote A2A agents
│   ├── langgraph_agent.py   ← same workflow as LangGraph StateGraph (OpenAI)
│   └── crewai_agent.py      ← same workflow as CrewAI crew (OpenAI)
├── output/                  ← generated briefs saved here
├── memory/                  ← persisted agent memory (auto-created)
│   ├── episodes.jsonl
│   ├── semantic.json
│   └── sessions.json
├── .env.example
└── pyproject.toml
```

---

## Agent Memory

`memory.py` implements three memory types that persist to disk across runs:

**Episodic** — records each run (topic, score, what worked, what failed) as JSONL. Relevant past episodes are retrieved by similarity and injected into analyst prompts — the agent learns from past mistakes.

**Semantic** — accumulated facts and preferences extracted from evaluator feedback ("talking points often score low on first draft", "Tavily returns better results with the year in the query"). Stored as JSON, categories: `preference`, `domain_knowledge`, `tool_insight`, `quality_pattern`.

**Session** — saves Agent SDK session IDs for resume capability. Long-running workflows can restart from the exact point they left off.

---

## A2A Protocol

The agent is fully A2A-compliant — callable by any A2A-compatible agent regardless of framework.

```bash
# Start the A2A server
uv run python src/a2a_server.py
# Agent Card: http://localhost:8000/.well-known/agent.json
# A2A endpoint: http://localhost:8000/a2a

# Call it from another terminal
uv run python src/a2a_client.py
```

Skills exposed: `meeting_prep`, `competitive_analysis`, `technology_evaluation`

The `multi_agent_consulting_brief()` function in `a2a_client.py` demonstrates parallel orchestration of multiple A2A agents — the same parallelization pattern but now distributed across the network.

---

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/abhisheksett/consulting-agent-with-sdk
cd consulting-agent-with-sdk
uv sync

cp .env.example .env
# Add to .env:
# TAVILY_API_KEY=...       (required for all versions)
# OPENAI_API_KEY=...       (required for LangGraph + CrewAI versions)
```

**Note:** The Claude Agent SDK version (`agent.py`) runs via Claude Code — no Anthropic API key needed. The LangGraph and CrewAI versions use OpenAI.

---

## Usage

```bash
# Claude Agent SDK version (requires Claude Code)
uv run python -m src.agent "AI agents in healthcare" "UnitedHealth Group"

# LangGraph version (requires OPENAI_API_KEY)
uv run python src/langgraph_agent.py

# CrewAI version (requires OPENAI_API_KEY)
uv run python src/crewai_agent.py

# A2A server
uv run python src/a2a_server.py
```

The brief is saved to `output/brief-<topic>.md`.

---

## Key Concepts Demonstrated

- **Claude Agent SDK** — `query()` loop, `AgentDefinition`, subagent delegation, session management
- **Custom MCP tools** — FastMCP `@tool` decorator, in-process server
- **Agent Skills** — `SKILL.md` format, progressive disclosure, portable domain knowledge
- **Agent Memory** — episodic, semantic, session — persisted across runs
- **A2A Protocol** — Agent Card, Task lifecycle, JSON-RPC 2.0, push notifications, parallel multi-agent orchestration
- **LangGraph** — `StateGraph`, typed state, conditional edges, eval loop as graph construct
- **CrewAI** — role-based agents, task context passing, sequential crew execution

---

## Related Projects

- [rag-explorer](https://github.com/abhisheksett/rag-explorer) — Naive vs Hybrid vs Agentic RAG on cloud architecture docs
- [consulting-agent](https://github.com/abhisheksett/consulting-agent) — Claude Code version of the same agent (skills + subagents, no SDK)
