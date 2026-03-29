# Consulting Intelligence Agent (Agent SDK)

A multi-agent system that generates polished consulting meeting preparation briefs using the Claude Agent SDK. The pipeline orchestrates three specialized subagents — researcher, analyst, and evaluator — with your code driving every phase rather than relying on the model to self-direct.

## How it works

The agent runs a four-phase pipeline:

1. **Research** — A researcher subagent uses Tavily-powered MCP tools (`search_industry`, `research_company`, `get_tech_trends`) plus `WebSearch`/`WebFetch` to gather current intelligence.
2. **Synthesis** — An analyst subagent structures the findings into a client-ready brief following the `consulting-brief-generator` skill format and saves it to `output/`.
3. **Evaluation** — An evaluator subagent scores the brief against a six-criterion rubric (executive summary, research depth, talking points, risk awareness, strategic questions, overall readiness) and returns structured JSON.
4. **Revision** — If the brief scores below 20/30, your code passes the evaluator's specific feedback back to the analyst for a targeted revision. This loop repeats until the quality bar is met.

The key distinction from a Claude Code slash command: **your orchestrator code** parses every output, checks every score, and makes every branching decision deterministically. The subagents do not talk to each other directly.

## Project structure

```
src/
  agent.py      # Orchestration logic and subagent definitions
  models.py     # Pydantic models for evaluator output validation
  tools.py      # Tavily MCP tools exposed as an in-process MCP server
skills/
  consulting-brief-generator/SKILL.md   # Brief format the analyst follows
  eval-consulting-brief/SKILL.md        # Rubric the evaluator uses
output/          # Generated briefs are saved here
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

Run the agent from the `src/` directory:

```bash
cd src
uv run python agent.py "AI agents in healthcare" "UnitedHealth Group"
```

Arguments:

- `topic` (required) — industry or technology focus
- `client_name` (optional) — specific company to research

The brief is saved to `output/brief-<topic>.md`.

## Dependencies

| Package            | Purpose                                        |
| ------------------ | ---------------------------------------------- |
| `claude-agent-sdk` | Agent orchestration and subagent delegation    |
| `mcp`              | In-process MCP server for custom tools         |
| `tavily-python`    | Web search and company research                |
| `pydantic`         | Structured validation of evaluator JSON output |
| `python-dotenv`    | API key loading from `.env`                    |
