"""
Consulting Intelligence Agent — Agent SDK Orchestrator

THIS IS THE KEY DIFFERENCE FROM CLAUDE CODE:
- Claude Code: you type a slash command, the MODEL decides the flow
- Agent SDK: YOUR CODE controls every phase, parses every output,
  decides every branch

Same subagents, same skills, same MCP tools — different driver.
"""

from tools import research_server
from models import EvalResult
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AgentDefinition,
    AssistantMessage,
    TextBlock,
)
import asyncio
import json
import re
import sys
import os

from dotenv import load_dotenv

load_dotenv()  # loads ANTHROPIC_API_KEY and TAVILY_API_KEY from .env


# ═══════════════════════════════════════════════
# SECTION 1: SUBAGENT DEFINITIONS
# ═══════════════════════════════════════════════
#
# These are the SAME agents as your .claude/agents/*.md files,
# but defined programmatically. The SDK lets you do either:
#   - Programmatic (below) — full control in code
#   - Filesystem (.claude/agents/*.md) — reuse Claude Code files
#
# Programmatic takes precedence if both exist with the same name.
#
# Each AgentDefinition has:
#   - name: identifier (how you reference it)
#   - description: Claude reads this to decide WHEN to delegate
#   - prompt: the subagent's system prompt (its personality/instructions)
#   - tools: allowlist of tools this subagent can use
#
# Note: the "model" field is optional. If omitted, the subagent
# uses the same model as the parent. You could use "sonnet" for
# cheaper subagents or "opus" for the critical ones.
# ═══════════════════════════════════════════════

RESEARCHER = AgentDefinition(
    description=(
        "Research specialist for gathering raw intelligence from the web. "
        "Use when you need current data about companies, industries, "
        "technology trends, or market analysis."
    ),
    prompt=(
        "You are a senior research analyst at a top-tier consulting firm.\n\n"
        "## Your job\n"
        "Gather comprehensive, CURRENT intelligence on the given topic.\n\n"
        "## Rules\n"
        "1. Use the consulting-research MCP tools as primary sources\n"
        "2. Also use WebSearch and WebFetch for additional depth\n"
        "3. Focus on FACTS and DATA — no opinions, no filler\n"
        "4. Always include source URLs for every finding\n"
        "5. Organize findings into categories:\n"
        "   - Industry data & market trends\n"
        "   - Company-specific news & strategy\n"
        "   - Technology & AI relevance\n"
        "   - Competitive landscape\n"
        "6. Aim for 10-15 key findings minimum\n"
        "7. Return a structured markdown summary"
    ),
    tools=["WebSearch", "WebFetch", "Bash", "Read"],
    # MCP tools (search_industry, etc.) are inherited from parent's
    # mcp_servers config — no need to list them here
)

ANALYST = AgentDefinition(
    description=(
        "Consulting analyst that synthesizes raw research into "
        "structured client-ready briefs. Use after research is "
        "gathered and needs to be turned into a polished deliverable."
    ),
    prompt=(
        "You are a senior consulting analyst.\n\n"
        "## Your job\n"
        "Take raw research findings and synthesize them into a polished, "
        "client-ready meeting preparation brief.\n\n"
        "## Rules\n"
        "1. Follow the consulting-brief skill format EXACTLY\n"
        "2. Every claim MUST be backed by the research provided\n"
        "3. Write for a partner audience — crisp, authoritative, actionable\n"
        "4. Target 500-800 words\n"
        "5. Save the final brief to the output/ directory\n"
        "6. If given revision feedback, only fix the cited weaknesses — "
        "   keep strong sections unchanged"
    ),
    tools=["Read", "Write", "Bash"],
)

EVALUATOR = AgentDefinition(
    description=(
        "Quality evaluator that scores consulting briefs against "
        "a rubric. Use to assess whether a brief meets the quality "
        "bar for a partner to use in a client meeting."
    ),
    prompt=(
        "You are a senior partner reviewing meeting prep materials. "
        "You are demanding but fair.\n\n"
        "## Your job\n"
        "Score the brief using the eval-brief skill rubric.\n\n"
        "## Rules\n"
        "1. Score each criterion 1-5 (be honest, not generous)\n"
        "2. Output ONLY valid JSON in this exact format:\n"
        '{\n'
        '  "scores": {\n'
        '    "executive_summary": <1-5>,\n'
        '    "research_depth": <1-5>,\n'
        '    "talking_points": <1-5>,\n'
        '    "risk_awareness": <1-5>,\n'
        '    "strategic_questions": <1-5>,\n'
        '    "overall_readiness": <1-5>\n'
        '  },\n'
        '  "total_score": <sum>,\n'
        '  "passed": <true if >= 20>,\n'
        '  "feedback": "<specific improvement suggestions>",\n'
        '  "missing_elements": ["<list>", "<of>", "<gaps>"]\n'
        '}\n'
        "3. If total < 20, provide specific, actionable revision instructions\n"
        "4. No preamble, no markdown wrapping — just the JSON"
    ),
    tools=["Read", "Bash"],
)


# ═══════════════════════════════════════════════
# SECTION 2: HELPER FUNCTION
# ═══════════════════════════════════════════════
#
# This wraps the SDK's query() call into a simple
# "send prompt, get text back" function.
#
# The raw SDK streams message objects (AssistantMessage,
# SystemMessage, ToolUseMessage, etc.). We extract just
# the text content for cleaner orchestration.
# ═══════════════════════════════════════════════

async def run_query(
    prompt: str,
    agents: dict[str, AgentDefinition],
    mcp_servers: dict | None = None,
    max_turns: int = 15,
) -> str:
    """
    Run a single agent query and return the collected text response.

    Args:
        prompt: The instruction for this phase
        agents: Dict of available subagent definitions
        mcp_servers: Dict of MCP servers (in-process or external)
        max_turns: Safety limit on agent loop iterations

    Returns:
        Collected text output as a string
    """
    options = ClaudeAgentOptions(
        # Tools the MAIN agent can use (includes Agent for subagent delegation)
        allowed_tools=[
            "Read", "Write", "Bash",
            "WebSearch", "WebFetch",
            "Glob", "Grep",
            "Agent",  # <-- THIS enables subagent delegation
        ],
        agents=agents,
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
        permission_mode="bypassPermissions",

        # Load skills from project filesystem (skills/ directory)
        # Without this, the SDK won't discover your SKILL.md files
        setting_sources=["project"],
    )

    collected_text = []

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    collected_text.append(block.text)

    return "\n".join(collected_text)


# ═══════════════════════════════════════════════
# SECTION 3: JSON PARSER HELPER
# ═══════════════════════════════════════════════
#
# LLMs sometimes wrap JSON in markdown code fences
# or add preamble text. This extracts the JSON reliably.
# ═══════════════════════════════════════════════

def extract_json(text: str) -> dict | None:
    """Extract a JSON object from text that may contain markdown or preamble."""

    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object in the text
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


# ═══════════════════════════════════════════════
# SECTION 4: THE MAIN ORCHESTRATION FLOW
# ═══════════════════════════════════════════════
#
# THIS is where the Agent SDK shines vs Claude Code.
#
# In Claude Code, you wrote a slash command and hoped
# the model followed it correctly.
#
# Here, YOUR CODE:
#   1. Calls each phase explicitly
#   2. Parses the output between phases
#   3. Passes specific data to the next phase
#   4. Checks the eval score with real code (not LLM judgment)
#   5. Decides deterministically whether to revise
#
# Same subagents, same skills, same MCP — but YOU drive.
# ═══════════════════════════════════════════════

async def generate_brief(topic: str, client_name: str | None = None):
    """
    Generate a consulting meeting prep brief through 4 phases.

    Args:
        topic: Industry or technology focus (e.g., "AI agents in healthcare")
        client_name: Optional specific client (e.g., "UnitedHealth Group")
    """

    # Bundle all agents — the main agent can delegate to any of them
    all_agents = {
        "researcher": RESEARCHER,
        "analyst": ANALYST,
        "evaluator": EVALUATOR,
    }

    # In-process MCP server with Tavily search tools
    mcp = {"consulting-research": research_server}

    # Build the meeting context string
    context = f"Topic: {topic}"
    if client_name:
        context += f"\nClient: {client_name}"

    output_filename = f"brief-{topic.lower().replace(' ', '-')[:50]}.md"
    output_path = f"output/{output_filename}"

    print("═" * 55)
    print(f"  Consulting Intelligence Agent")
    print(f"  Topic: {topic}")
    if client_name:
        print(f"  Client: {client_name}")
    print("═" * 55)

    # ── PHASE 1: RESEARCH ──────────────────────
    #
    # Delegate to researcher subagent.
    # It uses MCP tools (search_industry, research_company, get_tech_trends)
    # and built-in WebSearch/WebFetch in its own isolated context.
    # We get back only the condensed findings.
    #
    print("\n🔍 Phase 1: Research")
    print("   Delegating to researcher subagent...")

    research_prompt = f"""
    Use the researcher subagent to gather comprehensive intelligence.

    {context}

    The researcher should use:
    - search_industry tool for broad industry trends
    - research_company tool for client-specific data (if client given)
    - get_tech_trends tool for technology/AI relevance
    - WebSearch for additional depth

    Return structured findings with source URLs.
    """

    findings = await run_query(
        prompt=research_prompt,
        agents=all_agents,
        mcp_servers=mcp,
        max_turns=20,
    )

    print(f"   ✅ Research complete ({len(findings)} chars)")

    # ── PHASE 2: SYNTHESIS ─────────────────────
    #
    # YOUR CODE passes the findings to the analyst subagent.
    # The analyst loads the consulting-brief skill from skills/ directory
    # and structures the output accordingly.
    #
    # Note: we truncate findings to 8000 chars to leave room
    # in the analyst's context for the skill instructions.
    #
    print("\n📝 Phase 2: Synthesis")
    print("   Delegating to analyst subagent...")

    synthesis_prompt = f"""
    Use the analyst subagent to create a consulting meeting prep brief.

    ## Meeting context
    {context}

    ## Research findings to synthesize
    {findings[:8000]}

    The analyst must:
    1. Follow the consulting-brief skill format EXACTLY
    2. Back every claim with the research findings above
    3. Save the brief to {output_path}
    """

    brief = await run_query(
        prompt=synthesis_prompt,
        agents=all_agents,
        max_turns=15,
    )

    print(f"   ✅ Brief drafted ({len(brief)} chars)")

    # ── PHASE 3: EVALUATION ────────────────────
    #
    # YOUR CODE sends the brief to the evaluator subagent.
    # The evaluator loads the eval-brief skill and returns JSON scores.
    # Then YOUR CODE parses the JSON — not the model.
    #
    print("\n⚖️  Phase 3: Evaluation")
    print("   Delegating to evaluator subagent...")

    eval_prompt = f"""
    Use the evaluator subagent to score this consulting brief.

    ## Brief to evaluate
    {brief[:6000]}

    Score using the eval-brief skill rubric.
    Return ONLY a valid JSON object — no preamble, no markdown.
    """

    eval_response = await run_query(
        prompt=eval_prompt,
        agents=all_agents,
        max_turns=10,
    )

    # YOUR CODE parses the score — not the model
    eval_json = extract_json(eval_response)

    if eval_json:
        try:
            eval_result = EvalResult.model_validate(eval_json)
            total = eval_result.total_score
            passed = eval_result.passed
            feedback = eval_result.feedback
            missing = eval_result.missing_elements
        except Exception as e:
            print(f"   ⚠️  Pydantic validation failed: {e}")
            print(f"   Raw JSON: {eval_json}")
            total, passed, feedback, missing = 0, False, "Parse error", []
    else:
        print(f"   ⚠️  Could not extract JSON from eval response")
        print(f"   Raw response: {eval_response[:500]}")
        total, passed, feedback, missing = 0, False, "No JSON found", []

    print(f"   Score: {total}/30 — {'✅ PASS' if passed else '❌ FAIL'}")

    # ── PHASE 4: REVISION (deterministic) ──────
    #
    # THIS is the critical difference from Claude Code.
    #
    # Claude Code: the model reads the score and MIGHT revise.
    # Agent SDK: YOUR CODE checks the score and ALWAYS revises if < 20.
    #
    # You could also:
    #   - retry up to N times
    #   - log every revision to a database
    #   - send an alert if the brief can't pass after 2 attempts
    #   - escalate to a human reviewer
    #
    # None of this is possible with Claude Code's implicit flow.
    #
    if not passed and total > 0:
        print("\n🔄 Phase 4: Revision")
        print(f"   Feedback: {feedback}")
        print(f"   Missing: {missing}")
        print("   Delegating back to analyst subagent...")

        revision_prompt = f"""
        Use the analyst subagent to revise the consulting brief.

        ## Original brief
        {brief[:6000]}

        ## Evaluator feedback (score: {total}/30)
        {feedback}

        ## Missing elements
        {json.dumps(missing)}

        Rules:
        1. ONLY fix the weaknesses identified above
        2. Keep strong sections unchanged
        3. Follow the consulting-brief skill format
        4. Save revised brief to {output_path}
        """

        brief = await run_query(
            prompt=revision_prompt,
            agents=all_agents,
            max_turns=15,
        )

        print("   ✅ Brief revised")

        # Optional: re-evaluate after revision
        # You could add another eval cycle here if you want.
        # In Claude Code, this is up to the model's judgment.
        # Here, it's YOUR decision.

    elif total == 0:
        print("\n⚠️  Skipping revision — eval parsing failed")

    # ── PHASE 5: OUTPUT ────────────────────────
    print("\n" + "═" * 55)
    print(f"  ✅ COMPLETE")
    print(f"  📋 Topic: {topic}")
    if client_name:
        print(f"  🏢 Client: {client_name}")
    print(f"  📊 Score: {total}/30 ({'PASS' if passed else 'FAIL'})")
    print(f"  📁 Output: {output_path}")
    print("═" * 55)

    return brief, eval_result if eval_json else None


# ═══════════════════════════════════════════════
# SECTION 5: ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    topic = (
        sys.argv[1] if len(sys.argv) > 1
        else "AI agents in financial services"
    )
    client = sys.argv[2] if len(sys.argv) > 2 else None

    brief, eval_result = asyncio.run(generate_brief(topic, client))
