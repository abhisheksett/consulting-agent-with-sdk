"""
Consulting Intelligence Agent — Agent SDK Orchestrator
with Router, Parallelization, and Human-in-the-Loop patterns.

PATTERNS IMPLEMENTED:
1. ROUTER — classifies request type, routes to correct workflow
2. PARALLELIZATION — fans out research to 3 agents simultaneously
3. ORCHESTRATOR-WORKER — coordinates multi-phase workflow
4. EVALUATOR-OPTIMIZER — bounded eval/revision loop
5. HUMAN-IN-THE-LOOP — approval gate before final output
"""

from .tools import research_server
from .models import EvalResult, RouteClassification
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

load_dotenv()


# ═══════════════════════════════════════════════
# SECTION 1: SUBAGENT DEFINITIONS
# ═══════════════════════════════════════════════

# --- ROUTER (Pattern 1: Router) ---
# This agent ONLY classifies — it does NO real work.
# Its sole job is to read the user's request and pick a workflow.
ROUTER = AgentDefinition(
    description=(
        "Request classifier. Use FIRST on every incoming request to "
        "determine which workflow to run. Never does actual work."
    ),
    prompt=(
        "You are a request classifier for a consulting intelligence system.\n\n"
        "## Your ONLY job\n"
        "Read the user's request and classify it into ONE workflow type.\n\n"
        "## Workflow types\n"
        "- meeting_prep: User wants to prepare for a client meeting. "
        "Keywords: meeting, prepare, brief, client visit, talking points.\n"
        "- competitive_analysis: User wants to compare companies or "
        "understand market position. Keywords: competitor, market share, "
        "compare, vs, landscape, positioning.\n"
        "- technology_evaluation: User wants to assess whether a "
        "technology is right for enterprise adoption. Keywords: evaluate, "
        "assess, should we adopt, technology choice, build vs buy.\n\n"
        "## Output format (strict JSON, nothing else)\n"
        '{"workflow": "meeting_prep|competitive_analysis|technology_evaluation", '
        '"confidence": 0.0-1.0, '
        '"reasoning": "one sentence why"}\n\n'
        "## Rules\n"
        "1. Pick EXACTLY one workflow\n"
        "2. If unsure, default to meeting_prep\n"
        "3. Output ONLY the JSON — no preamble, no explanation"
    ),
    tools=[],  # Router gets NO tools — it only classifies
)

# --- RESEARCHERS (Pattern 2: Parallelization) ---
# Three specialized researchers that run SIMULTANEOUSLY.
# Each focuses on one domain, keeping context lean and fast.
INDUSTRY_RESEARCHER = AgentDefinition(
    description=(
        "Industry trends researcher. Use for broad market and sector analysis."
    ),
    prompt=(
        "You are an industry analyst.\n\n"
        "## Your ONLY job\n"
        "Research industry-level trends, market size, growth rates, "
        "and sector dynamics for the given topic.\n\n"
        "## Rules\n"
        "1. Use search_industry MCP tool as primary source\n"
        "2. Also use WebSearch for depth\n"
        "3. Focus on macro trends, not company-specific news\n"
        "4. Return 5-7 findings with source URLs\n"
        "5. Structured markdown output"
    ),
    tools=["WebSearch", "WebFetch", "Read"],
)

COMPANY_RESEARCHER = AgentDefinition(
    description=(
        "Company-specific researcher. Use for news, strategy, and "
        "intelligence about a specific organization."
    ),
    prompt=(
        "You are a company intelligence analyst.\n\n"
        "## Your ONLY job\n"
        "Research a specific company — recent news, strategic moves, "
        "leadership changes, financial performance, challenges.\n\n"
        "## Rules\n"
        "1. Use research_company MCP tool as primary source\n"
        "2. Also use WebSearch for recent news\n"
        "3. Focus on last 6 months of activity\n"
        "4. Return 5-7 findings with source URLs\n"
        "5. Structured markdown output"
    ),
    tools=["WebSearch", "WebFetch", "Read"],
)

TECH_RESEARCHER = AgentDefinition(
    description=(
        "Technology trends researcher. Use for AI, cloud, and "
        "enterprise technology adoption analysis."
    ),
    prompt=(
        "You are a technology analyst.\n\n"
        "## Your ONLY job\n"
        "Research technology and AI trends relevant to the given "
        "industry or company.\n\n"
        "## Rules\n"
        "1. Use get_tech_trends MCP tool as primary source\n"
        "2. Focus on enterprise adoption, not consumer tech\n"
        "3. Include what competitors are doing with technology\n"
        "4. Return 5-7 findings with source URLs\n"
        "5. Structured markdown output"
    ),
    tools=["WebSearch", "WebFetch", "Read"],
)

# --- ANALYST & EVALUATOR (Pattern 3 & 4: Orchestrator + Eval) ---
# Same as your original consulting agent
ANALYST = AgentDefinition(
    description=(
        "Consulting analyst that synthesizes raw research into "
        "structured client-ready briefs. Use after research is "
        "gathered and needs to be turned into a polished deliverable."
    ),
    prompt=(
        "You are a senior consulting analyst at Deloitte.\n\n"
        "## Your job\n"
        "Take raw research findings and synthesize them into a polished, "
        "client-ready deliverable.\n\n"
        "## Rules\n"
        "1. Follow the appropriate skill format EXACTLY\n"
        "2. Every claim MUST be backed by the research provided\n"
        "3. Write for a partner audience — crisp, authoritative, actionable\n"
        "4. Target 500-800 words\n"
        "5. Save output to the output/ directory\n"
        "6. If given revision feedback, only fix the cited weaknesses"
    ),
    tools=["Read", "Write", "Bash"],
)

EVALUATOR = AgentDefinition(
    description=(
        "Quality evaluator that scores deliverables against "
        "a rubric. Use to assess whether output meets quality bar."
    ),
    prompt=(
        "You are a senior partner reviewing deliverables. "
        "You are demanding but fair.\n\n"
        "## Your job\n"
        "Score the deliverable using the eval-brief skill rubric.\n\n"
        "## Rules\n"
        "1. Score each criterion 1-5 (be honest, not generous)\n"
        "2. Output ONLY valid JSON in the eval-brief format\n"
        "3. If total < 20, provide specific revision instructions\n"
        "4. No preamble, no markdown wrapping — just JSON"
    ),
    tools=["Read", "Bash"],
)


# ═══════════════════════════════════════════════
# SECTION 2: HELPER FUNCTIONS
# ═══════════════════════════════════════════════

async def run_query(
    prompt: str,
    agents: dict[str, AgentDefinition],
    mcp_servers: dict | None = None,
    max_turns: int = 15,
) -> str:
    """Run a single agent query and return collected text."""
    options = ClaudeAgentOptions(
        allowed_tools=[
            "Read", "Write", "Bash",
            "WebSearch", "WebFetch",
            "Glob", "Grep",
            "Agent",
        ],
        agents=agents,
        mcp_servers=mcp_servers or {},
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        setting_sources=["project"],
    )

    collected = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    collected.append(block.text)

    return "\n".join(collected)


def extract_json(text: str) -> dict | None:
    """Extract a JSON object from text that may contain markdown."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


# ═══════════════════════════════════════════════
# PATTERN 1: ROUTER
# ═══════════════════════════════════════════════
#
# Classifies the user's request into a workflow type.
# YOUR CODE reads the classification and branches.
# The router agent has NO tools — it only classifies.
#

async def classify_request(user_input: str) -> RouteClassification:
    """Route the user's request to the appropriate workflow."""

    router_agents = {"router": ROUTER}

    response = await run_query(
        prompt=f"Classify this request:\n\n{user_input}",
        agents=router_agents,
        max_turns=5,
    )

    route_json = extract_json(response)
    if route_json:
        try:
            return RouteClassification.model_validate(route_json)
        except Exception:
            pass

    # Default fallback if classification fails
    return RouteClassification(
        workflow="meeting_prep",
        confidence=0.5,
        reasoning="Classification failed, defaulting to meeting prep",
    )


# ═══════════════════════════════════════════════
# PATTERN 2: PARALLEL RESEARCH
# ═══════════════════════════════════════════════
#
# Instead of one researcher doing everything sequentially,
# fan out to 3 specialized researchers simultaneously.
#
# asyncio.gather runs all three concurrently.
# Total time = slowest researcher, not sum of all three.
#

async def parallel_research(
    topic: str,
    client_name: str | None,
    mcp: dict,
) -> dict[str, str]:
    """Fan out research to 3 specialized researchers in parallel."""

    # Each researcher gets its own agent dict — isolated
    industry_agents = {"industry_researcher": INDUSTRY_RESEARCHER}
    company_agents = {"company_researcher": COMPANY_RESEARCHER}
    tech_agents = {"tech_researcher": TECH_RESEARCHER}

    # Build prompts for each researcher
    industry_prompt = (
        f"Use the industry_researcher subagent to research industry "
        f"trends for: {topic}"
    )

    company_prompt = (
        f"Use the company_researcher subagent to research "
        f"{'the company: ' + client_name if client_name else 'key players in: ' + topic}"
    )

    tech_prompt = (
        f"Use the tech_researcher subagent to research technology "
        f"and AI trends relevant to: {topic}"
    )

    # PARALLEL EXECUTION — all three run at the same time
    # asyncio.gather is the key — it starts all coroutines
    # concurrently and waits for all to complete.
    industry, company, tech = await asyncio.gather(
        run_query(industry_prompt, industry_agents, mcp, max_turns=15),
        run_query(company_prompt, company_agents, mcp, max_turns=15),
        run_query(tech_prompt, tech_agents, mcp, max_turns=15),
    )

    return {
        "industry": industry,
        "company": company,
        "technology": tech,
    }


# ═══════════════════════════════════════════════
# PATTERN 5: HUMAN-IN-THE-LOOP
# ═══════════════════════════════════════════════
#
# Before saving the final brief, pause and ask for
# human approval. The agent cannot proceed without it.
#
# In production, this could be:
#   - A Slack message to the partner asking for approval
#   - A web UI with approve/reject buttons
#   - An email with a review link
#
# For now, we use terminal input.
#

async def human_approval_gate(brief: str, eval_score: int) -> str:
    """
    Present the brief to the human and get approval.

    Returns:
        "approved" — save and finalize
        "revise" — user wants changes (with feedback)
        "rejected" — discard entirely
    """
    print("\n" + "═" * 55)
    print("  🔒 HUMAN APPROVAL REQUIRED")
    print("═" * 55)
    print(f"\n  Eval score: {eval_score}/30")
    print(f"  Brief length: {len(brief)} chars")
    print("\n--- BRIEF PREVIEW (first 500 chars) ---")
    print(brief[:500])
    print("--- END PREVIEW ---\n")

    # In production, replace this with an API call, webhook, or UI
    print("  Options:")
    print("  [a] Approve — save and finalize")
    print("  [r] Revise — provide feedback for another revision")
    print("  [d] Discard — reject entirely")

    while True:
        choice = input("\n  Your choice (a/r/d): ").strip().lower()

        if choice == "a":
            return "approved"
        elif choice == "r":
            feedback = input("  What should be changed? > ")
            return f"revise:{feedback}"
        elif choice == "d":
            return "rejected"
        else:
            print("  Invalid choice. Enter a, r, or d.")


# ═══════════════════════════════════════════════
# SECTION 3: WORKFLOW IMPLEMENTATIONS
# ═══════════════════════════════════════════════
#
# Each workflow is a separate function.
# The router determines which one runs.
#
# Currently, meeting_prep is fully implemented.
# competitive_analysis and technology_evaluation
# are stubs you can build out later — they'd follow
# the same pattern with different skills and prompts.
#

async def workflow_meeting_prep(topic: str, client_name: str | None):
    """
    Full meeting prep workflow:
    Parallel Research → Synthesis → Eval → Revision loop → Human approval
    """

    all_agents = {
        "analyst": ANALYST,
        "evaluator": EVALUATOR,
    }
    mcp = {"consulting-research": research_server}

    output_filename = f"brief-{topic.lower().replace(' ', '-')[:50]}.md"
    output_path = f"output/{output_filename}"

    # ── PHASE 1: PARALLEL RESEARCH ─────────────
    # Pattern 2: Three researchers run simultaneously
    print("\n🔍 Phase 1: Parallel Research")
    print("   Fanning out to 3 researchers simultaneously...")

    research = await parallel_research(topic, client_name, mcp)

    print(f"   ✅ Industry:   {len(research['industry'])} chars")
    print(f"   ✅ Company:    {len(research['company'])} chars")
    print(f"   ✅ Technology:  {len(research['technology'])} chars")

    # Merge all research into one block for the analyst
    combined_research = (
        f"## Industry Research\n{research['industry'][:3000]}\n\n"
        f"## Company Research\n{research['company'][:3000]}\n\n"
        f"## Technology Research\n{research['technology'][:3000]}"
    )

    # ── PHASE 2: SYNTHESIS ─────────────────────
    print("\n📝 Phase 2: Synthesis")
    print("   Delegating to analyst subagent...")

    synthesis_prompt = f"""
    Use the analyst subagent to create a consulting meeting prep brief.

    ## Meeting context
    Topic: {topic}
    {"Client: " + client_name if client_name else ""}

    ## Research findings (from 3 parallel researchers)
    {combined_research}

    The analyst must:
    1. Follow the consulting-brief skill format EXACTLY
    2. Back every claim with the research above
    3. Save the brief to {output_path}
    """

    brief = await run_query(
        prompt=synthesis_prompt,
        agents=all_agents,
        max_turns=15,
    )

    print(f"   ✅ Brief drafted ({len(brief)} chars)")

    # ── PHASE 3: EVAL + REVISION LOOP ──────────
    # Pattern 4: Evaluator-Optimizer
    MAX_REVISIONS = 2
    revision_count = 0
    total = 0
    passed = False

    print("\n⚖️  Phase 3: Evaluation")

    eval_prompt = f"""
    Use the evaluator subagent to score this consulting brief.

    ## Brief to evaluate
    {brief[:6000]}

    Score using the eval-brief skill rubric.
    Return ONLY valid JSON.
    """

    eval_response = await run_query(
        prompt=eval_prompt,
        agents=all_agents,
        max_turns=10,
    )

    eval_json = extract_json(eval_response)
    if eval_json:
        try:
            eval_result = EvalResult.model_validate(eval_json)
            total = eval_result.total_score
            passed = eval_result.passed
            feedback = eval_result.feedback
            missing = eval_result.missing_elements
        except Exception as e:
            print(f"   ⚠️  Eval parse error: {e}")
            feedback, missing = "Parse error", []
    else:
        feedback, missing = "No JSON found", []

    print(f"   Score: {total}/30 — {'✅ PASS' if passed else '❌ FAIL'}")

    # Bounded revision loop
    while not passed and total > 0 and revision_count < MAX_REVISIONS:
        revision_count += 1
        print(f"\n🔄 Revision {revision_count}/{MAX_REVISIONS}")

        revision_prompt = f"""
        Use the analyst subagent to revise the consulting brief.

        ## Original brief
        {brief[:6000]}

        ## Feedback (score: {total}/30)
        {feedback}

        ## Missing elements
        {json.dumps(missing)}

        ONLY fix the weaknesses. Keep strong sections unchanged.
        Save to {output_path}
        """

        brief = await run_query(
            prompt=revision_prompt,
            agents=all_agents,
            max_turns=15,
        )

        # Re-evaluate
        eval_response = await run_query(
            prompt=f"Use the evaluator subagent to score this revised brief.\n\n{brief[:6000]}\n\nReturn ONLY JSON.",
            agents=all_agents,
            max_turns=10,
        )

        eval_json = extract_json(eval_response)
        if eval_json:
            try:
                eval_result = EvalResult.model_validate(eval_json)
                total = eval_result.total_score
                passed = eval_result.passed
                feedback = eval_result.feedback
                missing = eval_result.missing_elements
            except Exception:
                break
        else:
            break

        print(f"   Re-eval: {total}/30 — {'✅ PASS' if passed else '❌ FAIL'}")

    # ── PHASE 4: HUMAN APPROVAL ────────────────
    # Pattern 5: Human-in-the-Loop
    print("\n🔒 Phase 4: Human Approval")

    approval = await human_approval_gate(brief, total)

    if approval == "approved":
        print("   ✅ Approved — saving final brief")
        # Brief is already saved by the analyst subagent

    elif approval.startswith("revise:"):
        human_feedback = approval.split(":", 1)[1]
        print(f"   📝 Human requested changes: {human_feedback}")

        revision_prompt = f"""
        Use the analyst subagent to revise the brief based on
        human feedback.

        ## Current brief
        {brief[:6000]}

        ## Human feedback
        {human_feedback}

        Apply the requested changes. Save to {output_path}
        """

        brief = await run_query(
            prompt=revision_prompt,
            agents=all_agents,
            max_turns=15,
        )

        print("   ✅ Revised per human feedback — saving")

    elif approval == "rejected":
        print("   ❌ Rejected — discarding brief")
        return None, None

    # ── FINAL OUTPUT ───────────────────────────
    print("\n" + "═" * 55)
    print(f"  ✅ COMPLETE")
    print(f"  📋 Topic: {topic}")
    if client_name:
        print(f"  🏢 Client: {client_name}")
    print(f"  📊 Score: {total}/30")
    print(f"  🔄 Revisions: {revision_count}")
    print(f"  📁 Output: {output_path}")
    print("═" * 55)

    return brief, eval_result if eval_json else None


async def workflow_competitive_analysis(topic: str, client_name: str | None):
    """Competitive analysis workflow — stub for you to implement."""
    print("\n📊 Competitive Analysis workflow")
    print("   [Not yet implemented — uses same patterns with different skills]")
    print("   You'd create a competitive-analysis SKILL.md and a different")
    print("   set of prompts for the analyst subagent.")
    return None, None


async def workflow_technology_evaluation(topic: str, client_name: str | None):
    """Technology evaluation workflow — stub for you to implement."""
    print("\n🔬 Technology Evaluation workflow")
    print("   [Not yet implemented — uses same patterns with different skills]")
    print("   You'd create a tech-evaluation SKILL.md and a different")
    print("   set of prompts for the analyst subagent.")
    return None, None


# ═══════════════════════════════════════════════
# SECTION 4: MAIN ENTRY POINT
# ═══════════════════════════════════════════════
#
# The full flow:
# 1. ROUTER classifies the request
# 2. YOUR CODE branches to the right workflow
# 3. Workflow runs (with parallelization, eval, human gate)
#

async def main(user_input: str, client_name: str | None = None):
    """
    Main entry point — routes request to appropriate workflow.
    """

    print("═" * 55)
    print("  Consulting Intelligence Agent")
    print(f"  Request: {user_input[:80]}...")
    print("═" * 55)

    # ── STEP 1: ROUTE ──────────────────────────
    # Pattern 1: Router classifies the request
    print("\n🔀 Step 1: Routing")
    print("   Classifying request...")

    route = await classify_request(user_input)

    print(f"   Workflow:   {route.workflow}")
    print(f"   Confidence: {route.confidence:.0%}")
    print(f"   Reasoning:  {route.reasoning}")

    # Low confidence? Ask human to confirm
    # (Another human-in-the-loop checkpoint!)
    if route.confidence < 0.7:
        print(f"\n   ⚠️  Low confidence ({route.confidence:.0%})")
        confirm = input(f"   Proceed with '{route.workflow}'? (y/n): ")
        if confirm.lower() != "y":
            workflow_choice = input(
                "   Enter workflow (meeting_prep / competitive_analysis / technology_evaluation): "
            )
            route.workflow = workflow_choice.strip()

    # ── STEP 2: EXECUTE WORKFLOW ───────────────
    # YOUR CODE branches deterministically based on classification
    topic = user_input  # The full request becomes the topic

    if route.workflow == "meeting_prep":
        return await workflow_meeting_prep(topic, client_name)

    elif route.workflow == "competitive_analysis":
        return await workflow_competitive_analysis(topic, client_name)

    elif route.workflow == "technology_evaluation":
        return await workflow_technology_evaluation(topic, client_name)

    else:
        print(f"   ❌ Unknown workflow: {route.workflow}")
        print("   Defaulting to meeting_prep")
        return await workflow_meeting_prep(topic, client_name)


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    user_input = (
        sys.argv[1] if len(sys.argv) > 1
        else "Prepare me for a meeting about AI agents in healthcare"
    )
    client = sys.argv[2] if len(sys.argv) > 2 else None

    brief, eval_result = asyncio.run(main(user_input, client))
