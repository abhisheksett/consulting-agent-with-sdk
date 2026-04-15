"""
crewai_consulting_agent.py

Same consulting agent — built with CrewAI instead of LangGraph.

Compare the two files side by side:
  LangGraph: StateGraph, nodes, edges, routing functions, compile()
  CrewAI:    Agent, Task, Crew, kickoff()

CrewAI feels like describing a team.
LangGraph feels like drawing a flowchart.

Notice what disappears in CrewAI:
  - No explicit state management
  - No routing functions
  - No graph compilation
  - No edge definitions

Notice what you give up:
  - Less control over the exact execution path
  - Harder to implement conditional loops (eval → revise)
  - Less visibility into intermediate state
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# ── MODEL + TOOLS ──────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini")
tavily = TavilySearchResults(max_results=5)


# ── WRAP TOOL FOR CREWAI ───────────────────────────
# CrewAI has its own tool decorator
@tool("Web Search")
def web_search(query: str) -> str:
    """Search the web for current information about a topic."""
    results = tavily.invoke(query)
    return str(results)


# ══════════════════════════════════════════════════
# AGENTS
# Each agent has:
#   role:      job title (used in prompts automatically)
#   goal:      what this agent optimizes for
#   backstory: context that shapes their personality and approach
#   tools:     what they can use
#   llm:       which model to use (can differ per agent)
#
# The backstory is more than flavour — it genuinely affects
# how the agent approaches its work. A "10 years experience"
# backstory produces different output than "junior analyst".
# ══════════════════════════════════════════════════

researcher = Agent(
    role="Senior Research Analyst",
    goal=(
        "Find comprehensive, current intelligence about the topic. "
        "Focus on recent developments, key players, and quantified trends."
    ),
    backstory=(
        "You've spent 10 years researching enterprise technology and industry trends "
        "for top consulting firms. You're known for finding the specific facts and "
        "figures that make a brief credible — not just general observations. "
        "You always search for multiple angles before concluding."
    ),
    tools=[web_search],
    llm=llm,
    verbose=True,          # print agent reasoning to console
    allow_delegation=False,  # this agent doesn't delegate to others
)

analyst = Agent(
    role="Senior Consulting Analyst",
    goal=(
        "Transform raw research into a crisp, insight-driven consulting brief "
        "that a partner can use to lead a client meeting."
    ),
    backstory=(
        "You've written hundreds of client briefs at top-tier consulting firms. "
        "You know that clients don't want summaries — they want insights. "
        "You always structure briefs with an executive summary, key trends, "
        "specific talking points, and questions that show you understand their business."
    ),
    tools=[],     # analyst doesn't need tools — works from context
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

evaluator = Agent(
    role="Senior Consulting Partner",
    goal=(
        "Ensure the brief meets partner-level quality standards. "
        "Score objectively and give specific, actionable feedback."
    ),
    backstory=(
        "You're a managing partner who has reviewed thousands of client briefs. "
        "You have zero tolerance for generic statements and buzzwords. "
        "You score briefs on research depth, insight quality, and actionability. "
        "Your feedback is direct and specific — you tell analysts exactly what to fix."
    ),
    tools=[],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ══════════════════════════════════════════════════
# TASKS
# Each task has:
#   description:     what to do (supports {variable} templating)
#   expected_output: what the output should look like
#   agent:           who does this task
#   context:         list of tasks whose output this task needs
#                    (CrewAI passes prior task outputs automatically)
# ══════════════════════════════════════════════════

def build_tasks(topic: str, client_name: str | None = None) -> list[Task]:
    client_context = f" for our meeting with {client_name}" if client_name else ""

    research_task = Task(
        description=(
            f"Research the current state of {topic}{client_context}. "
            f"Search for: recent industry developments (last 6 months), "
            f"key players and their positions, technology trends, "
            f"challenges enterprises are facing. "
            f"{'Pay particular attention to ' + client_name + ' specifically.' if client_name else ''}"
        ),
        expected_output=(
            "Structured research findings with: "
            "1) Key trends with specific facts/figures "
            "2) Key players and their current positioning "
            "3) Recent developments (last 6 months) "
            "4) Challenges and opportunities "
            "Include source context where relevant."
        ),
        agent=researcher,
    )

    brief_task = Task(
        description=(
            f"Using the research findings, write a consulting meeting prep brief "
            f"on {topic}{client_context}. "
            f"The brief is for a senior partner who needs to lead a 60-minute "
            f"client meeting. Make every point specific and insight-driven. "
            f"No generic statements."
        ),
        expected_output=(
            "A complete meeting prep brief with: "
            "1) Executive Summary (2-3 sentences, most important insight first) "
            "2) Key Industry Trends (3-5 bullets with specific data points) "
            "3) Talking Points (3-5 insight-driven points, not generic observations) "
            "4) Risk Factors (2-3 specific risks relevant to this client) "
            "5) Recommended Questions (5 questions that demonstrate expertise)"
        ),
        agent=analyst,
        context=[research_task],   # analyst receives researcher's output
    )

    eval_task = Task(
        description=(
            "Review the consulting brief and evaluate its quality. "
            "Score it across three dimensions (0-10 each): "
            "Research Depth, Insight Quality, Actionability. "
            "Be a tough but fair critic. Identify the 2-3 most important improvements."
        ),
        expected_output=(
            "Evaluation in this format:\n"
            "TOTAL SCORE: X/30\n"
            "Research Depth: X/10 — [one sentence rationale]\n"
            "Insight Quality: X/10 — [one sentence rationale]\n"
            "Actionability: X/10 — [one sentence rationale]\n"
            "TOP IMPROVEMENTS:\n"
            "1. [specific, actionable improvement]\n"
            "2. [specific, actionable improvement]\n"
            "3. [specific, actionable improvement]"
        ),
        agent=evaluator,
        context=[research_task, brief_task],  # evaluator sees both
    )

    return [research_task, brief_task, eval_task]


# ══════════════════════════════════════════════════
# CREW
# Assembles agents + tasks + process into a runnable team.
#
# Process.sequential: tasks run in order, each gets prior outputs
# Process.hierarchical: a manager agent coordinates the team
#                       (CrewAI creates the manager automatically)
# ══════════════════════════════════════════════════

def build_crew(topic: str, client_name: str | None = None) -> Crew:
    tasks = build_tasks(topic, client_name)

    return Crew(
        agents=[researcher, analyst, evaluator],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,          # print crew coordination to console
        memory=True,           # agents share memory across tasks
        # manager_llm=llm,     # uncomment for Process.hierarchical
    )


# ══════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════

def run(topic: str, client_name: str | None = None):
    print(f"\n{'='*55}")
    print(f"  Consulting Agent (CrewAI)")
    print(f"  Topic: {topic}")
    if client_name:
        print(f"  Client: {client_name}")
    print(f"{'='*55}\n")

    crew = build_crew(topic, client_name)

    # kickoff() runs the crew synchronously
    # kickoff_async() for async contexts
    result = crew.kickoff()

    print(f"\n{'='*55}")
    print(f"  Final Output:")
    print(f"{'='*55}")
    print(result.raw)

    return result


if __name__ == "__main__":
    run(
        topic="AI agents in financial services",
        client_name="Goldman Sachs"
    )
