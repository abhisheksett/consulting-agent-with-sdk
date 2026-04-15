"""
langgraph_agent.py

Your consulting agent rebuilt as a LangGraph.
Same logic as agent.py — different execution model.

Compare:
  agent.py (SDK):     Python code drives the loop explicitly
  This file (Graph):  Graph structure drives the loop, nodes are functions

The key difference you'll feel:
  SDK:       you write: "call researcher, then analyst, then evaluator, if score < 20 retry"
  LangGraph: you declare: "these are my nodes, these are my edges, compile and run"
"""

import asyncio
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# ── MODEL ──────────────────────────────────────────
# Uses OpenAI — make sure OPENAI_API_KEY is set in .env
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = TavilySearchResults(max_results=5)

MAX_REVISIONS = 2


# ══════════════════════════════════════════════════
# STATE
# The single shared dict that flows through all nodes.
# Every node reads from this, writes back changed fields only.
# ══════════════════════════════════════════════════

class ConsultingState(TypedDict):
    # Input — set once, never changes
    topic: str
    client_name: str | None

    # Set by researcher node
    research: str

    # Set by analyst node
    brief: str

    # Set by evaluator node
    eval_score: int        # 0-30
    eval_feedback: str

    # Tracks loop iterations
    revision_count: int

    # Conversation history (append-only via add_messages reducer)
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════
# NODES
# Plain async functions. Take full state, return only
# the fields they changed. Other fields pass through.
# ══════════════════════════════════════════════════

async def researcher_node(state: ConsultingState) -> dict:
    """
    Node 1: Research
    Searches for current information about the topic.
    Writes to: state["research"]
    """
    print(f"🔍  Researching: {state['topic']}")

    query = f"{state['topic']} {state.get('client_name', '')} industry trends 2026"
    search_results = await search_tool.ainvoke(query)

    # Condense results into structured findings
    condensed = llm.invoke([
        SystemMessage(content="You are a research analyst. Condense these search results into "
                      "structured findings: key trends, key players, recent developments. "
                      "Be specific. Include facts and figures where available."
                      ),
        HumanMessage(
            content=f"Topic: {query}\n\nResults: {str(search_results)}")
    ])

    return {"research": condensed.content}


async def analyst_node(state: ConsultingState) -> dict:
    """
    Node 2: Analysis → Brief
    Takes research findings and generates a consulting brief.
    Writes to: state["brief"]

    On revision runs, also receives state["eval_feedback"] —
    the evaluator's critique from the previous iteration.
    Previous feedback is naturally available because it's in state.
    """
    print(f"✍️   Writing brief (revision {state['revision_count'] + 1})")

    user_prompt = f"""
        Topic: {state['topic']}
        {"Client: " + state['client_name'] if state.get('client_name') else ""}

        Research Findings:
        {state['research']}
    """

    if state.get('eval_feedback') and state['revision_count'] > 0:
        user_prompt += f"""
            Previous Brief Feedback (address these issues in your revision):
            {state['eval_feedback']}
        """

    response = llm.invoke([
        SystemMessage(content="You are a senior consulting analyst. Write a meeting prep brief with: "
                      "1) Executive Summary (2-3 sentences) "
                      "2) Key Industry Trends (3-5 bullet points) "
                      "3) Talking Points (3-5 specific, insight-driven points) "
                      "4) Risk Factors (2-3 items) "
                      "5) Recommended Questions (5 questions for the meeting) "
                      "Be specific. Use data from the research. Avoid generic statements."
                      ),
        HumanMessage(content=user_prompt)
    ])

    return {"brief": response.content}


async def evaluator_node(state: ConsultingState) -> dict:
    """
    Node 3: Evaluation
    Scores the brief and provides feedback.
    Writes to: state["eval_score"], state["eval_feedback"], state["revision_count"]

    The conditional edge AFTER this node reads eval_score to decide:
    route back to analyst (revise) or go to END (done)
    """
    print(f"⚖️   Evaluating brief...")

    response = llm.invoke([
        SystemMessage(content="You are a senior consulting partner reviewing a meeting prep brief. "
                      "Score it out of 30 across: "
                      "Research depth (0-10), Insight quality (0-10), Actionability (0-10). "
                      "Respond in this exact format:\n"
                      "SCORE: <total>/30\n"
                      "FEEDBACK: <specific critique, max 3 sentences, focus on what to improve>"
                      ),
        HumanMessage(content=f"Brief to evaluate:\n\n{state['brief']}")
    ])

    content = response.content
    score = 20  # default if parsing fails
    feedback = content

    if "SCORE:" in content:
        try:
            score_line = [l for l in content.split('\n') if 'SCORE:' in l][0]
            score = int(score_line.split('SCORE:')[1].split('/')[0].strip())
        except (ValueError, IndexError):
            pass

    if "FEEDBACK:" in content:
        feedback = content.split("FEEDBACK:")[1].strip()

    print(f"   Score: {score}/30")

    return {
        "eval_score": score,
        "eval_feedback": feedback,
        "revision_count": state["revision_count"] + 1,
    }


# ══════════════════════════════════════════════════
# ROUTING FUNCTION
# Returns the name of the next node as a string.
# LangGraph uses the return value to route.
# ══════════════════════════════════════════════════

def route_after_eval(state: ConsultingState) -> str:
    if state["eval_score"] < 20 and state["revision_count"] < MAX_REVISIONS:
        print(f"   Score {state['eval_score']}/30 < 20 — requesting revision")
        return "analyst"
    else:
        print(f"   Score {state['eval_score']}/30 — accepting brief")
        return END


# ══════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════

def build_graph():
    """
    Build and compile the consulting agent graph.

    Visual representation:
                    ┌─────────────┐
                    │  researcher │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
              ┌────►│   analyst   │
              │     └──────┬──────┘
              │            │
              │     ┌──────▼──────┐
              │     │  evaluator  │
              │     └──────┬──────┘
              │            │
              │     route_after_eval()
              │      ├── score < 20 AND revisions < 2 → analyst (loop)
              └──────┘   └── otherwise → END
    """
    graph = StateGraph(ConsultingState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst",    analyst_node)
    graph.add_node("evaluator",  evaluator_node)

    graph.set_entry_point("researcher")

    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst",    "evaluator")

    graph.add_conditional_edges(
        "evaluator",
        route_after_eval,
        {
            "analyst": "analyst",
            END: END,
        }
    )

    return graph.compile()


# ══════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════

async def run(topic: str, client_name: str | None = None):
    app = build_graph()

    initial_state = {
        "topic": topic,
        "client_name": client_name,
        "research": "",
        "brief": "",
        "eval_score": 0,
        "eval_feedback": "",
        "revision_count": 0,
        "messages": [],
    }

    print(f"\n{'='*55}")
    print(f"  Consulting Agent (LangGraph + OpenAI)")
    print(f"  Topic: {topic}")
    if client_name:
        print(f"  Client: {client_name}")
    print(f"{'='*55}\n")

    final_state = await app.ainvoke(initial_state)

    print(f"\n{'='*55}")
    print(f"  Final Score: {final_state['eval_score']}/30")
    print(f"  Revisions:   {final_state['revision_count']}")
    print(f"{'='*55}")
    print(f"\n{final_state['brief']}")

    return final_state


if __name__ == "__main__":
    asyncio.run(run(
        topic="AI agents in financial services",
        client_name="Goldman Sachs"
    ))
