import asyncio
import json
import os

from claude_agent_sdk import tool, create_sdk_mcp_server
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()

# Initialize Tavily client (synchronous — we'll wrap calls with to_thread)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ──────────────────────────────────────────────
# TOOL 1: Industry Research
# ──────────────────────────────────────────────
@tool(
    "search_industry",
    "Search the web for industry trends, market analysis, and sector "
    "insights relevant to consulting engagements. Use this for broad "
    "industry-level research.",
    {"query": str, "num_results": int},
)
async def search_industry(args: dict) -> dict:
    query = args["query"]
    num_results = args.get("num_results", 5)

    # Wrap sync Tavily call to avoid blocking the async event loop
    results = await asyncio.to_thread(
        tavily.search, query, max_results=num_results
    )

    return {
        "content": [
            {"type": "text", "text": json.dumps(results, indent=2)}
        ]
    }


# ──────────────────────────────────────────────
# TOOL 2: Company Research
# ──────────────────────────────────────────────
@tool(
    "research_company",
    "Research a specific company — recent news, strategic initiatives, "
    "financial performance, competitive position, and key challenges. "
    "Use this when preparing for a meeting with a specific client.",
    {"company_name": str},
)
async def research_company(args: dict) -> dict:
    company = args["company_name"]

    results = await asyncio.to_thread(
        tavily.search,
        f"{company} company news strategy challenges 2026",
        max_results=5,
    )

    return {
        "content": [
            {"type": "text", "text": json.dumps(results, indent=2)}
        ]
    }


# ──────────────────────────────────────────────
# TOOL 3: Technology Trends
# ──────────────────────────────────────────────
@tool(
    "get_tech_trends",
    "Get current technology and AI trends for a specific domain. "
    "Examples: 'AI agents', 'cloud migration', 'data platforms', "
    "'GenAI in healthcare'. Use for the technology relevance section.",
    {"domain": str},
)
async def get_tech_trends(args: dict) -> dict:
    domain = args["domain"]

    results = await asyncio.to_thread(
        tavily.search,
        f"{domain} enterprise technology trends adoption 2026",
        max_results=5,
    )

    return {
        "content": [
            {"type": "text", "text": json.dumps(results, indent=2)}
        ]
    }


# ──────────────────────────────────────────────
# PACKAGE INTO AN IN-PROCESS MCP SERVER
# ──────────────────────────────────────────────
# This is what you pass to ClaudeAgentOptions(mcp_servers={...})
# The SDK registers these tools so Claude can discover and call them.

research_server = create_sdk_mcp_server(
    name="consulting-research",
    version="1.0.0",
    tools=[search_industry, research_company, get_tech_trends],
)
