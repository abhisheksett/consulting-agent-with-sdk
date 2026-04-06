"""
a2a_client.py — Lets your consulting agent call other A2A agents

This is the client side of A2A. Your consulting agent can now:
1. Discover any A2A-compliant agent by fetching its Agent Card
2. Understand what skills that agent has
3. Send it a task and get the result back

Real scenario this enables:
  Your consulting agent needs a compliance check.
  It discovers the bank's compliance agent via A2A,
  sends a task, gets back a structured compliance report —
  without knowing anything about how that agent was built.

Flow:
  A2AClient.from_url(agent_url)   → fetch Agent Card, understand capabilities
  client.send_task(text)          → POST tasks/send to remote agent
  client.wait_for_result(task_id) → poll until completed/failed
"""

import asyncio
import json
import uuid
import httpx
from pydantic import BaseModel


# ══════════════════════════════════════════════════
# AGENT CARD MODELS
# ══════════════════════════════════════════════════

class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = []
    inputModes: list[str] = []
    outputModes: list[str] = []


class AgentCard(BaseModel):
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    skills: list[AgentSkill] = []

    def find_skill(self, keyword: str) -> AgentSkill | None:
        """Find a skill by keyword in name, description, or tags."""
        keyword = keyword.lower()
        for skill in self.skills:
            if (keyword in skill.name.lower() or
                keyword in skill.description.lower() or
                    any(keyword in tag for tag in skill.tags)):
                return skill
        return None

    def describe(self) -> str:
        """Human-readable summary — useful for injecting into agent prompts."""
        skills_text = "\n".join([
            f"  - {s.name}: {s.description}"
            for s in self.skills
        ])
        return f"{self.name}\n{self.description}\nSkills:\n{skills_text}"


# ══════════════════════════════════════════════════
# A2A CLIENT
# ══════════════════════════════════════════════════

class A2AClient:
    """
    Client for calling any A2A-compliant remote agent.

    Usage:
        client = await A2AClient.from_url("https://legal-agent.bank.internal")
        task_id = await client.send_task("Review this contract for MiFID II compliance")
        result = await client.wait_for_result(task_id)
    """

    def __init__(self, agent_card: AgentCard, http_client: httpx.AsyncClient):
        self.card = agent_card
        self._http = http_client

    @classmethod
    async def from_url(cls, base_url: str) -> "A2AClient":
        """
        Discover an agent by fetching its Agent Card.

        This is the A2A equivalent of "reading the docs" —
        but it's automated and machine-readable.

        Steps:
        1. Fetch /.well-known/agent.json
        2. Parse into AgentCard model
        3. Return a ready-to-use client
        """
        base_url = base_url.rstrip("/")
        card_url = f"{base_url}/.well-known/agent.json"

        http_client = httpx.AsyncClient(timeout=30.0)

        response = await http_client.get(card_url)
        response.raise_for_status()

        card_data = response.json()
        # Ensure the url in the card points to the A2A endpoint
        if "url" not in card_data:
            card_data["url"] = f"{base_url}/a2a"

        card = AgentCard(**card_data)
        print(f"✅  Discovered agent: {card.name}")
        print(f"    Skills: {', '.join(s.name for s in card.skills)}")

        return cls(card, http_client)

    async def send_task(
        self,
        text: str,
        skill_id: str | None = None,
        callback_url: str | None = None,
    ) -> str:
        """
        Send a task to the remote agent.

        Returns the task ID immediately — agent runs in background.
        Use wait_for_result() or provide a callback_url for the result.

        Args:
            text: the task description / request
            skill_id: optional specific skill to invoke
            callback_url: optional webhook URL for push notification
        """
        task_payload = {
            "id": str(uuid.uuid4()),
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": text}]
            }
        }
        if callback_url:
            task_payload["callbackUrl"] = callback_url

        # JSON-RPC 2.0 envelope
        rpc_request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": task_payload,
            "id": str(uuid.uuid4()),
        }

        response = await self._http.post(
            self.card.url,
            json=rpc_request,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        rpc_response = response.json()
        if "error" in rpc_response:
            raise RuntimeError(f"A2A error: {rpc_response['error']}")

        task_id = rpc_response["result"]["id"]
        print(f"📤  Task submitted: {task_id}")
        return task_id

    async def get_task_status(self, task_id: str) -> dict:
        """Check current status of a task."""
        rpc_request = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": task_id},
            "id": str(uuid.uuid4()),
        }
        response = await self._http.post(
            self.card.url,
            json=rpc_request,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        rpc_response = response.json()

        if "error" in rpc_response:
            raise RuntimeError(f"A2A error: {rpc_response['error']}")

        return rpc_response["result"]

    async def wait_for_result(
        self,
        task_id: str,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> str:
        """
        Poll until the task completes or fails.

        In production you'd use webhooks (push) instead of polling.
        Polling is simpler to understand and works fine for learning.

        Args:
            task_id: returned from send_task()
            poll_interval: seconds between status checks
            timeout: max seconds to wait before raising TimeoutError
        """
        elapsed = 0.0

        while elapsed < timeout:
            status = await self.get_task_status(task_id)
            state = status["state"]

            print(f"  ⏳  Task {task_id[:8]}... state={state}")

            if state == "completed":
                # Extract text from response message parts
                message = status.get("message", {})
                parts = message.get("parts", [])
                result = " ".join(
                    p["text"] for p in parts
                    if p.get("type") == "text" and p.get("text")
                )
                return result

            elif state == "failed":
                raise RuntimeError(
                    f"Task failed: {status.get('error', 'unknown error')}")

            elif state == "canceled":
                raise RuntimeError("Task was canceled")

            elif state == "input-required":
                # Remote agent needs more information
                # In a full implementation you'd surface this to the user
                raise RuntimeError(
                    "Remote agent requires additional input — not handled in this demo")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(
            f"Task {task_id} did not complete within {timeout}s")

    async def close(self):
        await self._http.aclose()


# ══════════════════════════════════════════════════
# MULTI-AGENT ORCHESTRATION EXAMPLE
# This shows how your consulting agent would coordinate
# with multiple specialist agents via A2A
# ══════════════════════════════════════════════════

async def multi_agent_consulting_brief(
    topic: str,
    client_name: str | None = None,
    specialist_agents: list[str] | None = None,
) -> dict:
    """
    Orchestrate multiple A2A agents to produce a richer brief.

    Instead of one agent doing everything, each specialist
    contributes their domain expertise in parallel.

    Example specialist_agents:
        ["http://localhost:8001",   # financial data agent
         "http://localhost:8002",   # regulatory compliance agent
         "http://localhost:8003"]   # market intelligence agent

    This is the Multi-Agent RAG pattern you built, but now the
    "parallel researchers" are separate A2A agents running on
    different servers — possibly built by different teams.
    """
    if not specialist_agents:
        print("⚠️  No specialist agents provided — running solo.")
        return {"topic": topic, "note": "No A2A agents configured"}

    context = f"Topic: {topic}"
    if client_name:
        context += f", Client: {client_name}"

    # ── DISCOVER all agents in parallel ──────────
    print(f"\n🔍  Discovering {len(specialist_agents)} specialist agents...")
    clients: list[A2AClient] = []
    for url in specialist_agents:
        try:
            client = await A2AClient.from_url(url)
            clients.append(client)
        except Exception as e:
            print(f"  ⚠️  Could not reach {url}: {e}")

    if not clients:
        return {"error": "No specialist agents reachable"}

    # ── SEND tasks to all agents in parallel ─────
    # This is asyncio.gather — same parallelization pattern
    # you built in the SDK project, now across the network
    print(f"\n📤  Sending tasks to {len(clients)} agents in parallel...")

    task_prompt = f"Provide your specialist analysis for a consulting brief on: {context}"

    task_ids = await asyncio.gather(*[
        client.send_task(task_prompt)
        for client in clients
    ])

    # ── WAIT for all results in parallel ─────────
    print(f"\n⏳  Waiting for all agents to complete...")
    results = await asyncio.gather(*[
        client.wait_for_result(task_id)
        for client, task_id in zip(clients, task_ids)
    ], return_exceptions=True)

    # ── COLLECT findings ─────────────────────────
    findings = {}
    for client, result in zip(clients, results):
        if isinstance(result, Exception):
            findings[client.card.name] = f"Error: {result}"
        else:
            findings[client.card.name] = result

    # Cleanup
    for client in clients:
        await client.close()

    return {
        "topic": topic,
        "client": client_name,
        "specialist_findings": findings,
        "agents_consulted": [c.card.name for c in clients],
    }


# ══════════════════════════════════════════════════
# DEMO — calls your own agent as if it were remote
# ══════════════════════════════════════════════════

async def demo():
    """
    Demo: call your consulting agent via A2A.

    Prerequisites:
        Start the server first: uv run python src/a2a_server.py
        Then run this:          uv run python src/a2a_client.py
    """
    print("=" * 55)
    print("  A2A Client Demo")
    print("  Calling consulting agent via A2A protocol")
    print("=" * 55)

    # Step 1: Discover the agent
    print("\n1️⃣  Discovering agent...")
    client = await A2AClient.from_url("http://localhost:8000")

    print(f"\n📋  Agent capabilities:\n{client.card.describe()}")

    # Step 2: Find a relevant skill
    skill = client.card.find_skill("meeting prep")
    if skill:
        print(f"\n✅  Found skill: {skill.name}")

    # Step 3: Send a task
    print("\n2️⃣  Sending task...")
    task_id = await client.send_task(
        "Prepare a meeting prep brief on AI agents in financial services "
        "for our meeting with Goldman Sachs next week."
    )

    # Step 4: Wait for result
    print("\n3️⃣  Waiting for result...")
    result = await client.wait_for_result(task_id)

    print(f"\n{'='*55}")
    print("  Result:")
    print('='*55)
    print(result[:1000] + "..." if len(result) > 1000 else result)

    await client.close()


if __name__ == "__main__":
    asyncio.run(demo())
