"""
a2a_server.py — Exposes your consulting agent as an A2A remote agent

What this does:
  Any A2A-compliant client agent can now discover and call your
  consulting agent over HTTP — regardless of what framework they
  use (LangGraph, OpenAI Agents SDK, Google ADK, etc.)

Three endpoints:
  GET  /.well-known/agent.json  → returns the Agent Card
  POST /a2a                     → receives a Task, runs the agent, returns result
  GET  /a2a/tasks/{id}          → check status of a running task

A2A uses JSON-RPC 2.0 over HTTP. Each request has:
  - jsonrpc: "2.0"
  - method: "tasks/send" | "tasks/get" | "tasks/cancel"
  - params: the actual payload
  - id: request ID for correlation

Run with:
  uv run python src/a2a_server.py
  # Server starts at http://localhost:8000
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from claude_agent_sdk import query, ClaudeAgentOptions

# ── Import your existing agent logic ──────────────
# Reuse the workflows you already built — A2A is just
# a transport layer on top of your existing agent.
from agent import workflow_meeting_prep, workflow_competitive_analysis, workflow_technology_evaluation

# ── In-memory task store ───────────────────────────
# Production: use Redis or a database
# For learning: a simple dict is enough to understand the pattern
tasks: dict[str, dict] = {}

AGENT_CARD_PATH = Path(__file__).parent.parent / ".well-known" / "agent.json"


# ══════════════════════════════════════════════════
# PYDANTIC MODELS — A2A message shapes
# ══════════════════════════════════════════════════

class MessagePart(BaseModel):
    type: Literal["text", "file", "data"]
    text: str | None = None


class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: list[MessagePart]


class Task(BaseModel):
    """
    The core A2A unit of work.
    Client agent creates this and POSTs it to your /a2a endpoint.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: Message
    # Optional: client supplies a webhook URL for push notifications
    # Your agent calls this URL when the task completes
    callbackUrl: str | None = None


class TaskStatus(BaseModel):
    """
    Current state of a task.
    Client can poll GET /a2a/tasks/{id} to check progress.
    """
    id: str
    state: Literal["submitted", "working",
                   "input-required", "completed", "failed", "canceled"]
    message: Message | None = None   # populated when state = "completed"
    error: str | None = None         # populated when state = "failed"
    createdAt: str
    updatedAt: str


class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    method: str
    params: dict
    id: str | int


class JsonRpcResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    result: dict | None = None
    error: dict | None = None
    id: str | int


# ══════════════════════════════════════════════════
# SKILL ROUTER
# Maps incoming task text → the right workflow
# ══════════════════════════════════════════════════

async def run_skill(task_text: str) -> str:
    """
    Route the task to the right workflow based on content.

    In a production system you'd use the skill ID from the task params.
    For learning purposes, we use simple keyword routing here.
    """
    text_lower = task_text.lower()

    if any(kw in text_lower for kw in ["meeting prep", "brief", "prepare for", "meeting with"]):
        # Extract topic from the message — simple heuristic
        result = await workflow_meeting_prep(task_text)
        return result.get("brief", "Brief generation failed.")

    elif any(kw in text_lower for kw in ["competitive", "competition", "competitors", "market"]):
        result = await workflow_competitive_analysis(task_text)
        return result.get("analysis", "Analysis failed.")

    elif any(kw in text_lower for kw in ["evaluate", "technology", "tech eval", "assess"]):
        result = await workflow_technology_evaluation(task_text)
        return result.get("evaluation", "Evaluation failed.")

    else:
        # Default: treat as meeting prep
        result = await workflow_meeting_prep(task_text)
        return result.get("brief", "Could not determine skill. Defaulted to meeting prep.")


# ══════════════════════════════════════════════════
# TASK HANDLERS
# ══════════════════════════════════════════════════

async def handle_tasks_send(params: dict, request_id: str | int) -> JsonRpcResponse:
    """
    Handle tasks/send — the main A2A method.

    Flow:
    1. Parse the incoming Task
    2. Store it with state = "submitted"
    3. Run the agent asynchronously
    4. Update task state to "working" → "completed" / "failed"
    5. If callbackUrl provided, POST result to webhook
    6. Return task ID immediately (client can poll or wait for webhook)
    """
    try:
        task = Task(**params)
    except Exception as e:
        return JsonRpcResponse(
            error={"code": -32602, "message": f"Invalid task params: {e}"},
            id=request_id,
        )

    now = datetime.now(timezone.utc).isoformat()

    # Store initial task state
    tasks[task.id] = {
        "id": task.id,
        "state": "submitted",
        "message": None,
        "error": None,
        "createdAt": now,
        "updatedAt": now,
    }

    # Extract text from message parts
    task_text = " ".join(
        part.text for part in task.message.parts
        if part.type == "text" and part.text
    )

    # Run agent in background — don't block the HTTP response
    asyncio.create_task(
        _execute_task(task.id, task_text, task.callbackUrl)
    )

    # Return immediately with task ID
    # Client either polls GET /a2a/tasks/{id} or waits for webhook callback
    return JsonRpcResponse(
        result={"id": task.id, "state": "submitted"},
        id=request_id,
    )


async def _execute_task(task_id: str, task_text: str, callback_url: str | None):
    """
    Background execution of the agent skill.
    Updates task state as work progresses.
    """
    # Mark as working
    tasks[task_id]["state"] = "working"
    tasks[task_id]["updatedAt"] = datetime.now(timezone.utc).isoformat()

    try:
        result_text = await run_skill(task_text)

        # Mark as completed with result
        tasks[task_id]["state"] = "completed"
        tasks[task_id]["message"] = {
            "role": "agent",
            "parts": [{"type": "text", "text": result_text}]
        }
        tasks[task_id]["updatedAt"] = datetime.now(timezone.utc).isoformat()

        # Push notification: if client gave us a webhook URL, call it
        if callback_url:
            await _send_callback(callback_url, tasks[task_id])

    except Exception as e:
        tasks[task_id]["state"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["updatedAt"] = datetime.now(timezone.utc).isoformat()

        if callback_url:
            await _send_callback(callback_url, tasks[task_id])


async def _send_callback(callback_url: str, task_data: dict):
    """
    POST task result to client's webhook URL.
    Client registered this URL when sending the task.
    This is A2A's push notification mechanism.
    """
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                callback_url,
                json={"task": task_data},
                timeout=10.0,
            )
    except Exception as e:
        print(f"⚠️  Callback to {callback_url} failed: {e}")


async def handle_tasks_get(params: dict, request_id: str | int) -> JsonRpcResponse:
    """
    Handle tasks/get — client polls for task status.
    """
    task_id = params.get("id")
    if not task_id or task_id not in tasks:
        return JsonRpcResponse(
            error={"code": -32602, "message": f"Task {task_id} not found"},
            id=request_id,
        )
    return JsonRpcResponse(result=tasks[task_id], id=request_id)


async def handle_tasks_cancel(params: dict, request_id: str | int) -> JsonRpcResponse:
    """
    Handle tasks/cancel — client requests cancellation.
    Note: cancellation is best-effort in this implementation.
    """
    task_id = params.get("id")
    if task_id in tasks and tasks[task_id]["state"] in ("submitted", "working"):
        tasks[task_id]["state"] = "canceled"
        tasks[task_id]["updatedAt"] = datetime.now(timezone.utc).isoformat()
        return JsonRpcResponse(result={"id": task_id, "state": "canceled"}, id=request_id)
    return JsonRpcResponse(
        error={"code": -32602, "message": f"Task {task_id} cannot be canceled"},
        id=request_id,
    )


# ══════════════════════════════════════════════════
# HTTP REQUEST HANDLER
# Simple HTTP server without requiring FastAPI
# ══════════════════════════════════════════════════

METHOD_HANDLERS = {
    "tasks/send":   handle_tasks_send,
    "tasks/get":    handle_tasks_get,
    "tasks/cancel": handle_tasks_cancel,
}


async def handle_request(method: str, path: str, body: bytes) -> tuple[int, dict]:
    """
    Route HTTP requests to the right handler.

    GET  /.well-known/agent.json → serve Agent Card
    POST /a2a                    → dispatch JSON-RPC method
    GET  /a2a/tasks/{id}         → shorthand task status check
    """

    # ── Agent Card endpoint ──
    if method == "GET" and path == "/.well-known/agent.json":
        card = json.loads(AGENT_CARD_PATH.read_text())
        return 200, card

    # ── Shorthand task status ──
    if method == "GET" and path.startswith("/a2a/tasks/"):
        task_id = path.split("/a2a/tasks/")[1]
        if task_id in tasks:
            return 200, tasks[task_id]
        return 404, {"error": f"Task {task_id} not found"}

    # ── Main A2A JSON-RPC endpoint ──
    if method == "POST" and path == "/a2a":
        try:
            rpc_request = JsonRpcRequest(**json.loads(body))
        except Exception as e:
            return 400, {"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {e}"}, "id": None}

        handler = METHOD_HANDLERS.get(rpc_request.method)
        if not handler:
            response = JsonRpcResponse(
                error={"code": -32601,
                       "message": f"Method '{rpc_request.method}' not found"},
                id=rpc_request.id,
            )
            return 200, response.model_dump()

        response = await handler(rpc_request.params, rpc_request.id)
        return 200, response.model_dump()

    return 404, {"error": "Not found"}


# ══════════════════════════════════════════════════
# MINIMAL HTTP SERVER
# ══════════════════════════════════════════════════

async def run_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Minimal asyncio HTTP server — no FastAPI dependency needed.
    In production you'd use FastAPI or Starlette for middleware,
    auth validation, rate limiting, etc.
    """

    async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            # Read HTTP request (simplified — handles common cases)
            raw = await reader.read(65536)
            if not raw:
                return

            lines = raw.split(b"\r\n")
            request_line = lines[0].decode()
            method, path, _ = request_line.split(" ", 2)

            # Find body (after blank line)
            body = b""
            if b"\r\n\r\n" in raw:
                body = raw.split(b"\r\n\r\n", 1)[1]

            status, response_body = await handle_request(method, path, body)

            response_json = json.dumps(response_body).encode()
            response = (
                f"HTTP/1.1 {status} {'OK' if status == 200 else 'Error'}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(response_json)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            ).encode() + response_json

            writer.write(response)
            await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(handle_connection, host, port)
    print(f"🚀  A2A Server running at http://{host}:{port}")
    print(f"📋  Agent Card: http://localhost:{port}/.well-known/agent.json")
    print(f"🔌  A2A Endpoint: http://localhost:{port}/a2a")
    print(f"\nPress Ctrl+C to stop\n")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(run_server())
