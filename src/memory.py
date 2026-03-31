"""
Memory system for the Consulting Intelligence Agent.

THREE TYPES OF MEMORY:

1. EPISODIC — "What happened before"
   - Records each run: topic, score, feedback, what worked/failed
   - Stored as JSONL (one JSON object per line, append-only)
   - Retrieved by topic similarity (keyword match for now, vector search in production)
   - Injected into the analyst prompt so it learns from past mistakes

2. SEMANTIC — "What I know"
   - Facts and preferences learned across all runs
   - Stored as JSON (a growing knowledge base)
   - Categories: preferences, domain_knowledge, tool_insights, quality_patterns
   - Injected into system prompts to shape overall behavior

3. SESSION — "Continue where I left off"
   - Tracks session IDs for Agent SDK resume capability
   - Stores the last session state so you can pick up mid-workflow
   - Useful for long-running tasks that span multiple invocations

STORAGE:
All memory is file-based (JSON/JSONL in a memory/ directory).
In production, you'd replace with:
  - Episodic → Postgres or SQLite with full-text search
  - Semantic → Vector DB (Mem0, Zep, Pinecone) for semantic retrieval
  - Session → Redis or the SDK's built-in session persistence

The file-based approach works for learning and prototyping.
The INTERFACE stays the same regardless of storage backend.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from models import Episode, SemanticFact


# ═══════════════════════════════════════════════
# MEMORY DIRECTORY SETUP
# ═══════════════════════════════════════════════

MEMORY_DIR = Path("memory")
EPISODES_FILE = MEMORY_DIR / "episodes.jsonl"
SEMANTIC_FILE = MEMORY_DIR / "semantic.json"
SESSIONS_FILE = MEMORY_DIR / "sessions.json"


def ensure_memory_dir():
    """Create memory directory and files if they don't exist."""
    MEMORY_DIR.mkdir(exist_ok=True)

    if not EPISODES_FILE.exists():
        EPISODES_FILE.touch()

    if not SEMANTIC_FILE.exists():
        SEMANTIC_FILE.write_text("[]")

    if not SESSIONS_FILE.exists():
        SESSIONS_FILE.write_text("{}")


# ═══════════════════════════════════════════════
# EPISODIC MEMORY
# ═══════════════════════════════════════════════
#
# "What happened before"
#
# After each run, we save an episode recording:
#   - what topic was researched
#   - what score the evaluator gave
#   - what feedback was provided
#   - what worked and what didn't
#
# Before the next run, we retrieve relevant episodes
# and inject them into the analyst's prompt so it
# can learn from past mistakes.
#
# Storage: JSONL (one JSON per line, append-only)
# Why JSONL: easy to append, easy to read line-by-line,
# no need to parse/rewrite the whole file on each write.
#

def save_episode(episode: Episode) -> None:
    """Append an episode to the episodic memory file."""
    ensure_memory_dir()
    with open(EPISODES_FILE, "a") as f:
        f.write(episode.model_dump_json() + "\n")


def load_all_episodes() -> list[Episode]:
    """Load all episodes from memory."""
    ensure_memory_dir()
    episodes = []
    with open(EPISODES_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(Episode.model_validate_json(line))
                except Exception:
                    continue  # skip malformed lines
    return episodes


def find_relevant_episodes(
    topic: str,
    client: str | None = None,
    max_results: int = 5,
) -> list[Episode]:
    """
    Find past episodes relevant to the current task.

    Current approach: keyword matching on topic and client.
    Production approach: vector similarity search using embeddings.

    This is intentionally simple — the INTERFACE is what matters.
    Swapping keyword match for vector search later doesn't change
    any calling code.
    """
    all_episodes = load_all_episodes()

    if not all_episodes:
        return []

    # Simple relevance scoring: count keyword overlaps
    topic_words = set(topic.lower().split())
    scored = []

    for ep in all_episodes:
        ep_words = set(ep.topic.lower().split())
        # Score: keyword overlap + bonus for same client
        score = len(topic_words & ep_words)
        if client and ep.client and client.lower() in ep.client.lower():
            score += 3  # big bonus for same client
        if ep.workflow == "meeting_prep":
            score += 1  # slight bonus for same workflow type
        scored.append((score, ep))

    # Sort by relevance score (descending), then recency
    scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

    return [ep for _, ep in scored[:max_results]]


def format_episodes_for_prompt(episodes: list[Episode]) -> str:
    """Format episodes into a prompt-friendly string."""
    if not episodes:
        return "No relevant past episodes found."

    lines = ["## Lessons from past briefs:"]
    for ep in episodes:
        lines.append(ep.to_summary())

    return "\n".join(lines)


# ═══════════════════════════════════════════════
# SEMANTIC MEMORY
# ═══════════════════════════════════════════════
#
# "What I know"
#
# Accumulated facts and preferences that apply across
# all runs. Not about specific events (that's episodic),
# but about general truths:
#   - "Healthcare clients are sensitive about HIPAA"
#   - "Talking points score poorly when they're generic"
#   - "Tavily returns better results with the year in the query"
#
# Storage: JSON array of SemanticFact objects
# In production: vector DB for semantic retrieval
#

def load_semantic_memory() -> list[SemanticFact]:
    """Load all semantic facts."""
    ensure_memory_dir()
    try:
        data = json.loads(SEMANTIC_FILE.read_text())
        return [SemanticFact.model_validate(item) for item in data]
    except Exception:
        return []


def save_semantic_memory(facts: list[SemanticFact]) -> None:
    """Save all semantic facts (full rewrite)."""
    ensure_memory_dir()
    SEMANTIC_FILE.write_text(
        json.dumps([f.model_dump() for f in facts], indent=2)
    )


def add_semantic_fact(fact: SemanticFact) -> None:
    """Add a new semantic fact, avoiding duplicates."""
    facts = load_semantic_memory()

    # Check for near-duplicates (simple string comparison)
    for existing in facts:
        if existing.fact.lower().strip() == fact.fact.lower().strip():
            # Update confidence and last_used instead of duplicating
            existing.confidence = max(existing.confidence, fact.confidence)
            existing.last_used = fact.created_at
            save_semantic_memory(facts)
            return

    facts.append(fact)
    save_semantic_memory(facts)


def get_semantic_facts_for_prompt(
    category: str | None = None,
    max_facts: int = 10,
) -> str:
    """
    Get semantic facts formatted for prompt injection.

    Args:
        category: Filter by category, or None for all
        max_facts: Maximum facts to include (context budget)
    """
    facts = load_semantic_memory()

    if category:
        facts = [f for f in facts if f.category == category]

    # Sort by confidence (highest first)
    facts.sort(key=lambda f: f.confidence, reverse=True)
    facts = facts[:max_facts]

    if not facts:
        return ""

    lines = ["## Known facts and preferences:"]
    for f in facts:
        lines.append(
            f"- [{f.category}] {f.fact} (confidence: {f.confidence:.0%})")

    return "\n".join(lines)


def extract_learnings_from_episode(episode: Episode) -> list[SemanticFact]:
    """
    Extract semantic facts from an episode.

    This is a simple rule-based extraction. In production,
    you'd use an LLM to analyze the episode and extract
    structured learnings. For example:

        prompt = f"Given this agent run: {episode}
                   What general facts or preferences should
                   the agent remember for future runs?"

    For now, we use heuristics.
    """
    now = datetime.now(timezone.utc).isoformat()
    learnings = []

    # Pattern: low scores indicate areas to improve
    if episode.eval_score < 18:
        learnings.append(SemanticFact(
            fact=f"Briefs about '{episode.topic}' tend to need extra attention — past score was {episode.eval_score}/30",
            source=f"episode:{episode.timestamp}",
            confidence=0.7,
            category="quality_pattern",
            created_at=now,
        ))

    # Pattern: high revision count = systematic issue
    if episode.revisions_needed >= 2:
        learnings.append(SemanticFact(
            fact=f"Topic '{episode.topic}' required {episode.revisions_needed} revisions. Common issue: {episode.what_failed}",
            source=f"episode:{episode.timestamp}",
            confidence=0.8,
            category="quality_pattern",
            created_at=now,
        ))

    # Pattern: extract what worked as a positive reinforcement
    if episode.eval_score >= 24 and episode.what_worked:
        learnings.append(SemanticFact(
            fact=f"High-scoring approach: {episode.what_worked}",
            source=f"episode:{episode.timestamp}",
            confidence=0.9,
            category="quality_pattern",
            created_at=now,
        ))

    # Pattern: extract evaluator feedback themes
    feedback_lower = episode.evaluator_feedback.lower()
    if "talking points" in feedback_lower:
        learnings.append(SemanticFact(
            fact="Talking points are a recurring weak area — make them specific and data-backed, never generic",
            source=f"episode:{episode.timestamp}",
            confidence=0.85,
            category="quality_pattern",
            created_at=now,
        ))

    if "risk" in feedback_lower or "sensitivity" in feedback_lower:
        learnings.append(SemanticFact(
            fact="Risk/sensitivity sections need more depth — always include regulatory and reputational risks",
            source=f"episode:{episode.timestamp}",
            confidence=0.85,
            category="quality_pattern",
            created_at=now,
        ))

    return learnings


# ═══════════════════════════════════════════════
# SESSION MEMORY
# ═══════════════════════════════════════════════
#
# "Continue where I left off"
#
# The Agent SDK supports session resumption — if you
# capture the session_id from a run, you can resume
# the exact conversation later.
#
# This is useful for:
#   - Long-running tasks interrupted by timeout
#   - Multi-step workflows across different invocations
#   - Debugging (replay from a specific point)
#
# Storage: JSON dict mapping workflow keys to session info
#

def save_session(
    key: str,
    session_id: str,
    metadata: dict | None = None,
) -> None:
    """Save a session ID for later resumption."""
    ensure_memory_dir()
    try:
        sessions = json.loads(SESSIONS_FILE.read_text())
    except Exception:
        sessions = {}

    sessions[key] = {
        "session_id": session_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }

    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2))


def load_session(key: str) -> dict | None:
    """Load a saved session by key. Returns None if not found."""
    ensure_memory_dir()
    try:
        sessions = json.loads(SESSIONS_FILE.read_text())
        return sessions.get(key)
    except Exception:
        return None


def list_sessions() -> dict:
    """List all saved sessions."""
    ensure_memory_dir()
    try:
        return json.loads(SESSIONS_FILE.read_text())
    except Exception:
        return {}


def delete_session(key: str) -> bool:
    """Delete a saved session. Returns True if found and deleted."""
    ensure_memory_dir()
    try:
        sessions = json.loads(SESSIONS_FILE.read_text())
        if key in sessions:
            del sessions[key]
            SESSIONS_FILE.write_text(json.dumps(sessions, indent=2))
            return True
    except Exception:
        pass
    return False


# ═══════════════════════════════════════════════
# CONVENIENCE: FULL MEMORY CONTEXT FOR PROMPTS
# ═══════════════════════════════════════════════

def get_memory_context(
    topic: str,
    client: str | None = None,
    max_episodes: int = 3,
    max_facts: int = 8,
) -> str:
    """
    Build a complete memory context block for injection into prompts.

    This is the single function you call before starting a workflow.
    It combines relevant episodic and semantic memory into one
    prompt-ready string.

    Returns empty string if no relevant memory exists.
    """
    sections = []

    # Episodic: past runs about similar topics
    episodes = find_relevant_episodes(topic, client, max_results=max_episodes)
    if episodes:
        sections.append(format_episodes_for_prompt(episodes))

    # Semantic: learned facts and preferences
    semantic = get_semantic_facts_for_prompt(max_facts=max_facts)
    if semantic:
        sections.append(semantic)

    if not sections:
        return ""

    return (
        "\n\n# AGENT MEMORY (from past experience)\n"
        + "\n\n".join(sections)
        + "\n\nUse these memories to improve your output. "
        "Avoid repeating past mistakes. Build on what worked."
    )
