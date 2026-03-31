"""
Pydantic models for structured agent outputs.

Why Pydantic here?
- The evaluator subagent returns JSON scores
- The router returns a classification
- Your orchestrator code needs to PARSE these reliably
- Pydantic validates the shape: if a field is missing or wrong type,
  you get a clear error instead of a silent bug
- In production, you'd also use these for API response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class EvalScores(BaseModel):
    """Individual scores for each evaluation criterion (1-5 scale)."""

    executive_summary: int = Field(
        ge=1, le=5, description="Crisp and under 4 sentences?")
    research_depth: int = Field(
        ge=1, le=5, description="At least 3 cited data points?")
    talking_points: int = Field(
        ge=1, le=5, description="Actionable and specific?")
    risk_awareness: int = Field(
        ge=1, le=5, description="Real sensitivities identified?")
    strategic_questions: int = Field(
        ge=1, le=5, description="Show deep preparation?")
    overall_readiness: int = Field(
        ge=1, le=5, description="Partner would feel confident?")


class EvalResult(BaseModel):
    """Complete evaluation result from the evaluator subagent."""

    scores: EvalScores
    total_score: int = Field(ge=0, le=30)
    passed: bool = Field(description="True if total_score >= 20")
    feedback: str = Field(description="Specific improvement suggestions")
    missing_elements: List[str] = Field(
        default_factory=list,
        description="List of gaps found in the brief",
    )


# ──────────────────────────────────────────────
# ROUTER PATTERN: Classification model
# ──────────────────────────────────────────────
#
# The router agent classifies incoming requests into
# a workflow type. Your code then branches based on
# this classification — deterministic routing.
#

class RouteClassification(BaseModel):
    """Router's classification of an incoming request."""

    workflow: Literal[
        "meeting_prep",
        "competitive_analysis",
        "technology_evaluation",
    ] = Field(
        description=(
            "meeting_prep: prepare for a client meeting with briefing doc. "
            "competitive_analysis: analyze competitors in a market. "
            "technology_evaluation: assess a technology for enterprise adoption."
        )
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Router's confidence in the classification (0-1)",
    )
    reasoning: str = Field(
        description="One sentence explaining why this workflow was chosen",
    )


# ──────────────────────────────────────────────
# MEMORY MODELS
# ──────────────────────────────────────────────

class Episode(BaseModel):
    """A single episodic memory — records what happened in a past run."""

    timestamp: str
    topic: str
    client: str | None = None
    workflow: str
    eval_score: int
    revisions_needed: int
    evaluator_feedback: str
    what_worked: str
    what_failed: str
    research_quality: str = ""
    brief_word_count: int = 0

    def to_summary(self) -> str:
        """Compact summary for injecting into prompts."""
        return (
            f"- [{self.timestamp[:10]}] {self.topic}"
            f"{' (' + self.client + ')' if self.client else ''}: "
            f"score {self.eval_score}/30, {self.revisions_needed} revisions. "
            f"Worked: {self.what_worked}. Failed: {self.what_failed}."
        )


class SemanticFact(BaseModel):
    """A single semantic memory — a learned fact or preference."""

    fact: str
    source: str  # which run/episode this was learned from
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    category: Literal[
        "preference",       # "partner prefers briefs under 600 words"
        "domain_knowledge",  # "healthcare clients care about HIPAA"
        "tool_insight",     # "Tavily works better with year in query"
        "quality_pattern",  # "talking points often score low on first draft"
    ]
    created_at: str
    last_used: str | None = None
