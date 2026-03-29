"""
Pydantic models for structured agent outputs.

Why Pydantic here?
- The evaluator subagent returns JSON scores
- Your orchestrator code needs to PARSE that JSON reliably
- Pydantic validates the shape: if a field is missing or wrong type,
  you get a clear error instead of a silent bug
- In production, you'd also use these for API response schemas
"""

from pydantic import BaseModel, Field
from typing import List


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
