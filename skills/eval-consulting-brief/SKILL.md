---
name: eval-consulting-brief
description: >
  Evaluates consulting briefs against quality criteria.
  Use when asked to review, score, or critique a consulting brief.
  Returns structured scores and actionable feedback.
---

# Consulting Brief Evaluator

## Scoring Rubric (1-5 each)

1. **Executive Summary** — Under 4 sentences? Captures essence?
2. **Research Depth** — At least 3 cited data points? Current info?
3. **Talking Points** — Actionable and specific? Would impress a client?
4. **Risk Awareness** — Identifies real sensitivities? No blind spots?
5. **Strategic Questions** — Show preparation? Uncover opportunities?
6. **Overall Readiness** — Could a partner walk in confident?

## Output Format (strict JSON)

```json
{
  "scores": {
    "executive_summary": 4,
    "research_depth": 3,
    "talking_points": 5,
    "risk_awareness": 3,
    "strategic_questions": 4,
    "overall_readiness": 4
  },
  "total_score": 23,
  "pass": true,
  "feedback": "Specific improvement suggestions here",
  "missing_elements": ["list", "of", "gaps"]
}
```

## Rules

- Score of 20/30 or above = PASS
- Below 20 = FAIL with specific revision instructions
- Be harsh but fair — a partner's reputation depends on this
