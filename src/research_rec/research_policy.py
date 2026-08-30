from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass
class ResearchDecision:
    candidate_name: str
    rationale: str
    reflection: str
    source: str
    input_tokens: int = 0
    output_tokens: int = 0


class ResearchPolicy(Protocol):
    def choose(self, candidates: Sequence[Any], records: list[dict[str, Any]]) -> ResearchDecision: ...


class QueuedResearchPolicy:
    """Reproducible offline control policy used when no model API is configured."""

    def choose(self, candidates: Sequence[Any], records: list[dict[str, Any]]) -> ResearchDecision:
        candidate = candidates[0]
        reflection = "No prior experiment; establish a control." if not records else _compact_reflection(records[-1])
        return ResearchDecision(candidate.name, candidate.hypothesis, reflection, "queued")


def _compact_reflection(record: dict[str, Any]) -> str:
    if record.get("status") != "completed":
        errors = record.get("errors", [])
        error_name = errors[-1].get("type", "unknown error") if errors else "unknown error"
        return f"Previous experiment failed with {error_name}; route to a different bounded candidate."
    metrics = record.get("metrics", {})
    return (
        f"Previous {record['name']} produced NDCG@10={metrics.get('ndcg@10', 0):.6f} "
        f"and Recall@50={metrics.get('recall@50', 0):.6f}; compare the next controlled change."
    )


class OpenAIResearchPolicy:
    """Result-driven experiment selection using Responses API Structured Outputs."""

    def __init__(
        self,
        model: str | None = None,
        *,
        max_output_tokens: int = 1200,
        max_total_tokens: int = 20_000,
    ):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as error:
            raise RuntimeError("Install the 'agent' extra to use decision_mode=openai") from error
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for decision_mode=openai")
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("Set decision_model or OPENAI_MODEL for decision_mode=openai")
        if max_output_tokens < 64 or max_total_tokens < max_output_tokens:
            raise ValueError("OpenAI token budgets must be positive and total must cover one response")
        self.max_output_tokens = max_output_tokens
        self.max_total_tokens = max_total_tokens
        self.total_tokens = 0
        self.client = OpenAI()

    def choose(self, candidates: Sequence[Any], records: list[dict[str, Any]]) -> ResearchDecision:
        candidate_payload = [
            {
                "name": candidate.name,
                "hypothesis": candidate.hypothesis,
                "config": candidate.config,
                "overrides": candidate.overrides,
            }
            for candidate in candidates
        ]
        history = [
            {
                "name": record.get("name"),
                "status": record.get("status"),
                "metrics": record.get("metrics"),
                "errors": record.get("errors"),
                "recoveries": record.get("recoveries"),
                "config_changes": record.get("config_changes"),
            }
            for record in records
        ]
        schema = {
            "type": "object",
            "properties": {
                "candidate_name": {"type": "string", "enum": [candidate.name for candidate in candidates]},
                "rationale": {"type": "string", "maxLength": 400},
                "reflection": {"type": "string", "maxLength": 400},
            },
            "required": ["candidate_name", "rationale", "reflection"],
            "additionalProperties": False,
        }
        request_input = json.dumps(
            {"objective": "maximize validation NDCG@10 and Recall@50", "history": history, "candidates": candidate_payload}
        )
        # Conservative preflight estimate; the exact usage returned by the API
        # is accumulated below and prevents another request after the cap.
        estimated_input_tokens = max(1, len(request_input) // 3)
        if self.total_tokens + estimated_input_tokens + self.max_output_tokens > self.max_total_tokens:
            raise RuntimeError("OpenAI research-policy token budget exhausted before request")
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You are an autonomous recommender-systems research planner. Select exactly one safe, bounded "
                "experiment from the supplied candidates. Reflect on measured validation results and failures, "
                "prefer controlled causal comparisons, avoid hidden-test reasoning, and do not invent metrics. "
                "Keep rationale and reflection to at most two concise sentences each."
            ),
            input=request_input,
            text={
                "format": {"type": "json_schema", "name": "experiment_decision", "strict": True, "schema": schema},
                "verbosity": "low",
            },
            reasoning={"effort": "low"},
            max_output_tokens=self.max_output_tokens,
            store=False,
        )
        parsed = json.loads(response.output_text)
        usage = response.usage
        input_tokens = int(usage.input_tokens if usage else 0)
        output_tokens = int(usage.output_tokens if usage else 0)
        self.total_tokens += input_tokens + output_tokens
        if self.total_tokens > self.max_total_tokens:
            raise RuntimeError("OpenAI research-policy token budget exceeded; refusing further decisions")
        return ResearchDecision(
            candidate_name=parsed["candidate_name"],
            rationale=parsed["rationale"],
            reflection=parsed["reflection"],
            source=f"openai:{self.model}",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
