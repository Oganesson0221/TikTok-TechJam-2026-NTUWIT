from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from .config import ExperimentConfig, apply_override, load_config, load_structured_file
from .training import train_experiment
from .research_policy import OpenAIResearchPolicy, QueuedResearchPolicy, ResearchPolicy


@dataclass
class Candidate:
    name: str
    hypothesis: str
    config: str
    overrides: list[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    max_experiments: int = 6
    walltime_hours: float = 4.0
    convergence_epsilon: float = 1e-4
    convergence_patience: int = 3
    max_retries: int = 2
    log_dir: str = "artifacts/agent_run"
    decision_mode: str = "queued"
    decision_model: str | None = None
    decision_max_output_tokens: int = 1200
    decision_max_total_tokens: int = 20_000
    candidates: list[Candidate] = field(default_factory=list)

    def validate(self) -> None:
        if self.max_experiments < 1 or self.walltime_hours <= 0:
            raise ValueError("Agent experiment and walltime budgets must be positive")
        if self.convergence_patience < 1 or self.max_retries < 0:
            raise ValueError("convergence_patience must be positive and max_retries non-negative")
        if not self.candidates:
            raise ValueError("Agent requires at least one candidate experiment")
        if self.decision_mode not in {"queued", "openai"}:
            raise ValueError("decision_mode must be queued or openai")
        if self.decision_max_output_tokens < 64 or self.decision_max_total_tokens < self.decision_max_output_tokens:
            raise ValueError("Decision token budgets must be positive and total must cover one response")


def load_agent_config(path: str | Path) -> AgentConfig:
    raw = load_structured_file(path)
    candidates = [Candidate(**candidate) for candidate in raw.pop("candidates", [])]
    unknown = set(raw) - {name for name in AgentConfig.__dataclass_fields__ if name != "candidates"}
    if unknown:
        raise ValueError(f"Unknown agent configuration keys: {sorted(unknown)}")
    config = AgentConfig(candidates=candidates, **raw)
    config.validate()
    return config


def _selection_score(metrics: dict[str, float]) -> float:
    return (float(metrics["ndcg@10"]) + float(metrics["recall@50"])) / 2.0


def _recovery(error: BaseException, config: ExperimentConfig) -> str | None:
    message = str(error).lower()
    if (
        "out of memory" in message
        or "mps backend out of memory" in message
        or "cublas_status_not_initialized" in message
    ):
        old = config.training.batch_size
        if old <= 1:
            return None
        config.training.batch_size = max(1, old // 2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        cause = "CUDA BLAS initialization failure" if "cublas_status_not_initialized" in message else "accelerator OOM"
        return f"Reduced batch_size from {old} to {config.training.batch_size} after {cause}"
    if isinstance(error, FloatingPointError) or "non-finite" in message or "nan" in message:
        old = config.training.learning_rate
        config.training.learning_rate *= 0.5
        return f"Reduced learning_rate from {old} to {config.training.learning_rate} after non-finite loss"
    return None


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_agent(
    agent: AgentConfig,
    smoke: bool = False,
    runner: Callable[[ExperimentConfig], dict[str, Any]] | None = None,
    policy: ResearchPolicy | None = None,
) -> dict[str, Any]:
    agent.validate()
    execute = runner or train_experiment
    started = time.perf_counter()
    log_dir = Path(agent.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    iterations_path = log_dir / "iterations.jsonl"
    iterations_path.unlink(missing_ok=True)
    records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    non_improving = 0
    stop_reason = "candidate_space_exhausted"
    decision_policy = policy or (
        OpenAIResearchPolicy(
            agent.decision_model,
            max_output_tokens=agent.decision_max_output_tokens,
            max_total_tokens=agent.decision_max_total_tokens,
        )
        if agent.decision_mode == "openai"
        else QueuedResearchPolicy()
    )
    remaining = list(agent.candidates)
    llm_input_tokens = 0
    llm_output_tokens = 0

    while remaining and len(records) < agent.max_experiments:
        iteration = len(records) + 1
        if time.perf_counter() - started >= agent.walltime_hours * 3600:
            stop_reason = "walltime_budget"
            break
        decision = decision_policy.choose(remaining, records)
        matching = [candidate for candidate in remaining if candidate.name == decision.candidate_name]
        if not matching:
            raise ValueError(f"Research policy selected unknown candidate: {decision.candidate_name}")
        candidate = matching[0]
        remaining.remove(candidate)
        llm_input_tokens += decision.input_tokens
        llm_output_tokens += decision.output_tokens
        experiment = load_config(candidate.config)
        for override in candidate.overrides:
            apply_override(experiment, override)
        experiment.training.experiment_name = candidate.name
        if smoke:
            experiment.data.max_train_rows = min(experiment.data.max_train_rows or 100_000, 100_000)
            experiment.data.max_validation_rows = min(experiment.data.max_validation_rows or 20_000, 20_000)
            experiment.training.epochs = min(experiment.training.epochs, 2)
            experiment.training.patience = min(experiment.training.patience, 2)
        experiment.validate()
        record: dict[str, Any] = {
            "iteration": iteration,
            "name": candidate.name,
            "hypothesis": candidate.hypothesis,
            "config_path": candidate.config,
            "config_changes": candidate.overrides,
            "status": "running",
            "errors": [],
            "recoveries": [],
            "manual_interventions": 0,
            "decision_source": decision.source,
            "agent_rationale": decision.rationale,
            "agent_reflection": decision.reflection,
            "llm_input_tokens": decision.input_tokens,
            "llm_output_tokens": decision.output_tokens,
        }
        iteration_started = time.perf_counter()
        result: dict[str, Any] | None = None
        for attempt in range(agent.max_retries + 1):
            try:
                result = execute(copy.deepcopy(experiment))
                break
            except Exception as error:  # recorded so a long autonomous run can continue
                event = {"attempt": attempt + 1, "type": type(error).__name__, "message": str(error)}
                record["errors"].append(event)
                action = _recovery(error, experiment)
                if action is None or attempt >= agent.max_retries:
                    break
                record["recoveries"].append(action)
        elapsed = time.perf_counter() - iteration_started
        record["elapsed_seconds"] = elapsed
        if result is None:
            record["status"] = "failed"
            _append_jsonl(iterations_path, record)
            records.append(record)
            continue
        score = _selection_score(result["best_metrics"])
        record.update(
            {
                "status": "completed",
                "metrics": result["best_metrics"],
                "selection_score": score,
                "checkpoint": result["checkpoint"],
                "device": result["device"],
                "epochs_completed": result["epochs_completed"],
                "accelerator_hours": elapsed / 3600 if result["device"] != "cpu" else 0.0,
            }
        )
        if best is None or score > best["selection_score"] + agent.convergence_epsilon:
            best = record
            non_improving = 0
        else:
            non_improving += 1
        _append_jsonl(iterations_path, record)
        records.append(record)
        if non_improving >= agent.convergence_patience:
            stop_reason = "converged"
            break

    total_elapsed = time.perf_counter() - started
    summary = {
        "status": "completed" if best is not None else "failed",
        "stop_reason": stop_reason,
        "experiments_attempted": len(records),
        "experiments_succeeded": sum(record["status"] == "completed" for record in records),
        "manual_interventions": 0,
        "llm_input_tokens": llm_input_tokens,
        "llm_output_tokens": llm_output_tokens,
        "total_elapsed_seconds": total_elapsed,
        "total_accelerator_hours": sum(float(record.get("accelerator_hours", 0)) for record in records),
        "best_experiment": best,
        "iterations_log": str(iterations_path.resolve()),
        "agent_config": {**asdict(agent), "candidates": [asdict(candidate) for candidate in agent.candidates]},
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
