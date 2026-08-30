#!/usr/bin/env python3
"""Ask the bounded OpenAI policy for one next-experiment decision.

This replays real completed metrics without retraining and writes an auditable
decision artifact. Credentials are read only from the process environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_rec.agent import Candidate
from research_rec.research_policy import OpenAIResearchPolicy


def _history() -> list[dict]:
    primary_path = Path("reports/nscc_iterations.jsonl")
    primary = [json.loads(line) for line in primary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bonus = json.loads(Path("reports/bonus_1k_results.json").read_text(encoding="utf-8"))
    bonus_records = [
        {
            "name": experiment["name"],
            "status": "completed",
            "metrics": {
                "ndcg@10": experiment["ndcg@10"],
                "recall@50": experiment["recall@50"],
            },
            "errors": [],
            "recoveries": experiment.get("automatic_recoveries", []),
            "config_changes": [],
        }
        for experiment in bonus["experiments"]
    ]
    return primary[-8:] + bonus_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one cost-bounded OpenAI research-policy decision")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--output", default="artifacts/openai_policy_audit.json")
    parser.add_argument("--max-output-tokens", type=int, default=2400)
    parser.add_argument("--max-total-tokens", type=int, default=20000)
    args = parser.parse_args()

    candidates = [
        Candidate(
            "kuairand_1k_side_allneg",
            "Test whether retaining all logged negative exposures complements the winning 1K side features.",
            "configs/nscc_1k_side_multitask_dcn.json",
            ["data.negative_ratio=null"],
        ),
        Candidate(
            "kuairand_1k_side_aux10",
            "Test whether weaker auxiliary transfer improves click ranking while keeping the winning side features.",
            "configs/nscc_1k_side_multitask_dcn.json",
            ["model.auxiliary_loss_weight=0.1"],
        ),
        Candidate(
            "dcn_long_schedule_seed44",
            "Add a third Pure seed to measure robustness of the validation-selected DCN result.",
            "configs/nscc_dcn_features.json",
            [
                "training.batch_size=1024",
                "training.learning_rate=0.0005",
                "training.epochs=80",
                "training.patience=10",
                "training.lr_scheduler_factor=0.5",
                "training.seed=44",
            ],
        ),
    ]
    policy = OpenAIResearchPolicy(
        args.model,
        max_output_tokens=args.max_output_tokens,
        max_total_tokens=args.max_total_tokens,
    )
    history = _history()
    decision = policy.choose(candidates, history)
    # Official GPT-5 mini standard token rates as of this recorded run.
    estimated_cost_usd = decision.input_tokens * 0.25 / 1_000_000 + decision.output_tokens * 2.00 / 1_000_000
    result = {
        "purpose": "real model-driven next-experiment selection over completed Pure and 1K history",
        "execution_scope": "decision only; no training or hidden-test metrics supplied to the model",
        "model": args.model,
        "decision": {
            "candidate_name": decision.candidate_name,
            "rationale": decision.rationale,
            "reflection": decision.reflection,
            "source": decision.source,
        },
        "usage": {
            "input_tokens": decision.input_tokens,
            "output_tokens": decision.output_tokens,
            "total_tokens": decision.input_tokens + decision.output_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "configured_max_output_tokens": args.max_output_tokens,
            "configured_max_total_tokens": args.max_total_tokens,
        },
        "history_records": len(history),
        "candidate_names": [candidate.name for candidate in candidates],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
