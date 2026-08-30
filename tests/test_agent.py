from pathlib import Path

from research_rec.agent import AgentConfig, Candidate, run_agent
from research_rec.research_policy import ResearchDecision


def test_agent_recovers_from_oom_and_records_run(tmp_path: Path):
    calls = 0

    def runner(config):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("CUDA out of memory")
        return {
            "best_metrics": {"ndcg@10": 0.2, "recall@50": 0.4},
            "checkpoint": str(tmp_path / "best.pt"),
            "device": "cuda",
            "epochs_completed": 1,
        }

    config = AgentConfig(
        max_experiments=1,
        max_retries=1,
        log_dir=str(tmp_path / "logs"),
        candidates=[Candidate("recover", "test recovery", "configs/mf_baseline.yaml")],
    )
    result = run_agent(config, smoke=True, runner=runner)
    assert result["status"] == "completed"
    assert result["experiments_succeeded"] == 1
    assert calls == 2
    assert "Reduced batch_size" in result["best_experiment"]["recoveries"][0]
    assert (tmp_path / "logs" / "iterations.jsonl").is_file()


def test_agent_treats_cublas_initialization_failure_as_recoverable(tmp_path: Path):
    calls = 0

    def runner(config):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("CUDA error: CUBLAS_STATUS_NOT_INITIALIZED")
        return {
            "best_metrics": {"ndcg@10": 0.3, "recall@50": 0.5},
            "checkpoint": str(tmp_path / "best.pt"),
            "device": "cuda",
            "epochs_completed": 1,
        }

    config = AgentConfig(
        max_experiments=1,
        max_retries=1,
        log_dir=str(tmp_path / "logs"),
        candidates=[Candidate("cublas", "test CUDA recovery", "configs/mf_baseline.yaml")],
    )
    result = run_agent(config, smoke=True, runner=runner)
    assert result["status"] == "completed"
    assert calls == 2
    assert "CUDA BLAS initialization failure" in result["best_experiment"]["recoveries"][0]


def test_result_driven_policy_selects_candidate_and_counts_tokens(tmp_path: Path):
    class FakePolicy:
        def choose(self, candidates, records):
            selected = candidates[-1]
            return ResearchDecision(selected.name, "metric-driven choice", "reflected on history", "test-llm", 11, 7)

    def runner(config):
        return {
            "best_metrics": {"ndcg@10": 0.4, "recall@50": 0.6},
            "checkpoint": str(tmp_path / "best.pt"),
            "device": "cpu",
            "epochs_completed": 1,
        }

    config = AgentConfig(
        max_experiments=1,
        log_dir=str(tmp_path / "policy_logs"),
        candidates=[
            Candidate("first", "first choice", "configs/mf_baseline.yaml"),
            Candidate("selected", "selected choice", "configs/mf_baseline.yaml"),
        ],
    )
    result = run_agent(config, smoke=True, runner=runner, policy=FakePolicy())
    assert result["best_experiment"]["name"] == "selected"
    assert result["best_experiment"]["decision_source"] == "test-llm"
    assert result["llm_input_tokens"] == 11
    assert result["llm_output_tokens"] == 7
