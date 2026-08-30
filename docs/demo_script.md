# Three-minute demo script

## 0:00–0:30 — Problem and architecture

Open the frontend Overview tab and select one of the three fictional personas.
Clarify that the task ranks candidate videos by click likelihood—it does not
detect fake videos. Show the persona's actual ranked export, then switch to
`docs/architecture.md`. Explain that the goal is not one hard-coded model;
it is an auditable research loop that proposes controlled experiments, trains,
evaluates, reflects, and recovers.

## 0:30–1:00 — Reproducible data and baseline

Run or show:

```bash
run-kuairand-baseline --download
```

Open `data/prepared/manifest.json` and point out archive MD5, exact row counts,
split hashes, feature files, and the excluded future-statistics file.

## 1:00–1:40 — Autonomous experiments

Show `configs/nscc_agent.json`, then:

```bash
run-research-agent --config configs/agent.yaml --smoke
```

Open `reports/nscc_iterations.jsonl` and highlight the hypothesis, configuration
change, metrics, runtime and checkpoint for each iteration.

Show `src/research_rec/research_policy.py`: model-driven mode reads the prior
metrics/errors and selects the next bounded experiment with a strict JSON
schema; queued mode makes the same execution path reproducible without cluster
API credentials.

## 1:40–2:15 — Failure recovery

Show the `dcn_side_neg4_retry` line. Explain:

1. CUDA returned `CUBLAS_STATUS_NOT_INITIALIZED`.
2. The agent classified it as recoverable.
3. Batch size changed from 2048 to 1024 and CUDA cache was cleared.
4. Retry succeeded without manual intervention and informed the final bounded
   learning-rate sweep.

## 2:15–2:45 — Results and cost

Open the frontend Results tab and show the NDCG progression graph, then open
`reports/nscc_results.json`. Show the model progression and total PBS GPU
hours. The final DCN reached NDCG@10 0.837940 with batch size 1024, learning
rate 0.0005, and seed 43. Its ceiling was 80 epochs; it stopped after 12 and
restored epoch 2. The seed-42 control reached 0.837561. Emphasize validation-best
checkpointing rather than treating later epochs as automatically better.

Open `reports/openai_policy_decision.json`. Show that the real GPT-5 mini
policy read 12 prior records, selected seed 44 for robustness, used 1,513 input
and 388 output tokens, and triggered a successful NSCC replication. Seed 44
scored 0.837006, producing a three-seed mean of 0.837502 without changing the
seed-43 winner.

Run or show the completed label-blind output:

```bash
export-recommendations --checkpoint artifacts/checkpoints/dcn_long_schedule_seed43/best.pt \
  --input-csv data/prepared/test.csv \
  --output-csv artifacts/submissions/kuairand_pure_top50.csv --top-k 50
```

Open `reports/submission_artifacts.json` and point out the row count, SHA-256,
and `label_columns_read: []`.

Then show `reports/bonus_1k_results.json`: the 1K side-feature DCN reaches
0.690004 NDCG@10 / 0.030152 Recall@50 across a median 4,682 candidates per user,
and its structured log shows two automatic CUDA recoveries with no manual
training intervention.

## 2:45–3:00 — Honest limitations

State that median candidates per user is 5 and 99.90% of evaluated users have
at most 50 candidates, so logged-impression Recall@50 is saturated. The
organizer evaluator was not supplied. Show `docs/baseline_contract.md`, which separates current
reproducible results from leaderboard claims and makes the evaluation adapter a
clear next integration point.
