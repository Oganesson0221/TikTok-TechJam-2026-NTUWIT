# LoopRank — TikTok TechJam 2026

Autonomous recommender-research pipeline for the official Starter Kit task:
predict `long_view` and rank logged candidates within each user. The official
metrics are GAUC and nDCG@5; the primary score is their mean.

## Submission-ready folders

- `submission/03_run_iteration_logs/` — per-iteration hypotheses, code diffs,
  official validation metrics, errors/recoveries, and intervention count.
- `submission/04_final_submission_results/` — final Pure prediction CSV,
  ensemble checkpoints, results/deltas, schema validation, and resource usage.

The required upload is
`submission/04_final_submission_results/files/kuairand_pure.csv`, with exact
columns `row_id,user_id,video_id,score` and 170,588 test rows.

The 1K prediction CSV and its checkpoint exceed GitHub's ordinary file-size
limit and are tracked with Git LFS. Before committing or cloning all submission
artifacts, run:

```bash
git lfs install
```

## Current required-benchmark result

| KuaiRand-Pure validation model | GAUC | nDCG@5 | Primary |
|---|---:|---:|---:|
| Official FM baseline (published) | 0.667400 | 0.535700 | 0.601600 |
| FM reproduced locally | 0.667133 | 0.535805 | 0.601469 |
| DeepFM | 0.661683 | 0.532799 | 0.597241 |
| Seed-averaged FM/DeepFM/context-FM ensemble | **0.669983** | **0.537199** | **0.603591** |

The ensemble weights were selected using validation only. Test outcome columns
are never requested by the prediction exporters.

The portal-requested supplemental metrics are **NDCG@10 0.812924** and
**Recall@50 0.999979**. They are kept separate from the executable Starter Kit
score; Recall@50 is nearly saturated because almost every Pure validation user
has at most 50 candidates.

The completed KuaiRand-1K A100 bonus run reached **0.647500 GAUC / 0.571905
nDCG@5 / 0.609702 primary**, with supplemental **0.562493 NDCG@10 /
0.096342 Recall@50**, and produced 4,132,081 aligned test scores.

## Completed OpenAI autonomous-agent run

The official Pure agent completed **6/6 experiments** with every decision made
by `gpt-5-mini`; no queued fallback was used. It consumed **4,112 input + 1,859
output = 5,971 API tokens**, stopped on convergence, and required **0 manual
interventions**. Its selected checkpoint was multitask DCN with auxiliary weight
0.1 at **0.668100 GAUC / 0.536646 nDCG@5 / 0.602373 primary**. Because the
seed-averaged ensemble remained stronger, the agent result is retained as the
autonomy audit and checkpoint rather than replacing the final prediction file.

The first NSCC launch exposed blocked API egress on compute nodes. That failure
and the successful recovery to the identical local OpenAI-mode configuration
are recorded in `submission/03_run_iteration_logs/`.

## Setup and verification

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,official,agent]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Prepare the date-based official Pure splits from the Starter Kit data:

```bash
PYTHONPATH=src python3 -m research_rec.official_prepare_cli \
  --data-dir data/raw/KuaiRand-Pure/data \
  --output-dir data/official_pure \
  --variant pure
```

The fixed split periods are train 2022-04-08 through 2022-04-21,
validation 2022-04-22 through 2022-04-28, and test 2022-04-29 through
2022-05-08.

Reproduce every Pure component, rebuild the frozen validation-selected
ensemble, populate the submission folder, and run the independent checker:

```bash
scripts/reproduce_official_pure.sh
```

This is a full training run. For a quick verification of the included
artifacts, run the test command above and the Starter Kit check shown in
`docs/submission_output.md`.

## KuaiRand-1K bonus run

The streaming hashed-FM implementation is in `src/research_rec/large_scale.py`.
It trains CSV chunks without loading the full corpus into memory, evaluates
both reported metric pairs, and writes the exact Starter Kit schema.

NSCC access intentionally uses an interactive SSH master session so passwords
never enter commands, logs, or repository files:

```bash
scripts/nscc_login.sh              # one password prompt; session backgrounds
scripts/nscc_sync.sh
scripts/nscc_submit.sh official-1k
scripts/nscc_status.sh
scripts/nscc_fetch_official_1k.sh
```

The login also transfers OPENAI_API_KEY from the ignored local .env through
encrypted SSH into a mode-600 remote file. The value is never synced, printed,
or committed. `scripts/nscc_shell.sh` automatically loads it into the remote
interactive environment; PBS jobs load the same protected file.

The submission includes only completed, fetched, and measured benchmark runs.

## Demo

```bash
PYTHONPATH=src python3 -m research_rec.demo_server
```

Open <http://127.0.0.1:8080>. The API reads the official four-column prediction
file and derives display ranks per user without modifying the submission.
