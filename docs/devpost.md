# Devpost project description

## Inspiration

Recommender-system research is an iterative loop: inspect data, propose a
change, train, evaluate, reflect, and repeat. Much of that loop is structured
but still consumes ML engineers' time. We built an autonomous research pipeline
that performs controlled recommendation experiments on KuaiRand-Pure and leaves
an auditable record of every success, failure, and recovery.

## What it does

The system downloads and verifies the official KuaiRand-Pure archive, creates
the fixed splits, audits leakage boundaries, and trains modular MF, DeepFM,
Deep & Cross Network, and multi-task DCN models. A bounded research agent proposes meaningful
configuration experiments, evaluates NDCG@10 and Recall@50, saves the
validation-best checkpoint, and stops on convergence or resource budgets.

When a run fails, the agent records the error and applies a targeted recovery.
On NSCC, DCN encountered a CUDA BLAS initialization failure. The agent reduced
batch size from 2048 to 1024, cleared CUDA state, retried, and produced the best
model without human model-tuning intervention.

A professional research console makes the model contract tangible. Three
fictional personas map to anonymized real KuaiRand users and display actual
ranked exports, while a dedicated results tab presents only successful
benchmark runs. The interface explicitly distinguishes recommendation ranking
from fake-video detection and explains the structured input and ranked output.

## How we built it

- Python, pandas and NumPy for data preparation and validation.
- PyTorch for MF, DeepFM, DCN, and multi-feedback learning.
- Pointwise BCE, pairwise BPR, and hybrid losses for controlled objective
  research against the ranking metrics.
- YAML/JSON experiment configurations for controlled code-free changes.
- PBS Pro and NSCC ASPIRE2A A100 GPUs for scheduled training.
- OpenAI Codex for repository implementation and engineering assistance.
- Pytest for feature, metric, model, recovery, and checkpoint tests.

No external training data or pretrained recommender weights are used. The
controller supports strict-schema model-driven decisions with token accounting,
and a deterministic offline mode when the cluster has no API credentials.

## Results

| Experiment | NDCG@10 | Recall@50 | Outcome |
|---|---:|---:|---|
| Item popularity | 0.817216 | 0.999867 | Non-personalized control |
| Matrix factorization | 0.818786 | 0.999878 | Personalized ID control |
| DeepFM, side features | 0.832888 | 0.999918 | Feature interactions improve ranking |
| DeepFM, all negatives | 0.833730 | 0.999921 | Preserving exposure distribution helps |
| DCN, long-schedule seed 43 | **0.837940** | 0.999918 | Best validation-selected checkpoint |

On the permitted KuaiRand-1K bonus corpus, the winning side-feature multi-task
DCN reached **0.690004 NDCG@10 and 0.030152 Recall@50**. Its median validation
candidate count is 4,682 and zero users have 50 or fewer candidates, so this
second benchmark directly addresses the saturated-Recall limitation in the
small Pure logged split. Static author/video/user attributes improved NDCG by
0.022614 (3.39% relative) over the strongest 1K ID/context control.

The completed primary A100 sweeps, recovery, and OpenAI-selected robustness run
consumed 0.56667 PBS GPU-hours. The
scores use logged validation impressions and must be checked with the
organizer's exact evaluator before being described as leaderboard-equivalent.
Against our internal popularity control, the final NDCG gain is 0.020724
(2.54% relative). We keep this separate from the unavailable official CWM
baseline delta.

## What we learned

Simply forcing the last epoch was counterproductive. The final DCN had an
80-epoch ceiling, stopped after 12, and restored epoch 2. Early stopping,
regularization, learning-rate control, and faithful negative exposure were more
valuable than raw epoch count.

The challenge statement references CWM, but the public upstream CWM repository
optimizes counterfactual watch-time/long-view objectives rather than the stated
click + Recall@50 contract. We retained its DeepFM/DCN backbone families while
making the click objective and evaluation assumptions explicit.

## Challenges

- The later official log is not globally timestamp-sorted, so literal row
  halves overlap in time. The manifest records this rather than silently
  changing organizer membership.
- Logged-impression Recall@50 is close to one because many users have fewer than
  50 logged candidates. Candidate-count diagnostics quantify the saturation; we
  also report a much harder full-catalog popularity result and clearly flag the
  missing organizer evaluator.
- NSCC's current PyTorch modulefile points to `site-package` instead of
  `site-packages`; the PBS job applies a narrow path correction and validates
  imports in a scheduled bootstrap job.

## Resource use

- Hardware: NVIDIA A100-SXM4-40GB on NSCC ASPIRE2A.
- Recorded primary GPU time: 0.56667 GPU-hours.
- Recorded optional 1K training GPU time: 0.41944 PBS GPU-hours (export
  inference reported separately).
- Manual model-training interventions: 0; bonus integration interventions: 2
  (one row-split audit correction and one export batch-size correction).
- Successful GPT-5 mini policy decision: 1,513 input + 388 output tokens,
  estimated $0.001154. One earlier capped request returned incomplete JSON;
  combined API spend remained below $0.01.
- Datasets: KuaiRand-Pure required benchmark plus the permitted KuaiRand-1K
  bonus benchmark; no unrelated external training data.

## Limitations and next steps

The organizer evaluation script, official numeric baseline, convergence epsilon,
and final submission schema were not included in the supplied materials. The
pipeline isolates these interfaces so they can be replaced without rewriting
training. The final repository includes a generic label-blind schema adapter,
multi-feedback objectives, repeated-seed evaluation, and an optional
KuaiRand-1K bonus pipeline.

## Team contributions

- Divisha: data understanding, fixed splits, baseline and metric verification.
- Rishika: modular recommendation models, features, negative sampling,
  checkpointing and experiment configurations.
- Saba: autonomous hypothesis/train/evaluate/reflect loop and recovery logic.
- Khanak: integration, structured logs, resource accounting, documentation and
  submission presentation.
