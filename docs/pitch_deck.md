# Final pitch deck outline

## Slide 1 — The bottleneck

**From repetitive recommender tuning to an auditable research agent**

MLEs repeatedly inspect, hypothesize, train, evaluate, and recover. Our system
turns that loop into a reproducible research process for KuaiRand rather than a
single notebook or one-off model.

## Slide 2 — End-to-end system

Show the frontend's persona ranking, then `docs/architecture.md`, and walk through:

1. verified official dataset and leakage-safe splits;
2. train-only feature encoding;
3. MF, DeepFM, DCN, multi-task DCN, and pairwise objectives;
4. result-driven experiment selection;
5. A100 training, validation-best checkpointing, recovery, and export.

State the scope precisely: structured click ranking, not fake-video detection.

## Slide 3 — What is autonomous

- Each iteration has a hypothesis and exact configuration diff.
- The policy reads prior metrics, errors, and recoveries before selecting the
  next bounded experiment.
- It stops on convergence or wall-clock/experiment budget.
- OOM, non-finite loss, and CUDA initialization failures trigger structured
  recovery rather than ending the run.
- Decision source, reflection, LLM tokens, GPU-hours, and manual interventions
  are recorded.

## Slide 4 — Model research, not blind tuning

- Static and temporal features capture user, item, context, and time.
- Logged-negative policies preserve the exposure distribution.
- Multi-task DCN tests transfer from long-view, like, and follow feedback.
- BPR/hybrid objectives optimize within-user clicked-over-non-clicked order.
- Larger epoch ceilings use schedulers and early stopping; later weights are
  never selected merely because they trained longer.

## Slide 5 — Evidence and recovery

Show `reports/nscc_iterations.jsonl`:

- actual `CUBLAS_STATUS_NOT_INITIALIZED` failure;
- automatic batch-size reduction from 2048 to 1024;
- successful retry and subsequent controlled sweep;
- zero manual interventions inside the experiment controller.

## Slide 6 — Results, cost, and honest evaluation

Show the final results table from `README.md` and resource totals from
`reports/nscc_results.json`. Explain that candidate diagnostics found a median
of five logged candidates per user and 99.90% at or below Recall@50, which is
why Recall saturates. NDCG is the discriminating local metric.

Contrast this with the completed KuaiRand-1K bonus result: median 4,682
candidates, zero users at or below 50, and 0.690004 NDCG@10 / 0.030152
Recall@50. Side features improve NDCG by 3.39% over the strongest 1K
ID/context control.

Do not claim an official delta: the organizer numeric baseline/evaluator was
not present in the released materials. Show `reports/official_source_audit.json`
and the label-blind ranked export instead.

## Slide 7 — Why it matters

The deliverable is a reusable recommender R&D control plane:

- cheap, bounded experiments instead of open-ended GPU use;
- auditable decisions instead of undocumented trial and error;
- reproducible failure recovery;
- schema-adaptable deployment output;
- optional scaling from KuaiRand-Pure to KuaiRand-1K.

## Likely judge questions

**Why is Recall@50 almost one?** 99.90% of evaluated logged user groups contain
50 or fewer candidates. We report this diagnostic explicitly and preserve the
required metric rather than hiding the saturation.

**Is this really autonomous?** The model-driven policy consumes experiment
history and emits a strict-schema selection and reflection. A real GPT-5 mini
decision over 12 completed records selected seed 44, which was then trained on
NSCC and recorded with exact token/cost use. Execution and
recovery are bounded; arbitrary generated code is not run on the cluster.

**Why not train for every epoch?** The validation-best DCN peaked early.
Eighty-epoch-cap experiments test longer schedules safely, but early stopping
prevents the final model from being an overfit last epoch.

**Did you use prohibited data?** No. Primary training uses KuaiRand-Pure; the
only additional corpus is the explicitly permitted KuaiRand-1K bonus variant.

**Can judges obtain predictions?** Yes. `export-recommendations` writes ranked,
label-blind CSV output and a checksum manifest; organizer column names can be
applied through a JSON mapping without retraining.
