# Baseline and evaluation contract

## What can be reproduced now

The challenge text fixes the label (`is_click`) and reports NDCG@10 and
Recall@50. This repository therefore provides two reproducible controls:

1. `run-kuairand-baseline`: smoothed item click-through rate, requiring no GPU.
2. `configs/mf_baseline.yaml`: a trainable ID-embedding baseline with early
   stopping and validation-best checkpointing.

Both train exclusively on `log_standard_4_08_to_4_21_pure.csv` and evaluate on
the first row-half of `log_standard_4_22_to_5_08_pure.csv`. The last row-half is
prepared but never accepted by the development training API.

## Why this is not labelled an exact CWM score reproduction

The public CWM repository optimizes watch-time/counterfactual-watch-time labels.
Its evaluator reports long-view NDCG, watch-time precision, AUC, MRR, and related
metrics. It does not expose this challenge's stated `is_click` + Recall@50
command or an organizer baseline score. Calling an unmodified upstream CWM run
an exact reproduction would therefore be misleading.

We audited upstream CWM commit `c36da4ba745a491545490be1b2b976180ab69c87`.
Its evaluator ranks observed rows within each `user_id`, matching this
repository's logged-impression grouping, but it does not implement the
challenge's click + Recall@50 reference score. The machine-readable audit and
source URLs are in `reports/official_source_audit.json`.

The DeepFM and DCN implementations here retain the backbone families used in
CWM while changing the objective to binary click prediction. Once the event's
starter evaluation script, submission schema, and numeric reference scores are
available, place them in the repository and compare with:

```bash
run-kuairand-baseline \
  --official-ndcg <NDCG_AT_10> \
  --official-recall <RECALL_AT_50>
```

## Evaluation protocol warning

The fast baseline reports two protocols:

- `validation_metrics`: ranks only logged validation impressions per user.
- `full_catalog_validation_metrics`: ranks all 7,583 items from the official
  static item table and treats unique clicked validation items as relevant.

The former can make Recall@50 nearly one because many users have fewer than 50
logged candidates. Training summaries now report median and p95 candidate
counts plus the fraction of evaluated users at or below K, making that
saturation measurable rather than implicit. The latter is much harder. Neither should be called the
official competition score until compared against the organizer-provided
evaluation script.

## Split audit

The instruction says to use the first and last 50% of the later standard log.
The source file is not globally ordered by `time_ms`, so those halves overlap in
date/time. The pipeline follows the literal row-half rule and records
`temporal_half_overlap: true` in `data/prepared/manifest.json`; it does not sort
the file and silently alter organizer-defined membership.
