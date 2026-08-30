# System architecture

This is a recommendation-ranking system. Its input is structured user, video
and impression context; its output is a click score and descending candidate
rank. It does not analyze media authenticity.

![Input and output scope](assets/system-scope.svg)

```mermaid
flowchart LR
    A[Official KuaiRand-Pure archive] --> B[MD5 verification and safe extraction]
    B --> C[Fixed train / validation / quarantined test split]
    C --> D[Leakage audit and manifest]
    D --> E[Feature encoder fit on train only]
    E --> F[MF / DeepFM / DCN / multi-task DCN]
    O[Structured model-driven policy] -->|metrics + failures| G[Bounded research agent]
    G -->|hypothesis + config diff| F
    F --> H[A100 training]
    H --> I[NDCG@10 + Recall@50]
    I --> G
    H --> J[Atomic validation-best checkpoint]
    G --> K[Convergence / walltime / experiment budget]
    G --> L[OOM, NaN and CUDA recovery]
    G --> M[Iteration JSONL + resource report]
    J --> N[Final selected model]
    M --> N
    N --> P[Label-blind ranked CSV + checksum]
```

## Safety boundaries

- The development API accepts only train and validation paths; it has no test
  argument.
- Categorical vocabularies are fit exclusively on training rows. Unseen
  validation categories map to a reserved unknown bucket.
- Only logged non-click impressions are used as negatives. No external data or
  synthetic unexposed pairs are introduced.
- The cumulative video-statistics file is excluded because it may contain
  future-derived aggregates.
- Checkpoints are written atomically and selected by validation metrics.

## Autonomous control contract

Each candidate is a bounded hypothesis with a configuration diff. In
model-driven mode, the policy reads previous metrics, errors, and recoveries and
selects the next candidate through a strict JSON schema; offline mode provides
a reproducible queue. The agent trains, evaluates, records the outcome, and
either advances, retries safely, or stops on convergence/budget. It does not
execute arbitrary generated code on the cluster. This makes the run
reproducible, inexpensive, and auditable.
