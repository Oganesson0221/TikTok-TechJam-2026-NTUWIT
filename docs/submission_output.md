# Final model output contract

## Checkpoint

`best.pt` is an atomic PyTorch checkpoint containing:

- validation-best model weights;
- optimizer state for reproducibility;
- complete experiment configuration;
- training-only categorical vocabularies and field dimensions;
- selected epoch and validation metrics.

## Ranked predictions

`export-recommendations` loads the checkpoint and reads only the feature columns
needed for inference. It intentionally does not request `is_click` or any
auxiliary outcome label from the input CSV.

Default CSV schema:

| Column | Meaning |
|---|---|
| `user_id` | KuaiRand user identifier |
| `video_id` | candidate video identifier |
| `score` | bounded click-propensity score used for ranking; not asserted to be calibrated |
| `rank` | one-based descending rank within the user |

Use `--top-k 50` for a ranked top-50 file, `--columns` to omit unneeded fields,
and `--column-map` with a JSON object to apply organizer-specific names. The
adjacent `.meta.json` records paths, row/user counts, selected columns, SHA-256,
and `label_columns_read: []`.

The completed KuaiRand-1K bonus export is
`artifacts/submissions/kuairand_1k_side_multitask_test_scores.csv`: 3,328,531
rows for 490 users, with SHA-256
`9e92d73a3184f5dae590928d6b50d684cce6f2f61d8536b4942da0531916709d`.
Its manifest also records `label_columns_read: []`.

Example:

```bash
export-recommendations \
  --checkpoint artifacts/checkpoints/<winner>/best.pt \
  --input-csv data/prepared/test.csv \
  --output-csv artifacts/submissions/kuairand_pure_top50.csv \
  --top-k 50 \
  --columns user_id,video_id,score,rank \
  --column-map configs/submission_column_map.example.json
```

The released challenge materials do not currently specify whether submission
rows should contain scores, ranks, or top-K item arrays. This adapter covers the
first two without retraining; an array serializer can be added once the exact
organizer example is available. Do not rename the generic output “official”
until it has been checked against that example.
