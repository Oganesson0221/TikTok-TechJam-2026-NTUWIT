# Model artifacts

## Selected local checkpoints

| Role                         | Local path                                                 |   Size | SHA-256                                                            |
| ---------------------------- | ---------------------------------------------------------- | -----: | ------------------------------------------------------------------ |
| Primary KuaiRand-Pure winner | `artifacts/checkpoints/dcn_long_schedule_seed43/best.pt`   | 8.6 MB | `03ff13e3cd2939df6392a04dd356cb39aca36236289a185198caf1c196224436` |
| KuaiRand-1K bonus winner     | `artifacts/checkpoints/kuairand_1k_side_multitask/best.pt` | 597 MB | `eeef415e913694f415d530a8a3418f95d1428334c792c117b5a858485a3671cf` |

The primary winner is a DCN-v1 trained on KuaiRand-Pure. It completed 12 of an
80-epoch ceiling and restored validation-best epoch 2. Early stopping selecting
an early epoch is expected: later epochs did not improve the validation
selection score. The bonus winner is the side-feature multi-task DCN reported
for KuaiRand-1K.

## Local prediction exports

- `artifacts/submissions/kuairand_pure_top50.csv`
- `artifacts/submissions/kuairand_pure_scores.csv`
- `artifacts/submissions/kuairand_1k_side_multitask_test_scores.csv`

Their row counts and export checksums are recorded in
[`reports/submission_artifacts.json`](../reports/submission_artifacts.json).
These are generic label-blind ranked outputs; apply the organizer's exact
column mapping when its submission schema is available.

## Sharing or deploying

The static frontend can be deployed without model weights because it contains
small, verified prediction examples. Full local inference additionally needs
the checkpoint and prepared data. For a judge or teammate, publish the two
checkpoints as GitHub Release assets, Git LFS objects, or private cloud-storage
downloads, then verify the SHA-256 values above. Do not remove the entire
`artifacts/` ignore rule merely to push generated outputs into ordinary Git
history.
