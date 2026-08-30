# Submission checklist

## Complete in the repository

- [x] Reproducible KuaiRand-Pure preparation and MD5 verification
- [x] Fixed split manifest and leakage audit
- [x] Baselines, modular recommenders, multi-task and pairwise objectives
- [x] Validation-best checkpoints and early stopping
- [x] Autonomous experiment execution, convergence, and recovery
- [x] Optional strict-schema model-driven decision policy
- [x] Iteration, resource, token, failure, and intervention records
- [x] Label-blind ranked prediction exporter and checksum manifest
- [x] KuaiRand-1K bonus preparation, A100 training, checkpoint, and held-out
  export path
- [x] README, architecture, Devpost draft, pitch outline, and demo script
- [x] Professional responsive frontend with personas, real ranked-output API,
  scope guidance, successful-results table, and accessible results graph
- [x] Automated tests and shell/config validation

## Team actions before the Devpost deadline

- [ ] Review team contribution wording with all four members
- [ ] Commit the repository and push it to a public GitHub URL
- [ ] Upload the selected checkpoints/predictions through the organizer's
  accepted artifact mechanism; do not commit raw data or credentials
- [ ] Replace the generic prediction column mapping if the organizer releases
  an exact schema or example
- [ ] Insert the official baseline delta if numeric reference scores appear
- [ ] Record the three-minute demo using `docs/demo_script.md`
- [ ] Upload the demo publicly/unlisted to YouTube
- [ ] Paste `docs/devpost.md` into Devpost and add repository/video links
- [ ] Submit before **1 September 2026, 12:00pm SGT**

## Claims to avoid

- Do not call logged validation metrics hidden-test or leaderboard results.
- Do not call the internal popularity delta the official CWM delta.
- Do not claim the earlier queued NSCC sweeps made LLM calls. The separate real
  GPT-5 mini decision selected the seed-44 robustness run and records its own
  actual API usage in `reports/openai_policy_decision.json`.
- Do not expose the NSCC password, SSH socket, raw dataset, or API keys.
