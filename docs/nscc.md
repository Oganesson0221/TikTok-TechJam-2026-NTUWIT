# NSCC A100 workflow

No password is stored in this repository. Authentication uses a temporary SSH
ControlMaster socket created by the user in their own Terminal.

## 1. Configure the allocation

```bash
cp cluster/nscc.env.example cluster/nscc.env
```

Replace `YOUR_PROJECT_ID` in `cluster/nscc.env`. The file is ignored by Git.
The default NTU endpoint is `aspire2antu.nscc.sg`; NTU may require its jump-host
access before that endpoint is reachable.

## 2. Authenticate

Run this in a normal macOS Terminal, not in chat:

```bash
scripts/nscc_login.sh
```

Type the account password only at SSH's invisible password prompt. Keep the
session open, or exit normally and rely on its four-hour persistent control
socket. Other scripts reuse that socket and never receive the password.

## 3. Inspect, synchronize, and bootstrap

```bash
scripts/nscc_inspect.sh
scripts/nscc_sync.sh
scripts/nscc_bootstrap.sh
```

Synchronization transfers source code and the 105 MB prepared dataset, but not
the duplicate raw/download archives, remote runtime cache, or local model
artifacts. Bootstrap verifies NSCC's maintained
`pytorch/2.10.0-py3-cu12.6` module and project imports in a one-hour CPU PBS
job. NSCC requires all computation, including environment preparation and
post-processing, to run through the scheduler.
The PBS files also correct the modulefile's current `site-package`/`site-packages`
path typo explicitly; bootstrap fails closed if PyTorch or project imports are
still unavailable.
Check the returned bootstrap job ID with `scripts/nscc_status.sh <JOB_ID>` and
submit training afterward. The submission helper adds a PBS `afterok`
dependency automatically, so training cannot start unless bootstrap succeeds.

## 4. Submit training

Full six-experiment autonomous run on one A100:

```bash
scripts/nscc_submit.sh agent
```

One model only:

```bash
scripts/nscc_submit.sh deepfm
scripts/nscc_submit.sh dcn
scripts/nscc_submit.sh dcn-retry
scripts/nscc_submit.sh hardened-sweep
```

The PBS request is one A100, 16 CPU cores, 110 GB RAM, and a four-hour walltime
for the agent (`cluster/nscc_train.pbs`). The single-model job requests two
hours. All jobs enter through the current `normal` routing queue. The allocation
is passed to `qsub -P` from ignored configuration rather than committed in the
PBS file.

## 5. Monitor and fetch

```bash
scripts/nscc_status.sh
scripts/nscc_fetch.sh
```

The autonomous run writes `artifacts/agent_run/iterations.jsonl`,
`artifacts/agent_run/summary.json`, and a validation-best checkpoint under each
experiment name. PBS output/error logs are fetched into
`artifacts/nscc_logs/`.

Scratch is transient working storage: NSCC may remove files not accessed for
more than 30 days and does not archive scratch at project end. Fetch important
checkpoints promptly with `scripts/nscc_fetch.sh`.

Optional KuaiRand-1K bonus preparation and training are scheduler-only jobs:

```bash
scripts/nscc_submit.sh bonus-prepare
scripts/nscc_submit.sh bonus-sweep
scripts/nscc_submit.sh bonus-side-sweep
```

Both training commands are held with `afterok` until the verified 1K archive
has been downloaded, extracted, split, and audited by the first job. Remote 1K
data is excluded from normal code synchronization so later `rsync --delete`
calls do not remove the large prepared corpus.

Bonus checkpoints can exceed 400 MB, so routine fetches skip them. After the
bonus summary identifies its winner, fetch only that checkpoint:

```bash
scripts/nscc_fetch_bonus_checkpoint.sh kuairand_1k_<winning_experiment>
```

Generate the large held-out score file as a scheduled inference job rather
than loading the 1K split on a laptop:

```bash
scripts/nscc_submit.sh bonus-export kuairand_1k_<winning_experiment>
```

Any disposable environment cache must also be removed through PBS rather than
on a login node; `cluster/nscc_cleanup.pbs` is the guarded cleanup job used for
that purpose.
