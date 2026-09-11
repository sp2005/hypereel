# Evaluation status

Last updated: 2026-09-11

## Current decision

The development release gate is **not yet met**, so neither the full first-video run nor the sealed holdout has been executed.

The best tuned development slice was iteration 18:

| Metric | Result | Gate |
|---|---:|---:|
| Candidate recall at IoU 0.30 | 1.000 | 0.800 |
| Selected-event precision | 0.667 | 0.700 |
| Selected-event recall | 1.000 | 0.700 |
| Selection F1 | 0.800 | 0.700 |
| Selected-event recall at IoU 0.30 | 1.000 | 0.700 |
| Schema pass rate | 1.000 | 1.000 |

The clean independent slice from a different game was iteration 20:

| Metric | Result | Gate |
|---|---:|---:|
| Candidate recall at IoU 0.30 | 1.000 | 0.800 |
| Selected-event precision | 1.000 | 0.700 |
| Selected-event recall | 0.500 | 0.700 |
| Selection F1 | 0.667 | 0.700 |
| Selected-event recall at IoU 0.30 | 0.500 | 0.700 |
| Negative-window specificity | 1.000 | monitored |
| Schema pass rate | 1.000 | 1.000 |

The independent slice therefore fails the recall, F1, and temporal-recall gates. Iteration 21 tested an alternative-label verifier, regressed to zero recall, and was rejected; its behavior was reverted while its result remains in the audit trail.

## Cost status

- Cumulative estimated Nebius spend: **$2.53027**
- Hard cap: **$5.00**
- Remaining headroom: **$2.46973**

The cumulative source of truth is [`iterations/spend-ledger.json`](iterations/spend-ledger.json). It is never cleared between iterations.

## What has been accepted

- One-to-one, label-aware temporal matching.
- Candidate and selected-event recall at IoU 0.10, 0.30, and 0.50.
- Precision, recall, micro/macro F1, confusion matrices, balanced accuracy, specificity, average precision, calibration error, and threshold sweeps.
- Proposal-to-classification-to-selection funnel metrics.
- Centered clip shaping for long evidence windows.
- Dense action-frame sampling with explicit before/after context.
- Contrastive verification against rebounds, misses, and unforced turnovers.
- A recipe-level two-second separation between selected highlight fragments.
- Append-only iteration results with change notes, dataset hashes, code revision, provider usage, and cumulative spend.

## Current limitation

The available Nebius account exposes only `openbmb/MiniCPM-V-4_5`. It has changed semantic verdicts across identical temperature-zero inputs and has confused or rejected steals, rebounds, and made baskets. The present single-label clip schema also cannot represent two different events occurring inside one candidate window.

More prompt tuning against known timestamps would overfit the development set. The next planned benchmark is `nvidia/Cosmos-Reason2-2B` using native video at 4 FPS on a Brev GPU, with multi-label timestamped output.

## Audit trail

- Human-readable cumulative table and decisions: [`iterations/iteration-log.md`](iterations/iteration-log.md)
- Append-only complete run records: [`iterations/history.jsonl`](iterations/history.jsonl)
- Cumulative provider cost: [`iterations/spend-ledger.json`](iterations/spend-ledger.json)
- Metric definitions and release gates: [`iterations/README.md`](iterations/README.md)
- Development golden datasets: [`golden/cases/development.pipeline.jsonl`](golden/cases/development.pipeline.jsonl)
- Sealed holdout dataset: [`holdout/`](holdout/)

## Next action

Authenticate the locally installed Brev CLI with `brev login`. After authentication, select the cheapest available GPU with at least 24 GB VRAM, enforce a $5 compute ceiling and automatic shutdown, deploy Cosmos-Reason2-2B, and run the same development slices without opening the holdout.
