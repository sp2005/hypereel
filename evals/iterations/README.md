# Nebius evaluation loop

This directory is the durable audit trail for the development-only fix/evaluate/retest loop. The third game in `evals/holdout/` remains sealed until the development gates pass.

## Evaluation matrix

| Layer | Metrics | Purpose |
|---|---|---|
| Proposal | candidate recall | Separates local event-window discovery failures from vision failures. A candidate matches when it overlaps the externally labeled action interval. |
| Event selection | TP, FP, FN; precision, recall, micro/macro F1; false positives/minute | Measures correct selected events with one-to-one, same-label action-interval matching. |
| Class balance | per-type precision/recall/F1; confusion matrix; balanced accuracy | Prevents the frequent classes from hiding failures on rarer event types and exposes label mix-ups. |
| Temporal quality | mean temporal IoU; recall at IoU 0.1/0.3/0.5; event-time and boundary error; action completeness; near misses | Measures whether a correct event produces a usable clip at the right time. |
| Determinism | schema pass rate; invalid boundaries; duplicate/overlap rates; budget compliance | Finds malformed provider output and structurally unusable reels. |
| Classification | negative-window specificity; average precision; calibration error; threshold sweep | Measures rejection quality, ranking, and whether confidence is trustworthy. |
| Funnel | proposal recall; classification event recall; selection survival rate | Identifies the exact stage where labeled events are lost. |
| Operations | success rate; latency; provider calls/tokens; estimated spend; cost per TP | Tracks reliability and cost, including cheap but useless executions. |

## Development gates

- Candidate recall: at least 0.80.
- Selected-event precision, recall, and F1: each at least 0.70.
- Classification schema pass rate: 1.00.
- Invalid boundary count: 0; selected overlap rate: 0.
- Total estimated provider spend across the loop: no more than USD 5.00.
- A full first-game run is allowed only after the gates hold on a development slice not used for the immediately preceding fix.
- The holdout game is not used to choose prompts, thresholds, sampling, or matching rules.

Metrics are always stored with the dataset hash, code revision, model/configuration, and the raw TP/FP/FN counts. A metric-definition correction is recorded as such; it never silently replaces the historical result.

`history.jsonl` is append-only and machine-readable. Live pipeline commands require a `--change-note`, and append the complete metric object after writing the immutable per-run report. `spend-ledger.json` is cumulative across processes and is checked before each provider call.
