# Nebius iteration log

Provider model: `openbmb/MiniCPM-V-4_5`. Dollar estimates use deliberately conservative configured token rates because a verified model-specific rate was unavailable. Every paid run was capped at five provider calls. Results below are development data only.

| Iteration | Slice / change | Candidate recall | Precision | Recall | F1 | Schema pass | TP/FP/FN | Estimated spend | Cumulative |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | Opening baseline; first four chronological candidates | N/A* | 0.00 | N/A | N/A | 0.75 | 0/2/0 | $0.05452 | $0.05452 |
| 2 | 60-100s positive slice; three frames | 0.00** | 0.00 | 0.00 | N/A | 1.00 | 0/2/1 | $0.05200 | $0.10652 |
| 3 | Added four-second context and five frames | 1.00** | 0.00 | 0.00 | N/A | 0.50 | 0/1/1 | $0.07806 | $0.18458 |
| 4 | Six-second context, nine frames, temporal basketball rubric, tolerant JSON | 1.00 | 0.00*** | 0.00*** | N/A | 1.00 | 0/2/1*** | $0.10725 | $0.29183 |
| 5 | Independent 145-190s slice with two events | 1.00 | 0.00 | 0.00 | N/A | 1.00 | 0/2/2 | $0.12119 | $0.41302 |
| 6 | Spread sampling plus shot-evidence guard and null-clip filter | 1.00 | 1.00 | 0.50 | 0.667 | 1.00 | 1/0/1 | $0.11207 | $0.52509 |
| 7 | Different-lighting game; spread sampling | 1.00 | 0.00 | 0.00 | N/A | 1.00 | 0/3/2 | $0.10672 | $0.63181 |
| 8 | Reference-stratified diagnostic sampling | 1.00 | 0.333 | 0.50 | 0.40 | 1.00 | 1/2/1 | $0.10614 | $0.73795 |
| 9 | Dense central-action frames | 1.00 | 0.333 | 0.50 | 0.40 | 1.00 | 1/2/1 | $0.10833 | $0.84628 |
| 10 | Reduced temporal context | 1.00 | 0.333 | 0.50 | 0.40 | 1.00 | 1/2/1 | $0.10954 | $0.95582 |
| 11 | Explicit before/action/after frame roles | 1.00 | 0.50 | 0.50 | 0.50 | 1.00 | 1/1/1 | $0.11358 | $1.06940 |
| 12 | Maximum-IoU diagnostic candidate alignment | 1.00 | 0.00 | 0.00 | N/A | 1.00 | 0/1/2 | $0.11040 | $1.17980 |
| 13 | Three-second proposal padding; exposed overly permissive matching† | 1.00 | 1.00† | 1.00† | 1.00† | 1.00 | 2/0/0† | $0.09189 | $1.27169 |
| 14 | IoU-aware primary match correction | 1.00 | N/A | 0.00 | N/A | 1.00 | 0/0/2 | $0.09085 | $1.36254 |
| 15 | Deterministic temperature; padded candidates | 1.00 | 0.667 | 1.00 | 0.80 | 1.00 | 2/1/0 | $0.09316 | $1.45570 |
| 16 | Core-frame verification | 1.00 | 0.667 | 1.00 | 0.80 | 1.00 | 2/1/0 | $0.15164 | $1.60734 |
| 17 | Explicit rebound/miss/turnover negatives | 1.00 | 0.667 | 1.00 | 0.80 | 1.00 | 2/1/0 | $0.15578 | $1.76312 |
| 18 | Centered clips plus contrastive verification | 1.00 | 0.667 | 1.00 | 0.80 | 1.00 | 2/1/0 | $0.15299 | $1.91611 |
| 19 | Fresh slice; revealed boundary and multi-event ambiguity | 0.667 | 0.50 | 0.333 | 0.40 | 1.00 | 1/1/2 | $0.20560 | $2.12171 |
| 20 | Clean slice on the other development game | 1.00 | 1.00 | 0.50 | 0.667 | 1.00 | 1/0/1 | $0.20528 | $2.32699 |
| 21 | Alternative-label verifier experiment (rejected) | 1.00 | N/A | 0.00 | N/A | 1.00 | 0/0/2 | $0.20328 | $2.53027 |

\* Iteration 1's capped horizon contained no reference positives, so positive-class recall is not applicable. Its original full-dataset candidate-recall value of 0 was invalid for a capped run.

\** Iterations 2 and 3 motivated the candidate-recall correction: a local motion window overlapped the labeled action interval but did not contain its single anchor timestamp. The current metric uses action-interval overlap. The table preserves the result available at each iteration and flags the definition change.

\*** Iteration 4 visibly produced a correctly labeled steal clip, but its historical report used anchor containment and also admitted an unclassified clip. These defects were fixed after the run; the historical numbers are retained rather than rewritten.

† Iteration 13 used any-overlap primary matching. Its mean temporal IoU was only 0.059 and recall at IoU 0.30 was zero, so the apparent perfect result was invalid. Iteration 14 corrected the definition; history remains unchanged.

## Decision

The best tuned-slice result is iteration 18 (precision 0.667, recall 1.00, F1 0.80, IoU-0.30 recall 1.00). The clean independent slice in iteration 20 has perfect precision but only 0.50 recall and 0.667 F1, so the release gate is not met. Iteration 21 was rejected and its code change reverted. The Nebius account exposes only `openbmb/MiniCPM-V-4_5`; repeated temperature-zero runs still changed semantic verdicts. A full-video paid run or sealed-holdout run is therefore not authorized. The durable next step is a stronger video-capable provider/model or a multi-label temporal event schema, not further timestamp-specific prompt tuning.
