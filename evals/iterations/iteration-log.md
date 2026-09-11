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

\* Iteration 1's capped horizon contained no reference positives, so positive-class recall is not applicable. Its original full-dataset candidate-recall value of 0 was invalid for a capped run.

\** Iterations 2 and 3 motivated the candidate-recall correction: a local motion window overlapped the labeled action interval but did not contain its single anchor timestamp. The current metric uses action-interval overlap. The table preserves the result available at each iteration and flags the definition change.

\*** Iteration 4 visibly produced a correctly labeled steal clip, but its historical report used anchor containment and also admitted an unclassified clip. These defects were fixed after the run; the historical numbers are retained rather than rewritten.

## Decision

Iteration 6 passes candidate recall, precision, schema, and structural gates, but fails the 0.70 recall and F1 gates. No full-video paid run or holdout run is authorized by the evaluation policy yet. The next development experiment should target proposal-to-classification alignment or a stronger video-capable model, using a fresh slice from development game 2 rather than further tuning on these same three steals.
