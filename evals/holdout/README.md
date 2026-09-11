# HypeReel final holdout

This directory contains the untouched third-game evaluation set. It is isolated
from `evals/golden/`, which contains the two development games.

## Rules

- Do not use these labels to change prompts, recipes, thresholds, sampling, or code.
- Do not add this case to any development or regression dataset.
- Freeze the program and evaluation protocol before running this case.
- Run it once for the final report, then report all metrics and failures without
  tuning against them.

The source includes all 194 externally labeled events. The runnable case evaluates
35 eligible highlights: made two-pointers, made three-pointers, steals, and blocks.
The remaining 159 events are explicit negatives.

## Final run only

```bash
python -m hypereel.evaluation run --mode pipeline \
  --dataset evals/holdout/cases/final.pipeline.jsonl \
  --output evals/results/final-holdout-v1
```

This command processes the complete video and may incur configured provider costs.
