# HypeReel golden dataset

This directory is the canonical, versioned development reference set for
HypeReel. It follows the course-kit pattern of keeping pristine expected data
separate from generated experiment outputs.

## Layout

```text
evals/golden/
├── manifest.json                 # version, hashes, counts, limitations
├── golden_events.jsonl           # one normalized gold row per annotated event
├── source/                       # immutable externally labeled source exports
│   ├── east-bay-elite-vs-spartans.external-labels.json
│   └── unlimited-vs-campus.external-labels.json
└── cases/                        # runnable HypeReel pipeline cases
    ├── east-bay-elite-vs-spartans.pipeline.jsonl
    ├── unlimited-vs-campus.pipeline.jsonl
    └── development.pipeline.jsonl
```

`golden_events.jsonl` contains all 462 annotations. Each row retains the original
event and outcome and adds `eligible_highlight` plus the expected HypeReel label.
Misses, free throws, turnovers, assists, and rebounds are explicit negative cases;
made two-pointers, made three-pointers, steals, and blocks are positives.

## Status and limitations

This is the frozen development gold set for Games 1 and 2. It must not contain the
future third-game holdout. The external event labels are source annotations, while the
action boundaries are provisional `event_time - 5s` / `event_time + 4s` windows.
Independent human verification should update the status in a new dataset version,
not silently rewrite this version.

## Rebuild and verify

```bash
python scripts/build_golden_dataset.py
pytest tests/test_external_reference_eval.py
```

## Run the development evaluation

This downloads two videos and may incur provider charges configured in `.env`:

```bash
python -m hypereel.evaluation run --mode pipeline \
  --dataset evals/golden/cases/development.pipeline.jsonl \
  --output evals/results/external-development-v1
```

Never write predictions or scores into this directory. Generated clips, traces,
reports, and metrics belong under `evals/results/` so the gold data remains clean.
