"""Allow `python -m hypereel.evaluation run ...`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
