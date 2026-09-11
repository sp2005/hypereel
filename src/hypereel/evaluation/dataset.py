"""Load the entire dataset before executing any case."""
from pathlib import Path
from .schemas import SelectionCase


def load_dataset(path: str | Path, case_model=SelectionCase) -> list:
    path = Path(path)
    cases = []
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = case_model.model_validate_json(line)
            if case.case_id in seen:
                raise ValueError(f"duplicate case_id: {case.case_id}")
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: dataset is empty")
    return cases
