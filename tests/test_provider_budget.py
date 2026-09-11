import json
from types import SimpleNamespace

import pytest

from hypereel.config import Settings
from hypereel.observability import (
    ProviderBudgetExceeded,
    authorize_provider_call,
    provider_budget_scope,
    record_provider_usage,
)


def _completion(prompt=1000, output=100):
    return SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=output
    ))


def test_provider_budget_persists_and_enforces_cumulative_cap(tmp_path):
    ledger = tmp_path / "spend.json"
    settings = Settings(
        max_provider_spend_usd=0.02,
        provider_call_reserve_usd=0.01,
        nebius_input_cost_per_million_usd=10,
        nebius_output_cost_per_million_usd=30,
        provider_spend_ledger_path=str(ledger),
    )
    with provider_budget_scope(settings):
        authorize_provider_call(provider="nebius", model="m", operation="vision")
        record_provider_usage(_completion())  # $0.013
    assert json.loads(ledger.read_text())["estimated_spend_usd"] == pytest.approx(0.013)

    with provider_budget_scope(settings):
        with pytest.raises(ProviderBudgetExceeded):
            authorize_provider_call(provider="nebius", model="m", operation="vision")
