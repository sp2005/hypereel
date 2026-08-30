"""Shared pytest fixtures for HypeReel.

Everything here is offline: no network, no API keys, no heavy media libs
required. Tests import the package via ``pythonpath=["src"]`` (see pyproject).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _force_mock_providers(monkeypatch):
    """Keep the suite hermetic regardless of the developer's .env.

    Several node tests assert on the *mock* provider's deterministic output. A
    real .env may set HYPEREEL_LLM_PROVIDER=nebius (for live runs), which would
    otherwise leak into tests. OS env vars outrank .env in pydantic-settings, so
    forcing them here pins providers to mock. Tests that need a specific LLM
    verdict monkeypatch get_llm_provider directly and are unaffected.
    """
    monkeypatch.setenv("HYPEREEL_VISION_PROVIDER", "mock")
    monkeypatch.setenv("HYPEREEL_LLM_PROVIDER", "mock")


from hypereel.models import (
    CandidateWindow,
    Classification,
    MomentType,
    Recipe,
    SelectionPolicy,
    SignalConfig,
    SubjectSelector,
)


@pytest.fixture
def basketball_recipe() -> Recipe:
    """A minimal but valid event_based recipe used across tests."""
    return Recipe(
        id="bball_test",
        name="Test Basketball",
        domain="basketball",
        highlight_model="event_based",
        moment_types=[
            MomentType(name="made_basket", description="player scores", weight=1.0, ideal_len=8),
            MomentType(name="block", description="player blocks a shot", weight=0.9, ideal_len=6),
            MomentType(name="steal_break", description="steal into fast break", weight=0.8, ideal_len=10),
        ],
        signals=[
            SignalConfig(type="audio_peak", role="proposer", weight=0.4),
            SignalConfig(type="motion_intensity", role="proposer", weight=0.3),
            SignalConfig(type="vision_classify", role="scorer", weight=0.3),
        ],
        subject_selector=SubjectSelector(
            type="jersey_number", value=23, team_color="red", audience="individual"
        ),
        selection=SelectionPolicy(max_duration=60, min_clip=4, max_clip=12, min_score=0.5),
    )


@pytest.fixture
def sample_candidates() -> list[CandidateWindow]:
    """Twelve evenly spaced candidate windows spanning ~2 minutes."""
    return [
        CandidateWindow(start=float(i * 10), end=float(i * 10 + 8), signal_scores={"audio_peak": 0.5 + 0.03 * i})
        for i in range(12)
    ]
