"""Runtime configuration for HypeReel, loaded from environment / .env.

Nothing here imports a model SDK or a media library — it only reads settings.
Providers are constructed lazily by ``providers.factory`` based on these values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:  # optional; .env is convenient but not required
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    """All runtime knobs. Construct via :func:`get_settings`."""

    vision_provider: str = "mock"
    llm_provider: str = "mock"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    groq_api_key: str = ""
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    nebius_api_key: str = ""
    nebius_base_url: str = "https://api.studio.nebius.com/v1"
    nebius_model: str = "Qwen/Qwen2-VL-72B-Instruct"

    fireworks_api_key: str = ""
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    fireworks_model: str = "accounts/fireworks/models/llama-v3p2-90b-vision-instruct"

    frames_per_candidate: int = 3
    motion_sample_fps: int = 2
    download_dir: str = "downloads"
    output_dir: str = "output"
    memory_path: str = "hypereel_memory.json"

    extra: dict = field(default_factory=dict)

    def key_for(self, provider: str) -> Optional[str]:
        return {
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
            "nebius": self.nebius_api_key,
            "fireworks": self.fireworks_api_key,
            "mock": "mock",
        }.get(provider)


def get_settings() -> Settings:
    """Read a fresh :class:`Settings` from the current environment."""
    return Settings(
        vision_provider=_get("HYPEREEL_VISION_PROVIDER", "mock") or "mock",
        llm_provider=_get("HYPEREEL_LLM_PROVIDER", "mock") or "mock",
        gemini_api_key=_get("GEMINI_API_KEY"),
        gemini_model=_get("GEMINI_MODEL", "gemini-2.0-flash"),
        groq_api_key=_get("GROQ_API_KEY"),
        groq_vision_model=_get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        nebius_api_key=_get("NEBIUS_API_KEY"),
        nebius_base_url=_get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1"),
        nebius_model=_get("NEBIUS_MODEL", "Qwen/Qwen2-VL-72B-Instruct"),
        fireworks_api_key=_get("FIREWORKS_API_KEY"),
        fireworks_base_url=_get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
        fireworks_model=_get(
            "FIREWORKS_MODEL",
            "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
        ),
        frames_per_candidate=_get_int("HYPEREEL_FRAMES_PER_CANDIDATE", 3),
        motion_sample_fps=_get_int("HYPEREEL_MOTION_SAMPLE_FPS", 2),
        download_dir=_get("HYPEREEL_DOWNLOAD_DIR", "downloads"),
        output_dir=_get("HYPEREEL_OUTPUT_DIR", "output"),
        memory_path=_get("HYPEREEL_MEMORY_PATH", "hypereel_memory.json"),
    )
