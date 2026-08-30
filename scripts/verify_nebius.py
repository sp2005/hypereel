"""Verify the live Nebius Token Factory call end-to-end.

The course requires at least one model call routed through Nebius. HypeReel's
providers deliberately *swallow* errors and fall back to mock so the pipeline
never crashes — great for a demo, but it means a bad API key or an
unavailable model fails *silently*. This script is the opposite: it makes a
real Nebius call and prints the RAW error if anything goes wrong, so you can
fix your `.env` before recording the demo.

Usage:
    # 1) put your key in .env:  NEBIUS_API_KEY=...   (and optionally NEBIUS_MODEL=...)
    # 2) install the provider SDK:  pip install -e ".[providers]"
    # 3) run:
    python scripts/verify_nebius.py

Exit code 0 = a live Nebius call succeeded; non-zero = it did not (see the error).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `hypereel` importable whether or not the package was pip-installed.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hypereel.config import get_settings


def main() -> int:
    settings = get_settings()

    print("== Nebius live-call check ==")
    print(f"  base_url : {settings.nebius_base_url}")
    print(f"  model    : {settings.nebius_model}")
    print(f"  api_key  : {'set (' + str(len(settings.nebius_api_key)) + ' chars)' if settings.nebius_api_key else 'MISSING'}")
    print()

    if not settings.nebius_api_key:
        print("FAIL: NEBIUS_API_KEY is not set. Add it to .env (see .env.example).")
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print('FAIL: the `openai` SDK is not installed. Run: pip install -e ".[providers]"')
        return 3

    # Raw call (no error-swallowing) so any failure is visible and diagnosable.
    client = OpenAI(api_key=settings.nebius_api_key, base_url=settings.nebius_base_url)
    try:
        completion = client.chat.completions.create(
            model=settings.nebius_model,
            messages=[
                {"role": "user", "content": "Reply with exactly: HypeReel Nebius check OK"}
            ],
            max_tokens=32,
        )
    except Exception as exc:  # surface the REAL error — the opposite of the provider
        print(f"FAIL: the Nebius call raised:\n  {type(exc).__name__}: {exc}")
        print("\nCommon fixes:")
        print("  * 401/403      -> wrong or expired NEBIUS_API_KEY")
        print("  * 404 model    -> NEBIUS_MODEL is not available; pick a current Nebius model")
        print("  * base_url     -> confirm NEBIUS_BASE_URL ends with /v1")
        return 1

    text = (completion.choices[0].message.content or "").strip()
    print("SUCCESS: Nebius returned a live response:")
    print(f"  {text!r}")
    print("\nYou can now run the graded pipeline through Nebius, e.g.:")
    print("  HYPEREEL_LLM_PROVIDER=nebius python -m hypereel.cli --recipe recipes/basketball_player.yaml --source <url>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
