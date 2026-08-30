"""Resolve a user-supplied source (local path or URL) into a local video file.

Isolates the messy, flaky part of the pipeline — network downloads and
missing local files — behind one call so the rest of the system only ever
deals with a local path + duration. Failures are always wrapped in
IngestError; a raw yt_dlp/ffmpeg exception must never escape this module.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


class IngestError(Exception):
    """Raised when a source cannot be resolved to a usable local video file."""


def is_url(source: str) -> bool:
    """True for http(s):// URLs (YouTube, direct links, etc.)."""
    if not source:
        return False
    try:
        parsed = urlparse(source)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https")


# Map free-text quality intent (from the user's prompt) to a max video height.
# Specific resolutions are checked before the generic "HD" catch-all so
# "720p" wins over the bare "hd" in the same sentence. ``None`` = no explicit
# preference (caller uses its default ceiling).
_QUALITY_KEYWORDS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (1080, ("1080", "fullhd", "full hd", "full-hd", "fhd")),
    (720, ("720",)),
    (480, ("480",)),
    (360, ("360", "low res", "low-res", "low quality", "low-quality",
           "lowest", "fast preview", "small")),
)
_HD_KEYWORDS = (
    "high definition", "high-definition", "high quality", "high-quality",
    "high resolution", "high-res", "hi-res", "best quality", "hd",
)


def parse_video_quality(text: Optional[str]) -> Optional[int]:
    """Extract a max video height (px) from a free-text prompt, or None.

    Lets a user fold output-quality intent into the same natural-language brief
    that describes the subject — e.g. "...prefer #23; use HD if available" ->
    1080, or "quick 360p preview" -> 360. Deterministic keyword match (no LLM),
    so it never fails and is trivially testable.
    """
    if not text:
        return None
    t = text.lower()
    for height, keys in _QUALITY_KEYWORDS:
        if any(k in t for k in keys):
            return height
    if any(k in t for k in _HD_KEYWORDS):
        return 1080  # generic "HD" -> best available up to 1080p
    return None


def _format_for_height(max_height: Optional[int]) -> str:
    """yt-dlp format string preferring a merged video+audio pair up to ``max_height``."""
    h = max_height or 1080
    return (
        f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/"
        f"bestvideo[height<={h}]+bestaudio/"
        f"best[height<={h}]/best"
    )


def _base_ydl_opts(download_dir: Path, max_height: Optional[int]) -> dict:
    """Fast default download options — no browser cookies, no JS runtime."""
    return {
        "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
        # Prefer a high-res video+audio pair merged by ffmpeg over a single
        # progressive MP4 — YouTube caps progressive ("mp4/best") at ~360p,
        # which is why early reels looked soft. Falls back to the best
        # progressive stream so a missing ffmpeg never breaks the download.
        "format": _format_for_height(max_height),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        # YouTube increasingly rejects yt-dlp's default web client with a bogus
        # "video is not available" — falling back through android/tv/safari
        # clients resolves videos the default can't.
        "extractor_args": {
            "youtube": {"player_client": ["android", "web_safari", "tv", "web"]}
        },
    }


def _hd_unlock_ydl_opts(base: dict) -> dict:
    """Add the SABR-defeating options that unlock HD (DASH) formats.

    Recent YouTube gates its HD streams behind SABR; getting past it needs a JS
    runtime (deno) plus remotely-fetched EJS components, and browser cookies to
    look like a signed-in web client. These are layered on top of ``base`` as a
    *first* attempt — if deno/Chrome aren't available the attempt fails fast and
    the caller falls back to ``base`` (which still gets progressive/≤360p), so a
    machine without the workaround degrades instead of erroring out.

    Unknown option keys are simply ignored by older yt-dlp builds, so this is
    safe to pass unconditionally.
    """
    opts = dict(base)
    opts["extractor_args"] = {"youtube": {"player_client": ["web_safari", "web", "tv"]}}
    # Canonical shapes (verified against yt_dlp.parse_options for the working CLI
    # flags: --cookies-from-browser chrome / --js-runtimes deno / --remote-components):
    opts["cookiesfrombrowser"] = ("chrome", None, None, None)
    opts["js_runtimes"] = {"deno": {"path": None}}
    opts["remote_components"] = ["ejs:github"]
    return opts


def probe_duration(video_path: str) -> float:
    """Best-effort media duration in seconds; 0.0 if it can't be determined."""
    # Try ffprobe first — a plain subprocess call, no python media deps needed.
    try:
        import subprocess

        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return max(0.0, float(result.stdout.strip()))
    except Exception:
        pass

    # Fall back to opencv if it happens to be installed.
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if fps > 0:
            return max(0.0, frames / fps)
    except Exception:
        pass

    return 0.0


def _resolve_downloaded_path(prepared: str, info: dict, download_dir: Path) -> Optional[str]:
    """Find the real file yt_dlp wrote, accounting for a post-download merge.

    When video+audio are merged, ``prepare_filename`` can still report the
    pre-merge name/extension while the actual output is ``<id>.mp4``. Prefer the
    definitive ``requested_downloads[*].filepath`` yt_dlp records, then the
    prepared path, then any ``<id>.*`` file in the download dir.
    """
    for dl in info.get("requested_downloads") or []:
        fp = dl.get("filepath")
        if fp and os.path.exists(fp):
            return fp
    if prepared and os.path.exists(prepared):
        return prepared
    # Last resort: swap the prepared extension for the merged container, then
    # fall back to globbing the video id.
    merged = os.path.splitext(prepared)[0] + ".mp4"
    if os.path.exists(merged):
        return merged
    vid = info.get("id")
    if vid:
        matches = sorted(download_dir.glob(f"{vid}.*"))
        if matches:
            return str(matches[0])
    return prepared or None


def resolve_source(
    source: str,
    settings,
    *,
    max_retries: int = 2,
    max_height: Optional[int] = None,
) -> tuple[str, float]:
    """Return (local_video_path, duration_seconds) for ``source``.

    Local paths are validated and probed directly. URLs are downloaded via
    yt_dlp into ``settings.download_dir``. ``max_height`` caps the resolution
    (from a prompt's quality intent; ``None`` = up to 1080p). When HD is wanted
    the download first tries the SABR-unlock profile (deno + remote EJS
    components + Chrome cookies) and, if that machine can't do it, falls back to
    the plain profile — so HD works where possible and never blocks a download
    where it isn't. Anything that goes wrong becomes an ``IngestError``; the
    rest of the pipeline never sees a raw yt_dlp/ffmpeg exception.
    """
    if not source:
        raise IngestError("no source provided")

    if not is_url(source):
        p = Path(source)
        if not p.exists() or not p.is_file():
            raise IngestError(f"local source not found: {source}")
        return str(p), probe_duration(str(p))

    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise IngestError("yt_dlp is not installed; cannot download URL sources") from exc

    download_dir = Path(getattr(settings, "download_dir", "downloads"))
    download_dir.mkdir(parents=True, exist_ok=True)

    base = _base_ydl_opts(download_dir, max_height)
    # HD is SABR-gated on modern YouTube; try the unlock profile first when a
    # high resolution is wanted, then the plain profile on later attempts so a
    # missing deno/Chrome degrades to ≤360p instead of failing the whole run.
    want_hd = max_height is None or max_height >= 720
    profiles = [_hd_unlock_ydl_opts(base), base] if want_hd else [base]

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        ydl_opts = profiles[min(attempt, len(profiles) - 1)]
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=True)
                path = ydl.prepare_filename(info)
            path = _resolve_downloaded_path(path, info, download_dir)
            if not path or not os.path.exists(path):
                raise IngestError(f"download reported success but file missing: {path}")
            duration = float(info.get("duration") or 0.0) or probe_duration(path)
            return path, duration
        except Exception as exc:  # yt_dlp/network errors must never escape this module
            last_exc = exc
            if attempt < max_retries:
                time.sleep(min(2**attempt, 10))
            continue

    raise IngestError(
        f"failed to download {source} after {max_retries + 1} attempt(s): {last_exc}"
    ) from last_exc
