"""HypeReel — a recipe-driven, agentic highlight-reel builder.

Point it at a video (YouTube URL or local file) and a *recipe* (the declarative
policy for what counts as a highlight), and it plans a detection strategy, finds
candidate moments, confirms them with a vision model, selects the best set to
fit a time budget, and renders a share-ready reel — pausing for human approval
before rendering and before sharing.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
