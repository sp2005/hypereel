# Samples

`sample_game.mp4` (not checked in) is a tiny synthetic clip — a moving
colored rectangle over a drifting background with a couple of brighter/faster
"spike" segments — used for offline demos and manual testing of the
audio/motion signal proposers without a real game recording.

Generate it with:

```
python samples/generate_sample.py
```

This needs `numpy` + `opencv-python`; if they're not installed the script
prints a message and exits cleanly. The full HypeReel pipeline also runs
end-to-end in mock mode with **no sample video required** — you only need
this file for a real (non-mock) local demo.
