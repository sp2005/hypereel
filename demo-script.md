# HypeReel — Demo Script (≤5 min)

Format: **[SCREEN]** cues are what to show; the plain text is what you say.
The spoken track runs ~4 minutes at a normal pace. Bracketed *(director notes)*
are for you, not to be read aloud.

## Recording plan (how it fits in 5 minutes)

A full live run takes 15 to 25 minutes, so don't record it uncut. Before you
record:
- Do a full run off-camera and keep the finished reel plus the completed panel
  (clip list, judge score, timings).
- Optionally do a second run with the quick-test cap set to ~40 windows so you
  have a shorter run to trigger on camera.

While recording, trigger a run to show the pipeline light up, then **time-lapse
or cut** through the long classify phase and jump to the finished reel you
already have. "Live" here means showing it work end to end, not real time.
Keep the segment budget roughly: intro + recipe 1:40, architecture + MINT 1:05,
live run 1:15, AI tools 0:40, close 0:10. That sums to ~4:50.

---

## 0:00 — What it is (authority context first)

**[SCREEN: HypeReel app open, a game thumbnail loaded]**

I built HypeReel, an agentic app that turns a full game recording into a
share-ready highlight reel. The flagship demo is a basketball game. A player
uploads their 60-minute game, describes what they want in plain language, and
gets back a short reel of the plays that matter, without the 3 to 4 hours of
scrubbing and clip-cutting it takes in a video editor today.

---

## 0:30 — The recipe abstraction

**[SCREEN: open `recipes/basketball_generic.yaml` briefly, then back]**

While the flagship demo I built was focused on a basketball game, the high-level
idea is to maintain an abstraction, and to maintain this abstraction I introduced
the notion of a recipe. You can think of it like a plug-in. A recipe is where
someone defines the key moments they are looking for in a highlight.

The recipe is a declarative YAML policy. It describes what counts as a highlight
for a given domain, so this pipeline can be used beyond basketball. A recipe can
be event-based. Basketball is the example: you look for discrete events, where a
vision model classifies each candidate window against a short list of named
moment types like made basket, three pointer, block, and steal break. Or it can
be quality-based. Take a real estate walkthrough. There are no discrete events to
detect there. Instead the vision model scores sliding windows against a
composition, lighting, and clarity rubric, and the sustained highest-scoring
segments become the clips.

---

## 1:15 — The architecture, and what makes it agentic

**[SCREEN: the "Under the hood" pipeline diagram in the app]**

Under the hood, HypeReel is a controllable, stateful, flow-based application with
cycles and human interrupts, built on LangGraph. The orchestration is a
deterministic LangGraph workflow — and that's deliberate. I added agentic
behavior only where it earns its cost.

There are two places where it earns it. The first is a cycle driven by LLM
judgment. Once the reel is assembled, an LLM judge scores it against the recipe's
acceptance criteria and can send it back for one revision, with stricter or
broader selection. The second is the human. The analysis steps run on their own,
and every write waits for a human: one gate to approve the clip list before
anything renders, and a second gate to approve before it shares.

---

## 2:15 — Where I stopped, and why (MINT)

I applied the MINT framework and stopped with HypeReel deployed only for a
basketball player. I didn't build a multi-agent system or a generic any-video
tool, because a single graph with a judge loop already handles the variance. The
recipe abstraction is what lets this extend to any kind of reel later, when there
is a reason to.

---

## 2:40 — Live run

**[SCREEN: pick source game → recipe `basketball_generic` → type the brief]**

Let me run it. I'll point it at a game, pick the generic basketball recipe, and
describe the team I want: the black and blue jerseys, and exclude the yellow
opponent.

**[SCREEN: pre-flight check / provider badge]**

Before it runs, there's a pre-flight check that pings the provider with a real
image, so a dead or out-of-budget provider fails right here instead of silently
returning zero clips. I hit exactly that failure earlier in the build, so I made
it loud.

**[SCREEN: click Find highlights; nodes light up in sequence, then time-lapse the classify phase]**

As it runs, this panel shows the real LangGraph nodes lighting up: ingest,
propose candidate windows, read the scoreboard, classify each window with the
vision model, then select to the time budget. Classification is the slow part,
so I'll speed through it.

**[SCREEN: point at the Judge node / the revise branch and the score change]**

Here is the part I want to show. The judge scored the first cut low and sent it
back. It found too much padding, re-selected for confirmed makes, and the score
went up. That is the system critiquing and improving its own output.

**[SCREEN: Gate 1 — the proposed clip list]**

Now it pauses at the first human gate. I can see every proposed clip and drop
anything that doesn't belong. I approve.

**[SCREEN: render → summary → Gate 2 → play the final reel]**

It renders, writes a short summary, and pauses again before sharing. I approve
the share, and here is the final reel.

---

## 4:00 — How I used AI coding tools

*(this is drawn from the actual build; trim to what you're comfortable saying)*

I built this with Claude Code on the LangChain and LangGraph track, using
subagents to work on parts in parallel. It scaffolded the graph quickly, but the
value was in the hard parts I iterated on: the OCR-free scoreboard reader, the
judge's revision logic, and provider error handling.

One example worth calling out. An early run came back with zero clips, and the
first read was that my prompt was wrong. I wrote a small probe script to test one
window against the live model, and it showed the real cause: the provider had hit
its budget and every call was failing silently. So I made that failure loud, with
a pre-flight check, instead of guessing at the prompt. I don't take the agent's
first explanation as the answer; I verify it.

---

## 4:45 — Close (useful, not a mic drop)

That's HypeReel. The recipe is the piece I would reuse. Point it at a different
domain, write a new recipe, and the same pipeline builds a different reel.
