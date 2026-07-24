# CLAUDE.md — instructions for this project

This is a football match-prediction engine (UCL, Premier League, La Liga).
The owner, Jamil, is **not a Python user**. Read this before doing anything.

## The default request: "Team vs Team"

When Jamil writes just a fixture — `Arsenal vs Chelsea`, `Real Madrid Barcelona`,
`Degerfors v Djurgården`, in any spelling or separator — he wants **ranked top
picks**, not a probability dump. This is the single most common thing he asks
for. Do it without being asked twice.

**Produce it by running:**

```bash
python -m src.toppick "Home team" "Away team"
```

Home team goes first. The command auto-selects the engine and prints the whole
answer. Present that output. The required shape (which `toppick` already
produces) is:

1. **One headline top pick** — the highest-confidence market, stated plainly.
2. **A ranked table** — every market with pick, probability, fair odds, and a
   confidence tier: STRONG / solid / lean / COIN FLIP.
3. **Correct score + winning margin** breakdown.
4. **A "do not bet these" list** — every market within 3 points of 50/50.
5. **The engine line** — which model produced it (see below).

## Always name the engine. This is not optional.

There are two engines, and they are **not** equally trustworthy:

| Engine | Used for | Accuracy |
|---|---|---|
| **Dixon-Coles** | EPL + La Liga (the 11 collected leagues) | **52.6% measured** over 8,360 out-of-sample matches |
| **Elo bridge** | every other European club, via ClubElo | **none measured** — a borrowed approximation |

`toppick` chooses automatically and labels its output. Never present an
Elo-bridge result as if it carried the measured EPL/La Liga accuracy. If Jamil
asks about a Swedish, Norwegian, Brazilian, or MLS club, it is the Elo bridge —
say so plainly and tell him the numbers are rough.

## Never dress up a coin flip as a pick

Any market within 3pp of even money is noise, not information. `toppick` already
labels these COIN FLIP and lists them under "do not bet these". Keep that
framing when you relay the output. Reporting the argmax of a 50/50 as a
"prediction" is worse than saying nothing.

## Honesty rules that must survive every session

- Realistic accuracy for this class of model is **~52-56%** on match results.
  Anyone claiming 70%+ is lying. Do not soften this.
- These are **probabilities, not tips**. A 70% favourite loses three times in
  ten — that is the model working, not failing.
- The model has **no team news** (injuries, suspensions, rotation). Say so when
  it matters, especially mid-season.
- Season projections are far more reliable than single matches.

## Jamil is not a Python user

Every feature needs a double-click entry point, not a terminal command. The
numbered `.bat` launchers in the project root are the interface:

- `1 - SHOW RESULTS.bat` — opens the predictions page in a browser
- `2 - UPDATE EVERYTHING.bat` — downloads latest results, rebuilds everything
- `3 - PREDICT ONE MATCH.bat` — full match card for one fixture
- `4 - TOP PICKS.bat` — ranked top picks for one fixture (the main one)

When adding a feature, add its launcher **in the same change**. A terminal
command may be offered as the advanced path, but the double-click route has to
exist first.

## Pipeline (for when real work is needed)

```
src/collect.py   download results (football-data.co.uk) + Elo (ClubElo), free
src/clean.py     normalise + validate; exits non-zero if data is bad
src/model.py     time-weighted Dixon-Coles, analytic gradient
src/backtest.py  walk-forward accuracy measurement
src/simulate.py  Monte Carlo season projection
src/ucl.py       Champions League via the ClubElo cross-league bridge
src/predict.py   full match cards
src/toppick.py   ranked top picks   <- the default request
src/report.py    builds docs/index.html for the browser
tests/           verify_model.py, verify_pipeline.py — run after any model change
```

Data lives in `data/` and is gitignored (reproducible). Predictions in
`output/` and `docs/` are committed on purpose — the git history proves a
prediction existed before kickoff and was not edited after.

## The season is not live yet

EPL and La Liga 2026-27 kick off around mid-August 2026. Until then, fixture
lists are empty (correct, not a bug), and `toppick` on those clubs uses the last
completed season's ratings. Off-season European leagues (e.g. Allsvenskan) work
today, but only through the weaker Elo bridge.
