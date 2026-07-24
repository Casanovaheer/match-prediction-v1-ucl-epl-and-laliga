# Match Prediction — UCL, Premier League, La Liga

Match-by-match probability forecasts for the 2026-27 season, built on free data
with no accounts, no API keys and no scraped bookmaker odds in the model.

Every fixture produces a full scoreline probability matrix. Every other market —
1X2, over/under, both teams to score, correct score, clean sheets — is read off
that one matrix, so the numbers can never contradict each other.

---

## Measured accuracy

This is the number that matters, and it is measured, not claimed. Walk-forward
across 11 seasons, refitting weekly, never letting the model see a result before
predicting it:

| | matches | accuracy | log-loss | Brier | RPS |
|---|---:|---:|---:|---:|---:|
| base rate (do nothing) | 8,360 | 45.0% | 1.0667 | 0.6447 | 0.2290 |
| **this model** | 8,360 | **52.6%** | **0.9845** | **0.5848** | **0.1999** |
| bookmaker closing odds | 8,359 | 54.6% | 0.9586 | 0.5686 | 0.1927 |

The model beats the base rate comfortably and sits **0.0259 log-loss behind the
closing line**. That is the expected, respectable place for a model built on free
data. Closing odds absorb team news, injuries and money that this model cannot
see.

Reproduce it yourself:

```bash
python -m src.backtest --from 2015 --to 2026
```

### Known weaknesses, measured

- **Strong away favourites are overconfident.** Predicted 74% → actual 69%.
- **High draw probabilities run hot.** Predicted 32.5% → actual 28.2%.
- **La Liga correct-score cells 0-0 and 1-1 carry a ~3pp bias.** Dixon-Coles has
  a single `rho`, and its correction moves 0-0 and 1-1 in the *same* direction.
  La Liga needs 0-0 down and 1-1 up, which no value of `rho` can express. The two
  errors cancel, so the draw rate, 1X2 and totals are unaffected — but do not
  trust those two individual scoreline probabilities in Spain.
- **Early season is weakest.** Promoted clubs have almost no data behind them and
  get a league-average prior.

---

## Quick start

```bash
pip install -r requirements.txt

python -m src.collect          # download ~109,000 matches (a few minutes)
python -m src.clean            # normalise and validate
python -m tests.verify_model   # confirm the model behaves

python -m src.predict "Real Madrid" "Barcelona"
```

```
==============================================================
  Real Madrid  v  Barcelona
  La Liga   ratings as of 2026-05-25
==============================================================

  HOME               DRAW               AWAY
  43.4%             23.1%             33.5%
  ###################==========...............

  fair odds        2.3    4.33    2.99
  expected goals   1.67 - 1.44   (total 3.12)

  SCORELINE MATRIX  (rows = Real Madrid, cols = Barcelona)
              0      1      2      3      4      5
     0      4.2    6.7    4.6    2.2    0.8    0.2
     1      7.7 [10.4]    7.7    3.7    1.3    0.4
     2      6.2    9.0    6.5    3.1    1.1    0.3
     3      3.5    5.0    3.6    1.7    0.6    0.2

  MARKETS
    over 1.5  81.5%     over 2.5  60.2%     over 3.5  37.9%
    BTTS      61.8%     CS home   23.6%     CS away   18.8%
```

### Other commands

```bash
python -m src.predict --round E0              # every unplayed fixture
python -m src.predict "Man City" Arsenal --json
python -m src.simulate --comp SP1             # season projection
python -m src.simulate --comp SP1 --as-of 2026-01-01   # replay a past projection
python -m src.ucl --sims 2000                 # Champions League
python -m src.report                          # build the published site
```

---

## How it works

```
football-data.co.uk ─┐
Understat / FBref    ├─→ collect ─→ clean ─→ RATINGS ─→ DIXON-COLES ─→ MATRIX ─┬─→ match cards
ClubElo              │                       (attack,   (low-score    (0-0 to  │
openfootball        ─┘                        defence)   correction)    6-6)   ├─→ MONTE CARLO ─→ tables
                                                                               └─→ ClubElo bridge ─→ UCL
```

**1. Ratings.** Each club gets an attack rating (goals created) and a defence
rating (goals conceded), both centred on the league average, plus a league-wide
intercept and home advantage. Recent matches count for more — weight halves every
180 days — which is what lets the model track form and new managers.

**2. Dixon-Coles.** Attack meets defence to give each side an expected goal count,
then the low-score correction fixes plain Poisson's well-known failure to predict
enough 0-0 and 1-1 results.

**3. The matrix.** Probability of every scoreline from 0-0 to 10-10. Everything
else is addition over cells.

**4. Monte Carlo.** Play the remaining fixtures 10,000 times, count how often each
club finishes where. Same method the Opta Supercomputer uses.

**5. Champions League.** Domestic ratings are *not* comparable across countries —
nothing in 109,000 domestic matches says whether the Eredivisie is stronger than
the Bundesliga. So the UCL layer bridges leagues with ClubElo, which is built from
actual cross-border results, calibrated to goals on 2,626 domestic matches
(0.196 goals per 100 Elo).

---

## Data

Everything is free and needs no account.

| What | Source | Coverage |
|---|---|---|
| Results + closing odds | football-data.co.uk | 11 leagues, 1993 → today |
| Club strength | ClubElo API | all European clubs |
| Fixtures | derived from the round-robin | — |

**109,318 matches** after cleaning, 502 clubs, 78.6% with market odds attached.

### Data defects this pipeline catches

Source data is not clean, and the failures are quiet ones:

- **Name typos that split a club's history in two.** `Villareal`/`Villarreal`,
  `M'Gladbach`/`M'gladbach`, two Greek transliterations, and 10 names with
  trailing whitespace. Left alone, a club's record splits and both halves get
  mediocre ratings.
- **Look-alikes that must *not* be merged.** `Reggiana` vs `Reggina` are different
  clubs (Reggio Emilia and Reggio Calabria); so are `Athinaikos` and
  `Panathinaikos`. An automatic fuzzy matcher destroys these, so the merge list is
  curated by hand and the fuzzy check runs only as a standing warning.
- **Impossible odds.** Four rows priced at under 100%, which would be a free
  arbitrage. The results are kept for training; only the bad prices are voided.

`python -m src.clean` exits non-zero if any check fails, so bad data cannot reach
a published prediction.

---

## Automation

`.github/workflows/weekly.yml` runs every Friday at 06:00 UTC on GitHub's free
runners:

collect → clean → verify → simulate → report → commit

Predictions are committed to `output/` and `docs/`. **That git history is the
point**: it proves each prediction existed before the match was played and was not
edited afterwards. `data/` is gitignored because it is fully reproducible and
would add roughly a gigabyte of history per year.

No Vercel and no Supabase needed. GitHub Actions is the scheduler, the repo is the
database, GitHub Pages is the host.

---

## Verification

```bash
python -m tests.verify_model
```

Checks maths (matrix sums to 1, markets are mutually consistent, monotonic
over/under ladders), **absence of look-ahead leakage**, calibration against
observed rates, and football plausibility.

That last category matters. An earlier version of this model passed every
mathematical check while ranking two relegated clubs top of the Premier League —
the defence sign was inverted. Weak plausibility tests are how that survives, so
the tests now assert that Real Madrid *and* Barcelona are both top-4 in La Liga
and that no recently relegated side rates in a top five.

The analytic likelihood gradient is checked against finite differences on every
run: identical log-likelihood and identical predicted probabilities, 40x faster.

---

## Honest limits

- **~52-56% on match results is the ceiling for this class of model.** Football is
  genuinely that noisy.
- **Matching the closing line is a good result. Beating it consistently on free
  data would be extraordinary.** Treat any claim above 70% accuracy as false.
- **Season projections are far more reliable than single matches.** 380 matches
  average out; one does not.
- **This is not betting advice.** It outputs probabilities. A 70% favourite loses
  three times in ten — that is the model working, not failing.

## Licence

MIT for the code. The underlying data belongs to its providers; football-data.co.uk
and ClubElo are free for personal use — check their terms before commercial use.
