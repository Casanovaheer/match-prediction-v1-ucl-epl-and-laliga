"""Stage 6 - Champions League.

The domestic model cannot touch this competition. Its attack and defence
ratings are only comparable *within* a league, because clubs in our data have
never played anyone outside their own division: nothing in 109,000 domestic
matches says whether the Eredivisie is stronger than the Bundesliga.

So the UCL layer uses a different bridge - ClubElo, which is built from actual
cross-border results and is therefore comparable across countries. Elo
differences are converted into expected goals by calibrating against domestic
matches, and from there the same Dixon-Coles scoreline machinery applies.

Two honest caveats:

  * The 2026-27 league-phase draw is not made yet. Rather than invent one
    fixed schedule, each simulation performs its own random pot-respecting
    draw. That is the correct way to represent a draw that has not happened:
    the uncertainty of the draw becomes part of the output.

  * The knockout bracket is simulated with the same per-match model. Two-leg
    ties are played as two matches, with away goals no longer used as a
    tiebreak (abolished from 2021-22); level ties are settled by a coin flip
    standing in for extra time and penalties.

    python -m src.ucl --teams config/ucl_2026_27.json
"""

from __future__ import annotations

import argparse
import io
import json

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.stats import poisson

from . import config as cfg
from .model import load_matches

# ClubElo names that differ from football-data.co.uk names. Exact matches are
# resolved automatically; only genuine disagreements are listed here.
ELO_NAME_MAP: dict[str, str] = {
    "Ath Madrid": "Atletico",
    "Ath Bilbao": "Athletic",
    "Vallecano": "Rayo Vallecano",
    "Sociedad": "Real Sociedad",
    "Espanol": "Espanyol",
    "La Coruna": "Depor",
    "Sp Gijon": "Sporting Gijon",
    "Bayern Munich": "Bayern",
    "Dortmund": "Dortmund",
    "Ein Frankfurt": "Frankfurt",
    "M'gladbach": "Gladbach",
    "Nott'm Forest": "Forest",
    "Sheffield United": "Sheffield United",
    "Man United": "Man United",
    "Man City": "Man City",
    "Paris SG": "Paris SG",
}


def fetch_elo(date: str) -> pd.DataFrame:
    """All clubs' Elo on a given date. One call, ~600 clubs, no account."""
    r = requests.get(f"{cfg.CLUBELO_API}/{date}", timeout=45)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df.dropna(subset=["Elo"])


def build_elo_lookup(elo: pd.DataFrame, teams: list[str]) -> dict[str, float]:
    """Map our team names onto ClubElo ratings. Reports what it could not find."""
    by_name = dict(zip(elo["Club"], elo["Elo"]))
    out: dict[str, float] = {}
    for t in teams:
        key = ELO_NAME_MAP.get(t, t)
        if key in by_name:
            out[t] = float(by_name[key])
    return out


def calibrate_elo_to_goals(
    matches: pd.DataFrame, elo_lookup: dict[str, float], season: str
) -> dict:
    """Fit expected goals as a function of Elo difference.

    Uses domestic matches from one season, where both the result and both
    clubs' Elo are known. Three parameters only - an intercept, a slope on the
    Elo gap, and home advantage - so it generalises across leagues instead of
    memorising them.
    """
    df = matches[matches["season"] == season].copy()
    df = df[df["home"].isin(elo_lookup) & df["away"].isin(elo_lookup)]
    if len(df) < 200:
        raise ValueError(f"only {len(df)} calibration matches - need 200+")

    eh = df["home"].map(elo_lookup).to_numpy(float)
    ea = df["away"].map(elo_lookup).to_numpy(float)
    gap = (eh - ea) / 100.0
    hg = df["hg"].to_numpy(int)
    ag = df["ag"].to_numpy(int)

    def nll(p):
        icept, slope, hadv = p
        lam = np.clip(np.exp(icept + slope * gap + hadv), 1e-9, 25)
        mu = np.clip(np.exp(icept - slope * gap), 1e-9, 25)
        return -np.sum(hg * np.log(lam) - lam + ag * np.log(mu) - mu)

    res = minimize(nll, [0.1, 0.15, 0.25], method="L-BFGS-B",
                   bounds=[(-2, 2), (0.0, 1.0), (-0.5, 1.0)])
    icept, slope, hadv = res.x
    return {
        "intercept": float(icept),
        "slope_per_100_elo": float(slope),
        "home_adv": float(hadv),
        "n_calibration_matches": int(len(df)),
        "converged": bool(res.success),
    }


def elo_score_matrix(
    elo_h: float, elo_a: float, cal: dict, neutral: bool = False, rho: float = -0.05
) -> np.ndarray:
    """Scoreline matrix from two Elo ratings."""
    gap = (elo_h - elo_a) / 100.0
    hadv = 0.0 if neutral else cal["home_adv"]
    lam = min(np.exp(cal["intercept"] + cal["slope_per_100_elo"] * gap + hadv), 25)
    mu = min(np.exp(cal["intercept"] - cal["slope_per_100_elo"] * gap), 25)

    n = cfg.MAX_GOALS + 1
    m = np.outer(poisson.pmf(np.arange(n), lam), poisson.pmf(np.arange(n), mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def _sample(mat: np.ndarray, rng, size: int = 1):
    n = cfg.MAX_GOALS + 1
    flat = mat.ravel() / mat.sum()
    d = rng.choice(flat.size, size=size, p=flat)
    return d // n, d % n


def simulate_ucl(
    teams: dict[str, float], cal: dict, n_sims: int = 2000, seed: int = cfg.SEED
) -> dict:
    """League phase with a fresh random draw each run, then the knockout rounds."""
    names = list(teams)
    n = len(names)
    if n != cfg.UCL_TEAMS:
        raise ValueError(f"need exactly {cfg.UCL_TEAMS} teams, got {n}")

    rng = np.random.default_rng(seed)
    ratings = np.array([teams[t] for t in names], float)
    # Pots of 9, seeded by Elo, mirroring how UEFA seeds the league phase.
    pot_of = np.argsort(-ratings) // 9
    pots = {p: np.where(pot_of == p)[0] for p in range(4)}

    tally = {
        k: np.zeros(n) for k in
        ("top8", "top24", "r16", "qf", "sf", "final", "winner", "points")
    }

    for _ in range(n_sims):
        pts = np.zeros(n)
        gd = np.zeros(n)

        # --- league phase: 2 opponents from each pot, one home one away
        for p in range(4):
            for i in range(n):
                candidates = [j for j in pots[p] if j != i]
                opps = rng.choice(candidates, size=min(2, len(candidates)), replace=False)
                for k, j in enumerate(opps):
                    h, a = (i, j) if k == 0 else (j, i)
                    mat = elo_score_matrix(ratings[h], ratings[a], cal)
                    hg, ag = _sample(mat, rng)
                    hg, ag = int(hg[0]), int(ag[0])
                    if hg > ag:
                        pts[h] += 3
                    elif ag > hg:
                        pts[a] += 3
                    else:
                        pts[h] += 1
                        pts[a] += 1
                    gd[h] += hg - ag
                    gd[a] += ag - hg

        tally["points"] += pts
        order = np.lexsort((-ratings, -gd, -pts))
        tally["top8"][order[:8]] += 1
        tally["top24"][order[:24]] += 1

        # --- knockout playoff: 9-24 pair off, winners join the top 8 in the R16
        playoff = list(order[8:24])
        rng.shuffle(playoff)
        advanced = []
        for x in range(0, len(playoff), 2):
            advanced.append(_tie(playoff[x], playoff[x + 1], ratings, cal, rng))

        r16 = list(order[:8]) + advanced
        tally["r16"][r16] += 1
        for label in ("qf", "sf", "final"):
            rng.shuffle(r16)
            nxt = [
                _tie(r16[x], r16[x + 1], ratings, cal, rng)
                for x in range(0, len(r16), 2)
            ]
            tally[label][nxt] += 1
            r16 = nxt
        # Final is a single match at a neutral venue.
        a, b = r16[0], r16[1] if len(r16) > 1 else r16[0]
        mat = elo_score_matrix(ratings[a], ratings[b], cal, neutral=True)
        hg, ag = _sample(mat, rng)
        win = a if hg[0] > ag[0] else (b if ag[0] > hg[0] else rng.choice([a, b]))
        tally["winner"][win] += 1

    rows = []
    for i, t in enumerate(names):
        rows.append(
            {
                "team": t,
                "elo": round(float(ratings[i]), 1),
                "pot": int(pot_of[i]) + 1,
                "proj_points": round(float(tally["points"][i] / n_sims), 1),
                "top8_pct": round(100 * tally["top8"][i] / n_sims, 1),
                "qualify_pct": round(100 * tally["top24"][i] / n_sims, 1),
                "r16_pct": round(100 * tally["r16"][i] / n_sims, 1),
                "qf_pct": round(100 * tally["qf"][i] / n_sims, 1),
                "sf_pct": round(100 * tally["sf"][i] / n_sims, 1),
                "final_pct": round(100 * tally["final"][i] / n_sims, 1),
                "win_pct": round(100 * tally["winner"][i] / n_sims, 1),
            }
        )
    rows.sort(key=lambda r: -r["win_pct"])
    return {"n_sims": n_sims, "calibration": cal, "teams": rows}


def _tie(i: int, j: int, ratings: np.ndarray, cal: dict, rng) -> int:
    """Two-legged tie. No away-goals rule; level ties go to a coin flip."""
    agg_i = agg_j = 0
    for h, a in ((i, j), (j, i)):
        mat = elo_score_matrix(ratings[h], ratings[a], cal)
        hg, ag = _sample(mat, rng)
        if h == i:
            agg_i += int(hg[0])
            agg_j += int(ag[0])
        else:
            agg_j += int(hg[0])
            agg_i += int(ag[0])
    if agg_i > agg_j:
        return i
    if agg_j > agg_i:
        return j
    return int(rng.choice([i, j]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Champions League simulation.")
    ap.add_argument("--teams", default=str(cfg.ROOT / "config" / "ucl_2026_27.json"))
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--elo-date", default=None, help="ClubElo snapshot date")
    args = ap.parse_args()

    print("=" * 74)
    print("STAGE 6 - CHAMPIONS LEAGUE SIMULATION")
    print("=" * 74)

    with open(args.teams, encoding="utf-8") as fh:
        spec = json.load(fh)
    team_names = spec["teams"]

    elo_date = args.elo_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"  fetching ClubElo snapshot for {elo_date} ...")
    elo = fetch_elo(elo_date)

    matches = load_matches()
    all_domestic = sorted(set(matches["home"]) | set(matches["away"]))
    domestic_elo = build_elo_lookup(elo, all_domestic)
    print(f"  matched {len(domestic_elo):,} of {len(all_domestic):,} domestic clubs to Elo")

    cal_season = matches["season"].iloc[-1]
    cal = calibrate_elo_to_goals(matches, domestic_elo, cal_season)
    print(
        f"  calibrated on {cal['n_calibration_matches']:,} matches from {cal_season}: "
        f"{cal['slope_per_100_elo']:.3f} goals per 100 Elo, "
        f"home advantage {cal['home_adv']:.3f}"
    )

    ucl_elo = build_elo_lookup(elo, team_names)
    missing = [t for t in team_names if t not in ucl_elo]
    if missing:
        print(f"\n  ERROR: no Elo rating for {len(missing)} club(s): {missing}")
        print("  Add them to ELO_NAME_MAP in src/ucl.py, or correct the names")
        print(f"  in {args.teams}.")
        return 1

    print(f"  simulating {args.sims:,} tournaments ...\n")
    out = simulate_ucl(ucl_elo, cal, n_sims=args.sims)

    print(f"  {'club':22} {'pot':>3} {'elo':>6} {'pts':>5} {'top8':>6} "
          f"{'R16':>6} {'QF':>6} {'SF':>6} {'win':>6}")
    print("  " + "-" * 72)
    for r in out["teams"]:
        print(
            f"  {r['team'][:22]:22} {r['pot']:>3} {r['elo']:>6.0f} "
            f"{r['proj_points']:>5.1f} {r['top8_pct']:>5.1f}% {r['r16_pct']:>5.1f}% "
            f"{r['qf_pct']:>5.1f}% {r['sf_pct']:>5.1f}% {r['win_pct']:>5.1f}%"
        )

    dest = cfg.OUTPUT / "ucl_2026_27.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  Written to {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
