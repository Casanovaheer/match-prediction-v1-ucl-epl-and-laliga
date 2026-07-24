"""Stages 2 and 3 - the ratings engine and the scoreline matrix.

This is a time-weighted Dixon-Coles model (Dixon & Coles, 1997, *Modelling
Association Football Scores and Inefficiencies in the Football Betting Market*).

Every team carries two numbers, both centred on zero so they read as
"relative to an average side in this league":

    attack   goals created   (higher = scores more, so higher is better)
    defence  goals conceded  (higher = leakier, so LOWER is better)

plus three league-wide parameters: an intercept setting the baseline scoring
rate, home advantage, and rho, the low-score correction. Rho exists because a
plain Poisson model systematically under-predicts 0-0 and 1-1, which are among
the most common football results.

Expected goals for a fixture:

    lambda_home = exp(intercept + attack_home + defence_away + home_advantage)
    lambda_away = exp(intercept + attack_away + defence_home)

Centring both vectors and carrying an explicit intercept is what makes the
parameters identifiable. Without the intercept, adding a constant to every
defence value would rescale every prediction with no change in likelihood, and
the fit would drift.

Recent matches count for more: weight decays exponentially with age, halving
every HALF_LIFE_DAYS. That is what lets the model track form, new managers and
promoted sides instead of averaging over a decade.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from . import config as cfg

# Rho must stay small or the low-score correction can drive a probability
# negative. These bounds are comfortably inside the safe region.
RHO_BOUNDS = (-0.25, 0.25)

# Mild shrinkage of team ratings toward league average. Stops a promoted side
# with four matches played from being handed an extreme rating.
L2_PENALTY = 0.02


@dataclass
class DixonColes:
    """A fitted model for one league at one point in time."""

    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    home_adv: float = 0.25
    rho: float = -0.05
    comp: str = ""
    as_of: pd.Timestamp | None = None
    n_matches: int = 0
    effective_n: float = 0.0
    converged: bool = False
    log_likelihood: float = 0.0
    # Log-likelihood gained by letting rho float instead of pinning it to 0.
    # Positive means the low-score correction is earning its place.
    rho_gain: float = 0.0

    # ------------------------------------------------------------------ fitting

    @staticmethod
    def _tau(hg, ag, lam, mu, rho):
        """Dixon-Coles dependence correction for the four lowest scorelines."""
        t = np.ones_like(lam, dtype=float)
        m = (hg == 0) & (ag == 0)
        t[m] = 1.0 - lam[m] * mu[m] * rho
        m = (hg == 0) & (ag == 1)
        t[m] = 1.0 + lam[m] * rho
        m = (hg == 1) & (ag == 0)
        t[m] = 1.0 + mu[m] * rho
        m = (hg == 1) & (ag == 1)
        t[m] = 1.0 - rho
        return t

    @classmethod
    def fit(
        cls,
        matches: pd.DataFrame,
        comp: str,
        as_of: pd.Timestamp,
        half_life_days: int = cfg.HALF_LIFE_DAYS,
        lookback_days: int = cfg.LOOKBACK_DAYS,
        min_matches: int = 50,
    ) -> "DixonColes":
        """Fit on every match in `comp` played strictly before `as_of`.

        The strict inequality matters: including the match being predicted
        would leak the answer into the model and make any backtest worthless.
        """
        as_of = pd.Timestamp(as_of)
        window_start = as_of - pd.Timedelta(days=lookback_days)

        df = matches[
            (matches["comp"] == comp)
            & (matches["date"] < as_of)
            & (matches["date"] >= window_start)
        ]
        if len(df) < min_matches:
            raise ValueError(
                f"only {len(df)} matches for {comp} before {as_of:%Y-%m-%d} "
                f"(need {min_matches})"
            )

        teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h = df["home"].map(idx).to_numpy(np.int64)
        a = df["away"].map(idx).to_numpy(np.int64)
        hg = df["hg"].to_numpy(np.int64)
        ag = df["ag"].to_numpy(np.int64)

        age_days = (as_of - df["date"]).dt.total_seconds().to_numpy() / 86400.0
        xi = math.log(2.0) / half_life_days
        w = np.exp(-xi * age_days)

        def unpack(params: np.ndarray):
            # Centre both vectors: only differences from the league average are
            # meaningful, and the intercept carries the overall goal level.
            att = params[:n]
            dfc = params[n : 2 * n]
            return (
                att - att.mean(),
                dfc - dfc.mean(),
                params[2 * n],      # intercept
                params[2 * n + 1],  # home advantage
                params[2 * n + 2],  # rho
            )

        def neg_log_lik(params: np.ndarray, fixed_rho: float | None = None) -> float:
            att, dfc, icept, hadv, rho = unpack(params)
            if fixed_rho is not None:
                rho = fixed_rho

            lam = np.exp(icept + att[h] + dfc[a] + hadv)
            mu = np.exp(icept + att[a] + dfc[h])
            lam = np.clip(lam, 1e-9, 25.0)
            mu = np.clip(mu, 1e-9, 25.0)

            tau = np.clip(cls._tau(hg, ag, lam, mu, rho), 1e-12, None)

            ll = w * (
                np.log(tau) + hg * np.log(lam) - lam + ag * np.log(mu) - mu
            )
            # Pure shrinkage toward league average - not load-bearing for
            # identifiability, which the centring and intercept handle.
            penalty = L2_PENALTY * (np.sum(att**2) + np.sum(dfc**2))
            return -ll.sum() + penalty

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.0, 0.25, -0.05]])
        bounds = (
            [(-3.0, 3.0)] * n
            + [(-3.0, 3.0)] * n
            + [(-2.0, 2.0), (-1.0, 1.5), RHO_BOUNDS]
        )

        opts = {"maxiter": 1000, "ftol": 1e-11}
        res = minimize(neg_log_lik, x0, method="L-BFGS-B", bounds=bounds, options=opts)

        # How much is the low-score correction actually earning? Refit with
        # rho pinned to zero and compare. Reported, not asserted - a league
        # where rho does nothing is information, not a failure.
        res0 = minimize(
            neg_log_lik, x0, args=(0.0,), method="L-BFGS-B", bounds=bounds, options=opts
        )

        att, dfc, icept, hadv, rho = unpack(res.x)

        return cls(
            teams=teams,
            attack={t: float(att[i]) for t, i in idx.items()},
            defence={t: float(dfc[i]) for t, i in idx.items()},
            intercept=float(icept),
            home_adv=float(hadv),
            rho=float(rho),
            comp=comp,
            as_of=as_of,
            n_matches=len(df),
            effective_n=float(w.sum()),
            converged=bool(res.success),
            log_likelihood=float(-res.fun),
            rho_gain=float(res0.fun - res.fun),
        )

    # --------------------------------------------------------------- prediction

    def knows(self, team: str) -> bool:
        return team in self.attack

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals for each side."""
        for t in (home, away):
            if t not in self.attack:
                raise KeyError(
                    f"{t!r} has no rating in {self.comp} as of "
                    f"{self.as_of:%Y-%m-%d}. Newly promoted, or a name mismatch?"
                )
        lam = math.exp(
            self.intercept + self.attack[home] + self.defence[away] + self.home_adv
        )
        mu = math.exp(self.intercept + self.attack[away] + self.defence[home])
        return min(lam, 25.0), min(mu, 25.0)

    def score_matrix(
        self, home: str, away: str, max_goals: int = cfg.MAX_GOALS
    ) -> np.ndarray:
        """P(home scores i, away scores j) for i, j in 0..max_goals.

        This is the single object everything else is derived from.
        """
        lam, mu = self.lambdas(home, away)
        gh = poisson.pmf(np.arange(max_goals + 1), lam)
        ga = poisson.pmf(np.arange(max_goals + 1), mu)
        m = np.outer(gh, ga)

        # Dixon-Coles correction on the four lowest scorelines.
        r = self.rho
        m[0, 0] *= 1.0 - lam * mu * r
        m[0, 1] *= 1.0 + lam * r
        m[1, 0] *= 1.0 + mu * r
        m[1, 1] *= 1.0 - r

        m = np.clip(m, 0.0, None)
        total = m.sum()
        if total <= 0:
            raise ValueError(f"degenerate score matrix for {home} v {away}")
        return m / total  # renormalise: the tail beyond max_goals is truncated

    def predict(self, home: str, away: str) -> dict:
        """Every market for one fixture, all read off one matrix."""
        m = self.score_matrix(home, away)
        lam, mu = self.lambdas(home, away)
        n = m.shape[0]

        tri = np.arange(n)
        home_win = float(np.tril(m, -1).sum())   # home goals > away goals
        draw = float(np.trace(m))
        away_win = float(np.triu(m, 1).sum())

        totals = tri[:, None] + tri[None, :]
        over25 = float(m[totals >= 3].sum())
        over15 = float(m[totals >= 2].sum())
        over35 = float(m[totals >= 4].sum())
        btts = float(m[1:, 1:].sum())

        flat = m.flatten()
        order = np.argsort(flat)[::-1][:5]
        top = [
            {
                "score": f"{int(i // n)}-{int(i % n)}",
                "prob": round(float(flat[i]), 4),
            }
            for i in order
        ]

        return {
            "home": home,
            "away": away,
            "comp": self.comp,
            "as_of": self.as_of.strftime("%Y-%m-%d") if self.as_of is not None else None,
            "lambda_home": round(lam, 3),
            "lambda_away": round(mu, 3),
            "p_home": round(home_win, 4),
            "p_draw": round(draw, 4),
            "p_away": round(away_win, 4),
            "p_over_1_5": round(over15, 4),
            "p_over_2_5": round(over25, 4),
            "p_over_3_5": round(over35, 4),
            "p_btts": round(btts, 4),
            "p_clean_sheet_home": round(float(m[:, 0].sum()), 4),
            "p_clean_sheet_away": round(float(m[0, :].sum()), 4),
            "expected_goals": round(lam + mu, 3),
            "top_scorelines": top,
            "fair_odds": {
                "home": round(1.0 / home_win, 2) if home_win > 0 else None,
                "draw": round(1.0 / draw, 2) if draw > 0 else None,
                "away": round(1.0 / away_win, 2) if away_win > 0 else None,
            },
        }

    # ------------------------------------------------------------------- output

    def ratings_table(self) -> pd.DataFrame:
        """Teams ranked by overall strength.

        strength = attack - defence, because attack is "goals created" (up is
        good) while defence is "goals conceded" (up is bad). Getting this sign
        wrong ranks the leakiest teams top, so it is covered by a test.

        The two expected-goal columns are against a perfectly average opponent,
        which makes them readable as real goals rather than log-space numbers.
        """
        rows = [
            {
                "team": t,
                "attack": self.attack[t],
                "defence": self.defence[t],
                "strength": self.attack[t] - self.defence[t],
                "xg_scored_home": math.exp(
                    self.intercept + self.attack[t] + self.home_adv
                ),
                "xg_conceded_home": math.exp(self.intercept + self.defence[t]),
            }
            for t in self.teams
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("strength", ascending=False)
            .reset_index(drop=True)
        )

    def to_dict(self) -> dict:
        return {
            "comp": self.comp,
            "as_of": self.as_of.strftime("%Y-%m-%d") if self.as_of is not None else None,
            "intercept": self.intercept,
            "home_adv": self.home_adv,
            "rho": self.rho,
            "rho_gain": self.rho_gain,
            "n_matches": self.n_matches,
            "effective_n": round(self.effective_n, 1),
            "converged": self.converged,
            "log_likelihood": self.log_likelihood,
            "attack": self.attack,
            "defence": self.defence,
        }

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


# ------------------------------------------------------------------- utilities


def load_matches() -> pd.DataFrame:
    """Load the cleaned match table."""
    path = cfg.DATA_PROC / "matches.csv"
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.clean` first.")
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    return df
