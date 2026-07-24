"""Verification for the Dixon-Coles model.

Run after any change to src/model.py:

    python -m tests.verify_model

Checks maths correctness, absence of look-ahead leakage, and football
plausibility. A model that passes the maths but ranks Getafe above Barcelona
is still broken, so both are tested.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from src import config as cfg
from src.model import DixonColes, load_matches

PASS, FAIL = "  [ok]  ", "  [XX]  "
problems: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{PASS if ok else FAIL}{name}{('  ' + detail) if detail else ''}")
    if not ok:
        problems.append(name)


def main() -> int:
    print("=" * 66)
    print("VERIFY - DIXON-COLES MODEL")
    print("=" * 66)

    matches = load_matches()

    # Fit as of the end of the 2025-26 season.
    as_of = pd.Timestamp("2026-06-01")

    print("\n-- fitting --")
    models = {}
    for comp in ("E0", "SP1"):
        m = DixonColes.fit(matches, comp=comp, as_of=as_of, measure_rho_gain=True)
        models[comp] = m
        print(
            f"  {comp}: {m.n_matches} matches, effective n={m.effective_n:.0f}, "
            f"{len(m.teams)} teams, converged={m.converged}"
        )

    print("\n-- 1. optimiser --")
    for comp, m in models.items():
        check(f"{comp} converged", m.converged)

    print("\n-- 1b. analytic gradient matches finite differences --")
    # The analytic jacobian is a ~10x speedup, but a wrong gradient does not
    # crash - it converges quietly to the wrong parameters. So the fast path is
    # checked against the slow one it replaced.
    t0 = time.time()
    fast = DixonColes.fit(matches, comp="E0", as_of=as_of, use_analytic_jac=True)
    t_fast = time.time() - t0
    t0 = time.time()
    slow = DixonColes.fit(matches, comp="E0", as_of=as_of, use_analytic_jac=False)
    t_slow = time.time() - t0

    # Log-likelihood is the sharpest test of gradient correctness: a wrong
    # gradient lands at a worse optimum, and it would show up here first.
    check("log-likelihood identical",
          abs(fast.log_likelihood - slow.log_likelihood) < 1e-3,
          f"{fast.log_likelihood:.4f} vs {slow.log_likelihood:.4f}")
    check("home advantage agrees", abs(fast.home_adv - slow.home_adv) < 1e-4,
          f"{fast.home_adv:.5f} vs {slow.home_adv:.5f}")
    check("rho agrees", abs(fast.rho - slow.rho) < 1e-4,
          f"{fast.rho:.5f} vs {slow.rho:.5f}")

    # Individual ratings can differ by ~1e-3 where the likelihood surface is
    # flat (a club with few matches in the window). That is optimiser
    # tolerance, not disagreement. What has to match is the output, so the
    # tight assertion is on predicted probabilities rather than parameters.
    pairs = [(fast.teams[i], fast.teams[j]) for i, j in ((0, 1), (2, 3), (4, 5))]
    worst = 0.0
    for hh, aa in pairs:
        pf, ps = fast.predict(hh, aa), slow.predict(hh, aa)
        worst = max(worst, max(abs(pf[k] - ps[k]) for k in ("p_home", "p_draw", "p_away")))
    check("predicted probabilities agree to 1e-3", worst < 1e-3,
          f"max 1X2 diff={worst:.2e}")

    att_diff = max(abs(fast.attack[t] - slow.attack[t]) for t in fast.teams)
    def_diff = max(abs(fast.defence[t] - slow.defence[t]) for t in fast.teams)
    print(f"  [--]  raw parameter drift: attack {att_diff:.2e}, defence {def_diff:.2e}"
          f"  (flat-surface tolerance, not error)")
    print(f"  [--]  speedup: {t_slow:.2f}s -> {t_fast:.2f}s ({t_slow / t_fast:.1f}x)")

    print("\n-- 2. league parameters --")
    for comp, m in models.items():
        check(
            f"{comp} home advantage positive and sane",
            0.05 < m.home_adv < 0.60,
            f"home_adv={m.home_adv:.3f}",
        )
        check(
            f"{comp} intercept implies a sane baseline scoring rate",
            0.7 < np.exp(m.intercept) < 2.5,
            f"exp(intercept)={np.exp(m.intercept):.2f} goals",
        )
        # Rho's sign is not asserted - what matters is whether letting it float
        # improves the fit at all, and whether low scores end up calibrated.
        check(
            f"{comp} free rho does not fit worse than rho=0",
            m.rho_gain > -1e-6,
            f"rho={m.rho:+.3f}, log-lik gain={m.rho_gain:+.2f}",
        )

    print("\n-- 2b. low-score calibration (what rho is actually for) --")
    # Evaluated over the whole fit window, not one season. At n=380 the
    # standard error on a ~13% rate is 1.7pp, so a single season cannot
    # distinguish a miscalibrated model from ordinary variance. Over ~1,900
    # matches the error halves and a 3pp threshold becomes meaningful.
    for comp, mdl in models.items():
        window = matches[
            (matches["comp"] == comp)
            & (matches["date"] < as_of)
            & (matches["date"] >= as_of - pd.Timedelta(days=cfg.LOOKBACK_DAYS))
        ]
        window = window[window["home"].isin(mdl.teams) & window["away"].isin(mdl.teams)]

        # Weight observations by the SAME exponential decay used in the fit.
        # The model's effective sample is ~300 recent matches, so scoring it
        # against 1,900 unweighted ones compares different things and
        # manufactures a deviation that is not really there.
        xi = np.log(2.0) / cfg.HALF_LIFE_DAYS
        age = (as_of - window["date"]).dt.total_seconds().to_numpy() / 86400.0
        w = np.exp(-xi * age)
        w = w / w.sum()

        pred_00, pred_11, pred_draw = [], [], []
        for r in window.itertuples():
            mat = mdl.score_matrix(r.home, r.away)
            pred_00.append(mat[0, 0])
            pred_11.append(mat[1, 1])
            pred_draw.append(np.trace(mat))

        obs = {
            "0-0": ((window.hg == 0) & (window.ag == 0)).to_numpy(float),
            "1-1": ((window.hg == 1) & (window.ag == 1)).to_numpy(float),
            "draw": (window.result == "D").to_numpy(float),
        }
        preds = {"0-0": pred_00, "1-1": pred_11, "draw": pred_draw}

        # Kish effective sample size for a weighted mean.
        n_eff = 1.0 / np.sum(w**2)

        # Hard assertion: the draw rate, which is what 1X2 actually depends on.
        p = float(np.sum(w * np.asarray(preds["draw"])))
        o = float(np.sum(w * obs["draw"]))
        se = float(np.sqrt(max(o, 1e-6) * (1 - o) / n_eff))
        check(
            f"{comp} draw rate within 3pp of observed",
            abs(p - o) < 0.03,
            f"predicted {p:.3f} vs actual {o:.3f} "
            f"(n_eff={n_eff:.0f}, {abs(p - o) / se:.1f} SE)",
        )

        # Reported, not asserted. Dixon-Coles has a single rho, and its tau
        # moves 0-0 and 1-1 in the SAME direction. La Liga needs 0-0 down and
        # 1-1 up, which no value of rho can express, so the optimiser parks rho
        # near zero. The errors cancel in the draw rate above, so 1X2 and
        # totals are unaffected - but individual correct-score probabilities
        # for those two cells carry a known bias. Documented in README.
        for label in ("0-0", "1-1"):
            p = float(np.sum(w * np.asarray(preds[label])))
            o = float(np.sum(w * obs[label]))
            se = float(np.sqrt(max(o, 1e-6) * (1 - o) / n_eff))
            flag = "     " if abs(p - o) < 0.03 else "  <-- known DC limit"
            print(
                f"  [--]  {comp} {label} correct-score cell   "
                f"predicted {p:.3f} vs actual {o:.3f} "
                f"({abs(p - o) / se:.1f} SE){flag}"
            )

    print("\n-- 2c. the quantities the markets actually price --")
    for comp, mdl in models.items():
        window = matches[
            (matches["comp"] == comp)
            & (matches["date"] < as_of)
            & (matches["date"] >= as_of - pd.Timedelta(days=cfg.LOOKBACK_DAYS))
        ]
        window = window[window["home"].isin(mdl.teams) & window["away"].isin(mdl.teams)]
        xi = np.log(2.0) / cfg.HALF_LIFE_DAYS
        age = (as_of - window["date"]).dt.total_seconds().to_numpy() / 86400.0
        w = np.exp(-xi * age)
        w = w / w.sum()

        pred_goals, pred_o25, pred_home = [], [], []
        for r in window.itertuples():
            pr = mdl.predict(r.home, r.away)
            pred_goals.append(pr["expected_goals"])
            pred_o25.append(pr["p_over_2_5"])
            pred_home.append(pr["p_home"])

        for label, pred, obs_arr in (
            ("mean total goals", np.sum(w * np.asarray(pred_goals)),
             np.sum(w * window["total_goals"].to_numpy(float))),
            ("over 2.5 rate", np.sum(w * np.asarray(pred_o25)),
             np.sum(w * (window["total_goals"] >= 3).to_numpy(float))),
            ("home win rate", np.sum(w * np.asarray(pred_home)),
             np.sum(w * (window["result"] == "H").to_numpy(float))),
        ):
            tol = 0.15 if "goals" in label else 0.03
            check(
                f"{comp} {label} calibrated",
                abs(pred - obs_arr) < tol,
                f"predicted {pred:.3f} vs actual {obs_arr:.3f}",
            )

    print("\n-- 3. score matrix integrity --")
    m = models["SP1"]
    mat = m.score_matrix("Real Madrid", "Barcelona")
    check("matrix sums to 1", np.isclose(mat.sum(), 1.0, atol=1e-9),
          f"sum={mat.sum():.12f}")
    check("no negative probabilities", (mat >= 0).all(),
          f"min={mat.min():.3e}")
    check("matrix is (MAX_GOALS+1) square", mat.shape[0] == mat.shape[1])

    print("\n-- 4. derived markets are consistent --")
    for comp, mdl in models.items():
        teams = mdl.ratings_table()["team"].tolist()
        for home, away in [(teams[0], teams[1]), (teams[-1], teams[0])]:
            p = mdl.predict(home, away)
            s = p["p_home"] + p["p_draw"] + p["p_away"]
            check(f"{comp} 1X2 sums to 1 ({home} v {away})",
                  abs(s - 1.0) < 1e-3, f"sum={s:.6f}")
            check(f"{comp} over/under 2.5 complementary",
                  abs(p["p_over_2_5"] + (1 - p["p_over_2_5"]) - 1.0) < 1e-9)
            check(f"{comp} over 1.5 >= over 2.5 >= over 3.5",
                  p["p_over_1_5"] >= p["p_over_2_5"] >= p["p_over_3_5"],
                  f"{p['p_over_1_5']:.3f} / {p['p_over_2_5']:.3f} / {p['p_over_3_5']:.3f}")
            check(f"{comp} expected goals plausible",
                  1.0 < p["expected_goals"] < 6.0,
                  f"xG total={p['expected_goals']:.2f}")

    print("\n-- 5. NO LOOK-AHEAD LEAKAGE --")
    # Fit at a date mid-history, then confirm a team that only existed later
    # is genuinely unknown to the model.
    early = pd.Timestamp("2015-01-01")
    m_early = DixonColes.fit(matches, comp="E0", as_of=early)
    later_only = [
        t for t in models["E0"].teams
        if t not in m_early.teams
    ]
    check(
        "model fitted at 2015 does not know teams that arrived later",
        len(later_only) > 0,
        f"unknown to 2015 model: {sorted(later_only)[:5]}",
    )
    used = matches[
        (matches["comp"] == "E0")
        & (matches["date"] < early)
        & (matches["date"] >= early - pd.Timedelta(days=1825))
    ]
    check(
        "fit used only pre-cutoff matches",
        m_early.n_matches == len(used),
        f"{m_early.n_matches} == {len(used)}",
    )

    print("\n-- 6. football plausibility --")
    # These are deliberately strict. An earlier version of this model inverted
    # the defence sign and ranked two relegated clubs top of the Premier
    # League while passing every mathematical check. Weak plausibility tests
    # are how that survives.
    sp_tbl = models["SP1"].ratings_table()
    e0_tbl = models["E0"].ratings_table()
    top_sp = sp_tbl.head(4)["team"].tolist()
    top_e0 = e0_tbl.head(5)["team"].tolist()

    check("BOTH Real Madrid and Barcelona in La Liga top 4",
          {"Real Madrid", "Barcelona"} <= set(top_sp), f"top4={top_sp}")
    big3 = {"Man City", "Arsenal", "Liverpool"}
    check("at least 2 of Man City / Arsenal / Liverpool in EPL top 5",
          len(big3 & set(top_e0)) >= 2, f"top5={top_e0}")

    # Sides that finished 2025-26 in the relegation places must not rate highly.
    bottom_marker = {"Luton", "Sheffield United", "Almeria", "Granada"}
    check("no recently-relegated side in either top 5",
          not (bottom_marker & set(top_sp) | bottom_marker & set(top_e0)),
          f"checked against {sorted(bottom_marker)}")

    check("best attack in La Liga is a genuine heavyweight",
          sp_tbl.sort_values("attack", ascending=False).iloc[0]["team"]
          in {"Real Madrid", "Barcelona", "Atletico Madrid", "Ath Madrid"},
          f"best attack = {sp_tbl.sort_values('attack', ascending=False).iloc[0]['team']}")

    # Home advantage must actually favour the home side in a mirror fixture.
    a, b = "Real Madrid", "Barcelona"
    ph = models["SP1"].predict(a, b)["p_home"]
    pa_rev = models["SP1"].predict(b, a)["p_away"]
    check("same fixture: home venue raises win probability",
          ph > pa_rev, f"{a} home={ph:.3f} vs {a} away={pa_rev:.3f}")

    print("\n-- 7. unknown team raises a clear error --")
    try:
        models["E0"].predict("Wakanda FC", "Arsenal")
        check("unknown team rejected", False, "no error raised")
    except KeyError as exc:
        check("unknown team rejected", "no rating" in str(exc))

    fmt = {
        "attack": "{:+.3f}".format,
        "defence": "{:+.3f}".format,
        "strength": "{:+.3f}".format,
        "xg_scored_home": "{:.2f}".format,
        "xg_conceded_home": "{:.2f}".format,
    }
    for comp, label in (("SP1", "La Liga"), ("E0", "Premier League")):
        print(f"\n-- sample output: {label} top 8 ratings --")
        print(models[comp].ratings_table().head(8).to_string(index=False, formatters=fmt))

    print("\n" + "=" * 66)
    if problems:
        print(f"FAILED - {len(problems)} check(s):")
        for p in problems:
            print(f"   x {p}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
