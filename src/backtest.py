"""Stage 4 - walk-forward backtest.

This is the stage that decides whether the model is worth anything, and it is
the stage most public football projects quietly skip.

Method: walk forward through history one week at a time. At each step, refit
using ONLY matches played before that date, then predict the coming week. No
match is ever predicted by a model that has seen it, or seen anything that
happened after it.

Scored on four measures:

    accuracy   share of matches where the highest-probability outcome won.
               The weakest measure - a model can win on accuracy by being
               confidently right on easy games and confidently wrong on hard
               ones. Reported because everyone asks for it.

    log-loss   -log(probability assigned to what actually happened). Punishes
               confident mistakes harshly. The primary measure.

    Brier      mean squared error across the three outcomes. Less brutal than
               log-loss about long shots.

    RPS        ranked probability score. Football-specific: it knows that
               home/draw/away is ordered, so predicting a home win when the
               away side wins is a worse miss than predicting a draw.

Every measure is also computed for the bookmakers' closing odds over exactly
the same fixtures, because "is it any good" only means anything relative to
the best freely available benchmark.

    python -m src.backtest --from 2015 --to 2026
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import config as cfg
from .model import DixonColes, load_matches

OUTCOMES = ("H", "D", "A")
EPS = 1e-15


# --------------------------------------------------------------------- scoring


def log_loss(probs: np.ndarray, actual: np.ndarray) -> float:
    """probs is (n, 3) in H/D/A order; actual is (n,) of 0/1/2."""
    p = np.clip(probs[np.arange(len(actual)), actual], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, actual: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual)), actual] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def rps(probs: np.ndarray, actual: np.ndarray) -> float:
    """Ranked probability score over the ordered outcome scale H < D < A."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual)), actual] = 1.0
    cum_p = np.cumsum(probs, axis=1)[:, :-1]
    cum_o = np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1)) / (probs.shape[1] - 1))


def accuracy(probs: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == actual))


def score_all(probs: np.ndarray, actual: np.ndarray) -> dict:
    return {
        "n": int(len(actual)),
        "accuracy": round(accuracy(probs, actual), 4),
        "log_loss": round(log_loss(probs, actual), 4),
        "brier": round(brier(probs, actual), 4),
        "rps": round(rps(probs, actual), 4),
    }


# -------------------------------------------------------------------- the walk


def walk_forward(
    matches: pd.DataFrame,
    comp: str,
    start_year: int,
    end_year: int,
    step_days: int = 7,
) -> pd.DataFrame:
    """Refit weekly, predict the following week, never look ahead."""
    df = matches[matches["comp"] == comp].sort_values("date")
    start = pd.Timestamp(f"{start_year}-07-01")
    end = pd.Timestamp(f"{end_year}-07-01")
    test = df[(df["date"] >= start) & (df["date"] < end)]
    if test.empty:
        return pd.DataFrame()

    rows = []
    cursor = start
    last_model = None
    n_fits = 0

    while cursor < end:
        window_end = cursor + pd.Timedelta(days=step_days)
        due = test[(test["date"] >= cursor) & (test["date"] < window_end)]

        if not due.empty:
            try:
                model = DixonColes.fit(df, comp=comp, as_of=cursor)
                last_model = model
                n_fits += 1
            except ValueError:
                model = last_model  # too little history yet - carry forward
            if model is not None:
                for r in due.itertuples():
                    p = model.predict(r.home, r.away, allow_unknown=True)
                    rows.append(
                        {
                            "date": r.date,
                            "comp": comp,
                            "season": r.season,
                            "home": r.home,
                            "away": r.away,
                            "hg": r.hg,
                            "ag": r.ag,
                            "result": r.result,
                            "p_h": p["p_home"],
                            "p_d": p["p_draw"],
                            "p_a": p["p_away"],
                            "p_over25": p["p_over_2_5"],
                            "p_btts": p["p_btts"],
                            "unrated": len(p["unrated"]),
                            "mkt_h": r.p_h_mkt,
                            "mkt_d": r.p_d_mkt,
                            "mkt_a": r.p_a_mkt,
                        }
                    )
        cursor = window_end

    out = pd.DataFrame(rows)
    out.attrs["n_fits"] = n_fits
    return out


def summarise(preds: pd.DataFrame) -> dict:
    """Score the model, then score the market over the identical fixtures."""
    actual = preds["result"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    model_p = preds[["p_h", "p_d", "p_a"]].to_numpy(float)
    model_p = model_p / model_p.sum(axis=1, keepdims=True)

    result = {"model": score_all(model_p, actual)}

    # The market only exists for part of history, so it is scored on the
    # overlap and the model is re-scored on that same overlap. Comparing a
    # model measured on 8,000 games to a market measured on 6,000 would be
    # meaningless.
    has_mkt = preds[["mkt_h", "mkt_d", "mkt_a"]].notna().all(axis=1).to_numpy()
    if has_mkt.sum() > 100:
        mkt_p = preds.loc[has_mkt, ["mkt_h", "mkt_d", "mkt_a"]].to_numpy(float)
        mkt_p = mkt_p / mkt_p.sum(axis=1, keepdims=True)
        result["market"] = score_all(mkt_p, actual[has_mkt])
        result["model_on_market_subset"] = score_all(model_p[has_mkt], actual[has_mkt])

    # A model that cannot beat "always predict the base rate" is worthless.
    base = np.tile(
        np.array([(actual == i).mean() for i in range(3)]), (len(actual), 1)
    )
    result["baseline_base_rate"] = score_all(base, actual)
    return result


def calibration_table(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Do things predicted at 30% actually happen 30% of the time?"""
    rows = []
    for outcome, col in zip(OUTCOMES, ("p_h", "p_d", "p_a")):
        p = preds[col].to_numpy(float)
        y = (preds["result"] == outcome).to_numpy(float)
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        for b in range(bins):
            m = idx == b
            if m.sum() < 20:
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "bucket": f"{edges[b]:.0%}-{edges[b + 1]:.0%}",
                    "n": int(m.sum()),
                    "predicted": round(float(p[m].mean()), 3),
                    "actual": round(float(y[m].mean()), 3),
                    "gap": round(float(y[m].mean() - p[m].mean()), 3),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward backtest.")
    ap.add_argument("--from", dest="from_year", type=int, default=2015)
    ap.add_argument("--to", dest="to_year", type=int, default=cfg.LAST_SEASON)
    ap.add_argument("--comps", nargs="*", default=list(cfg.MAIN_LEAGUES))
    args = ap.parse_args()

    print("=" * 68)
    print(f"STAGE 4 - WALK-FORWARD BACKTEST  ({args.from_year} to {args.to_year})")
    print("=" * 68)
    print("  Refitting weekly on pre-match data only. This takes a few minutes.\n")

    matches = load_matches()
    all_preds = []

    for comp in args.comps:
        t0 = time.time()
        preds = walk_forward(matches, comp, args.from_year, args.to_year)
        if preds.empty:
            print(f"  {comp}: no data in range")
            continue
        preds["comp"] = comp
        all_preds.append(preds)
        print(
            f"  {comp}: {len(preds):,} matches predicted from "
            f"{preds.attrs.get('n_fits', 0)} weekly refits "
            f"({time.time() - t0:.0f}s)"
        )

    if not all_preds:
        print("No predictions produced.")
        return 1

    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_csv(cfg.OUTPUT / "backtest_predictions.csv", index=False)

    report: dict = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "range": f"{args.from_year}-{args.to_year}",
        "overall": summarise(preds),
        "by_competition": {},
        "by_season": {},
    }
    for comp, grp in preds.groupby("comp"):
        report["by_competition"][comp] = summarise(grp)
    for season, grp in preds.groupby("season"):
        if len(grp) > 100:
            report["by_season"][season] = summarise(grp)["model"]

    with open(cfg.OUTPUT / "backtest_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    calib = calibration_table(preds)
    calib.to_csv(cfg.OUTPUT / "calibration.csv", index=False)

    # ------------------------------------------------------------- print report
    o = report["overall"]
    print("\n" + "=" * 68)
    print("RESULTS - all competitions, out of sample")
    print("=" * 68)
    hdr = f"  {'':28} {'n':>7} {'acc':>8} {'log-loss':>10} {'Brier':>8} {'RPS':>8}"
    print(hdr)
    print("  " + "-" * 66)
    for key, label in (
        ("baseline_base_rate", "base rate (do nothing)"),
        ("model", "our model"),
        ("model_on_market_subset", "our model, market subset"),
        ("market", "bookmaker closing odds"),
    ):
        if key not in o:
            continue
        s = o[key]
        print(
            f"  {label:28} {s['n']:>7,} {s['accuracy']:>7.1%} "
            f"{s['log_loss']:>10.4f} {s['brier']:>8.4f} {s['rps']:>8.4f}"
        )

    if "market" in o:
        gap = o["model_on_market_subset"]["log_loss"] - o["market"]["log_loss"]
        print(f"\n  log-loss gap to closing odds: {gap:+.4f}")
        print(
            "  (positive = market is sharper. Matching it is a strong result;\n"
            "   beating it consistently on free data would be extraordinary.)"
        )

    print("\n  By competition:")
    for comp, s in report["by_competition"].items():
        m = s["model"]
        print(
            f"    {comp:5} n={m['n']:>6,}  acc={m['accuracy']:.1%}  "
            f"log-loss={m['log_loss']:.4f}  RPS={m['rps']:.4f}"
        )

    print("\n  Calibration (are stated probabilities honest?):")
    print(calib.to_string(index=False))

    print("\n  Written to output/backtest_report.json, backtest_predictions.csv")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
