"""Top picks for a single fixture.

Give it two club names. It works out which engine can rate them, produces every
market, and ranks the picks by how much confidence each actually deserves.

    python -m src.toppick "Real Madrid" "Barcelona"
    python -m src.toppick "Degerfors" "Djurgarden"

Two engines, chosen automatically:

  Dixon-Coles   Used when both clubs are in one of the 11 collected leagues.
                This is the validated path - 52.6% measured accuracy over
                8,360 out-of-sample matches.

  Elo bridge    Used for any other European club, via ClubElo. A three-parameter
                approximation calibrated on Big-5 domestic football. It has NO
                measured accuracy, and the output says so every time.

The confidence tier matters more than the pick. A market sitting within 3
points of 50/50 is labelled a coin flip and should be ignored: the argmax of a
coin flip is noise wearing the costume of information.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from . import config as cfg
from .clean import display_name
from .model import DixonColes, load_matches

TIERS = ((0.20, "STRONG"), (0.10, "solid"), (0.03, "lean"))


def tier(p: float) -> str:
    edge = p - 0.5
    for cut, label in TIERS:
        if edge > cut:
            return label
    return "COIN FLIP - avoid"


def markets(m: np.ndarray, home: str, away: str) -> list[tuple[str, str, float]]:
    """Every market, already reduced to its more likely side."""
    n = m.shape[0]
    H = np.arange(n)[:, None] * np.ones((1, n))
    A = np.ones((n, 1)) * np.arange(n)[None, :]
    tot, diff = H + A, H - A

    ph = float(np.tril(m, -1).sum())
    pdw = float(np.trace(m))
    pa = float(np.triu(m, 1).sum())
    btts = float(m[1:, 1:].sum())
    h_scores = 1 - float(m[0, :].sum())
    a_scores = 1 - float(m[:, 0].sum())

    # Each market lists an EXHAUSTIVE set of outcomes summing to 1, and the
    # pick is the most likely of them. Two earlier bugs both came from
    # non-exhaustive sets: "Match result" omitted the draw, and the handicap
    # offered "home by 2+" against "away by 2+" with no "neither" - which made
    # a genuine 70/30 edge display as a 29.6% coin flip. The assertion below
    # makes that class of mistake impossible to ship.
    # `total` is what the outcomes must sum to. It is 1 for a normal market,
    # but 2 for double chance, whose three options each cover two of the three
    # results and therefore overlap.
    spec: list[tuple[str, list[tuple[str, float]], float]] = [
        ("Match result", [
            (f"{home} win", ph), ("Draw", pdw), (f"{away} win", pa)], 1.0),
        ("Double chance", [
            (f"{home} win or draw", ph + pdw),
            (f"{away} win or draw", pa + pdw),
            ("Either team wins (no draw)", ph + pa)], 2.0),
        ("Draw no bet", [
            (home, ph / (ph + pa)), (away, pa / (ph + pa))], 1.0),
        ("Over/Under 1.5", [
            ("Over 1.5", float(m[tot >= 2].sum())),
            ("Under 1.5", float(m[tot <= 1].sum()))], 1.0),
        ("Over/Under 2.5", [
            ("Over 2.5", float(m[tot >= 3].sum())),
            ("Under 2.5", float(m[tot <= 2].sum()))], 1.0),
        ("Over/Under 3.5", [
            ("Over 3.5", float(m[tot >= 4].sum())),
            ("Under 3.5", float(m[tot <= 3].sum()))], 1.0),
        ("Both teams score", [("Yes", btts), ("No", 1 - btts)], 1.0),
        ("Winning margin 2+", [
            (f"{away} by 2+", float(m[diff <= -2].sum())),
            ("anything else", float(m[diff > -2].sum()))], 1.0),
        (f"{home} to score", [("Yes", h_scores), ("No", 1 - h_scores)], 1.0),
        (f"{away} to score", [("Yes", a_scores), ("No", 1 - a_scores)], 1.0),
    ]

    out = []
    for market, outcomes, expected in spec:
        total = sum(p for _, p in outcomes)
        assert abs(total - expected) < 1e-6, (
            f"{market} outcomes sum to {total:.4f}, expected {expected}"
        )
        label, p = max(outcomes, key=lambda o: o[1])
        out.append((market, label, p))
    return out


def render(m: np.ndarray, home: str, away: str, engine: str, meta: str) -> str:
    n = m.shape[0]
    rows = markets(m, home, away)
    ranked = sorted(rows, key=lambda r: -r[2])
    best = ranked[0]

    lam = float(np.sum(m * np.arange(n)[:, None]))
    mu = float(np.sum(m * np.arange(n)[None, :]))

    L = []
    L.append("=" * 72)
    L.append(f"  {home}  v  {away}")
    L.append(f"  {meta}")
    L.append("=" * 72)
    L.append("")
    L.append("  TOP PICK")
    # Market name included: a bare "Yes" tells the reader nothing.
    L.append(
        f"    >>  {best[0]}:  {best[1]}"
        f"  --  {best[2]:.1%}   (fair odds {1 / best[2]:.2f})"
    )
    L.append("")
    L.append(f"  expected goals  {lam:.2f} - {mu:.2f}   (total {lam + mu:.2f})")
    L.append("")
    L.append(f"  {'MARKET':22} {'PICK':28} {'PROB':>7} {'ODDS':>7}  CONFIDENCE")
    L.append("  " + "-" * 72)
    for market, pick, p in ranked:
        L.append(
            f"  {market[:22]:22} {pick[:28]:28} {p:>6.1%} {1 / p:>7.2f}  {tier(p)}"
        )

    L.append("")
    L.append("  CORRECT SCORE")
    flat = m.ravel()
    for i in np.argsort(flat)[::-1][:6]:
        L.append(f"    {i // n}-{i % n}   {flat[i]:>6.1%}   fair {1 / flat[i]:.2f}")

    H = np.arange(n)[:, None] * np.ones((1, n))
    A = np.ones((n, 1)) * np.arange(n)[None, :]
    d = H - A
    L.append("")
    L.append("  WINNING MARGIN")
    for label, mask in (
        (f"{home} by 2+", d >= 2),
        (f"{home} by 1", d == 1),
        ("Draw", d == 0),
        (f"{away} by 1", d == -1),
        (f"{away} by 2", d == -2),
        (f"{away} by 3+", d <= -3),
    ):
        L.append(f"    {label[:22]:22} {float(m[mask].sum()):>6.1%}")

    flips = [r for r in ranked if tier(r[2]) == "COIN FLIP - avoid"]
    if flips:
        L.append("")
        L.append("  DO NOT BET THESE - the model has no view:")
        for market, pick, p in flips:
            L.append(f"    x {market} ({p:.1%} - effectively a coin flip)")

    L.append("")
    L.append("  ENGINE: " + engine)
    L.append("=" * 72)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ranked top picks for one fixture.")
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--elo-date", default=None)
    args = ap.parse_args()

    matches = load_matches()
    as_of = matches["date"].max() + pd.Timedelta(days=1)

    # Prefer the validated engine wherever it can reach both clubs.
    for comp in cfg.MAIN_LEAGUES:
        try:
            model = DixonColes.fit(matches, comp=comp, as_of=as_of)
        except ValueError:
            continue
        low = {t.lower(): t for t in model.teams}
        h = low.get(args.home.lower())
        a = low.get(args.away.lower())
        if h and a:
            mat = model.score_matrix(h, a)
            print(render(
                mat, display_name(h), display_name(a),
                "Dixon-Coles - the validated path. 52.6% accuracy measured over "
                "8,360\n          out-of-sample matches; closing odds score 54.6%.",
                f"{cfg.LEAGUES[comp]['name']}  ·  ratings as of {as_of:%Y-%m-%d}",
            ))
            return 0

    # Fall back to the Elo bridge for everyone else.
    from .ucl import (
        build_elo_lookup,
        calibrate_elo_to_goals,
        elo_score_matrix,
        fetch_elo,
    )

    date = args.elo_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    elo = fetch_elo(date)
    lookup = build_elo_lookup(elo, sorted(set(matches["home"]) | set(matches["away"])))
    cal = calibrate_elo_to_goals(matches, lookup, matches["season"].iloc[-1])

    by_name = {str(k).lower(): (str(k), float(v))
               for k, v in zip(elo["Club"], elo["Elo"])}
    h = by_name.get(args.home.lower())
    a = by_name.get(args.away.lower())
    if not h or not a:
        missing = args.home if not h else args.away
        import difflib
        close = difflib.get_close_matches(missing, list(elo["Club"]), n=6, cutoff=0.5)
        print(f"\n  Could not find '{missing}'.")
        print(f"  Closest names in ClubElo: {', '.join(close) if close else '(none)'}")
        print("  Tip: ClubElo uses spellings like 'Djurgarden', 'Mjaellby', 'Brugge'.")
        return 1

    mat = elo_score_matrix(h[1], a[1], cal)
    print(render(
        mat, h[0], a[0],
        "Elo bridge - WEAKER. This league is not in the model; ratings come\n"
        "          from ClubElo and the goals calibration is borrowed from Big-5\n"
        "          football. NO measured accuracy. No team news. Treat as rough.",
        f"Elo {h[1]:.0f} v {a[1]:.0f}  (gap {h[1] - a[1]:+.0f})  ·  snapshot {date}",
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
