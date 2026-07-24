"""Stage 7 - match cards.

The front door of the whole project. Give it a fixture, get every market.

    python -m src.predict "Real Madrid" "Barcelona"
    python -m src.predict "Man City" Arsenal --comp E0
    python -m src.predict --round E0          # every upcoming fixture
    python -m src.predict "Real Madrid" "Barcelona" --json

Team names are matched loosely, so "real madrid", "Real Madrid" and "madrid"
all work. If a name is ambiguous or unknown, it says so and lists the closest
alternatives rather than guessing.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys

import pandas as pd

from . import config as cfg
from .clean import display_name
from .model import DixonColes, load_matches

BAR_WIDTH = 44


def resolve_team(name: str, candidates: list[str]) -> str:
    """Loose name matching with a helpful failure."""
    lowered = {c.lower(): c for c in candidates}
    key = name.strip().lower()
    if key in lowered:
        return lowered[key]

    partial = [c for c in candidates if key in c.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise SystemExit(
            f"'{name}' is ambiguous. Did you mean: {', '.join(sorted(partial))}?"
        )

    close = difflib.get_close_matches(name, candidates, n=5, cutoff=0.5)
    hint = f" Closest: {', '.join(close)}" if close else ""
    raise SystemExit(f"No team matching '{name}' in this competition.{hint}")


def find_comp(matches: pd.DataFrame, home: str, away: str) -> str | None:
    """Work out which competition a fixture belongs to."""
    recent = matches[matches["date"] >= matches["date"].max() - pd.Timedelta(days=400)]
    for comp in cfg.MAIN_LEAGUES:
        teams = set(recent[recent["comp"] == comp]["home"]) | set(
            recent[recent["comp"] == comp]["away"]
        )
        if home.lower() in {t.lower() for t in teams} and away.lower() in {
            t.lower() for t in teams
        }:
            return comp
    return None


def bar(p_home: float, p_draw: float, p_away: float) -> str:
    h = round(p_home * BAR_WIDTH)
    d = round(p_draw * BAR_WIDTH)
    a = max(BAR_WIDTH - h - d, 0)
    return "#" * h + "=" * d + "." * a


def format_card(p: dict, model: DixonColes) -> str:
    """The human-readable match card."""
    home_d, away_d = display_name(p["home"]), display_name(p["away"])
    mat = model.score_matrix(p["home"], p["away"], allow_unknown=True)
    league = cfg.LEAGUES.get(p["comp"], {}).get("name", p["comp"])

    lines = []
    lines.append("=" * 62)
    lines.append(f"  {home_d}  v  {away_d}")
    lines.append(f"  {league}   ratings as of {p['as_of']}")
    lines.append("=" * 62)

    if p["unrated"]:
        lines.append(
            f"  ! No rating yet for: {', '.join(p['unrated'])}. Treated as"
        )
        lines.append("    league-average. Expect this to be wide of the mark.")
        lines.append("")

    lines.append("")
    lines.append(f"  {'HOME':<14}{'DRAW':^14}{'AWAY':>14}")
    lines.append(
        f"  {p['p_home']:<13.1%}{p['p_draw']:^15.1%}{p['p_away']:>13.1%}"
    )
    lines.append(f"  {bar(p['p_home'], p['p_draw'], p['p_away'])}")
    lines.append("")
    lines.append(
        f"  fair odds     {p['fair_odds']['home']:>6}  "
        f"{p['fair_odds']['draw']:>6}  {p['fair_odds']['away']:>6}"
    )
    lines.append(
        f"  expected goals   {p['lambda_home']:.2f} - {p['lambda_away']:.2f}"
        f"   (total {p['expected_goals']:.2f})"
    )

    lines.append("")
    lines.append("  SCORELINE MATRIX  (rows = " + home_d + ", cols = " + away_d + ")")
    span = min(6, mat.shape[0])
    header = "        " + "".join(f"{j:>7}" for j in range(span))
    lines.append(header)
    peak = divmod(int(mat.argmax()), mat.shape[1])
    for i in range(span):
        cells = []
        for j in range(span):
            v = f"{mat[i, j] * 100:.1f}"
            cells.append(f"{'[' + v + ']':>7}" if (i, j) == peak else f"{v:>7}")
        lines.append(f"     {i}  " + "".join(cells))
    lines.append("        (values are %, [ ] marks the single likeliest result)")

    lines.append("")
    lines.append("  MARKETS")
    lines.append(
        f"    over 1.5 {p['p_over_1_5']:>6.1%}     over 2.5 {p['p_over_2_5']:>6.1%}"
        f"     over 3.5 {p['p_over_3_5']:>6.1%}"
    )
    lines.append(
        f"    BTTS     {p['p_btts']:>6.1%}     CS home  "
        f"{p['p_clean_sheet_home']:>6.1%}     CS away  {p['p_clean_sheet_away']:>6.1%}"
    )

    lines.append("")
    lines.append("  MOST LIKELY SCORELINES")
    for s in p["top_scorelines"]:
        lines.append(f"    {s['score']:>5}   {s['prob']:>6.1%}")
    lines.append("=" * 62)
    return "\n".join(lines)


def upcoming_round(matches: pd.DataFrame, comp: str, model: DixonColes) -> list[dict]:
    """Predict every pairing not yet played in the current season."""
    season = matches[matches["comp"] == comp]["season"].iloc[-1]
    df = matches[(matches["comp"] == comp) & (matches["season"] == season)]
    teams = sorted(set(df["home"]) | set(df["away"]))
    played = set(zip(df["home"], df["away"]))
    out = []
    for h in teams:
        for a in teams:
            if h != a and (h, a) not in played:
                out.append(model.predict(h, a, allow_unknown=True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Predict a football match.")
    ap.add_argument("home", nargs="?", help="home team")
    ap.add_argument("away", nargs="?", help="away team")
    ap.add_argument("--comp", default=None, choices=list(cfg.MAIN_LEAGUES))
    ap.add_argument("--as-of", default=None, help="rate teams as of this date")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--round", metavar="COMP", default=None,
                    help="predict every unplayed fixture in a competition")
    args = ap.parse_args()

    matches = load_matches()

    if args.round:
        comp = args.round
        as_of = pd.Timestamp(args.as_of) if args.as_of else matches["date"].max() + pd.Timedelta(days=1)
        model = DixonColes.fit(matches, comp=comp, as_of=as_of)
        preds = upcoming_round(matches, comp, model)
        if args.json:
            json.dump(preds, sys.stdout, indent=2)
            return 0
        print(f"\n  {len(preds)} unplayed fixtures in {cfg.LEAGUES[comp]['name']}\n")
        print(f"  {'fixture':46} {'H':>7}{'D':>7}{'A':>7}{'o2.5':>8}")
        print("  " + "-" * 76)
        for p in sorted(preds, key=lambda x: -x["p_home"]):
            fixture = f"{display_name(p['home'])} v {display_name(p['away'])}"
            print(
                f"  {fixture[:46]:46} {p['p_home']:>6.1%} {p['p_draw']:>6.1%} "
                f"{p['p_away']:>6.1%} {p['p_over_2_5']:>7.1%}"
            )
        return 0

    if not args.home or not args.away:
        ap.print_help()
        return 1

    comp = args.comp
    if comp is None:
        comp = find_comp(matches, args.home, args.away)
        if comp is None:
            print(
                f"Could not tell which competition '{args.home}' v '{args.away}' "
                "is in.\nPass --comp E0 or --comp SP1 explicitly."
            )
            return 1

    as_of = (
        pd.Timestamp(args.as_of) if args.as_of
        else matches["date"].max() + pd.Timedelta(days=1)
    )
    model = DixonColes.fit(matches, comp=comp, as_of=as_of)

    home = resolve_team(args.home, model.teams)
    away = resolve_team(args.away, model.teams)
    p = model.predict(home, away, allow_unknown=True)

    if args.json:
        json.dump(p, sys.stdout, indent=2)
        print()
    else:
        print(format_card(p, model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
