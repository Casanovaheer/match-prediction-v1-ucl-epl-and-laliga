"""Stage 5 - Monte Carlo season simulation.

Takes the current table plus every remaining fixture, plays the rest of the
season N times by sampling each match from its own scoreline matrix, and counts
how often each club finishes where.

This is the same method the Opta Supercomputer uses. The arithmetic is not the
difficult part - the difficulty is all upstream, in the ratings.

Tie-breaking uses goal difference then goals scored. That is correct for the
Premier League. La Liga formally breaks ties on head-to-head record first,
which this does not reproduce; the effect on a 10,000-run distribution is
small but it is an approximation, not an exact rule.

    python -m src.simulate --comp E0 --season 2026-27
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config as cfg
from .model import DixonColes, load_matches


def current_table(
    matches: pd.DataFrame,
    comp: str,
    season: str,
    as_of: pd.Timestamp | None = None,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """Standings from matches already played this season.

    With as_of set, only matches before that date count - which is how a
    genuine mid-season projection is reproduced after the fact.
    """
    df = matches[(matches["comp"] == comp) & (matches["season"] == season)]
    if teams is None:
        teams = sorted(set(df["home"]) | set(df["away"]))
    if as_of is not None:
        df = df[df["date"] < as_of]
    rows = []
    for t in teams:
        home = df[df["home"] == t]
        away = df[df["away"] == t]
        gf = int(home["hg"].sum() + away["ag"].sum())
        ga = int(home["ag"].sum() + away["hg"].sum())
        w = int((home["hg"] > home["ag"]).sum() + (away["ag"] > away["hg"]).sum())
        d = int((home["hg"] == home["ag"]).sum() + (away["ag"] == away["hg"]).sum())
        losses = len(home) + len(away) - w - d
        rows.append(
            {
                "team": t,
                "played": len(home) + len(away),
                "won": w,
                "drawn": d,
                "lost": losses,
                "gf": gf,
                "ga": ga,
                "gd": gf - ga,
                "points": 3 * w + d,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["points", "gd", "gf"], ascending=False
    ).reset_index(drop=True)


def remaining_fixtures(
    matches: pd.DataFrame,
    comp: str,
    season: str,
    teams: list[str] | None = None,
    as_of: pd.Timestamp | None = None,
) -> list[tuple[str, str]]:
    """Every home/away pairing in the season that has not yet been played.

    Derived from the double round-robin rather than a published schedule, so
    it works before fixture lists are released. For a full-season projection
    the order of fixtures does not affect the distribution of final tables -
    only the set of remaining matches does.
    """
    df = matches[(matches["comp"] == comp) & (matches["season"] == season)]
    if teams is None:
        teams = sorted(set(df["home"]) | set(df["away"]))
    if as_of is not None:
        df = df[df["date"] < as_of]
    played = set(zip(df["home"], df["away"]))
    return [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]


def simulate(
    model: DixonColes,
    table: pd.DataFrame,
    fixtures: list[tuple[str, str]],
    n_sims: int = cfg.N_SIMS,
    seed: int = cfg.SEED,
) -> dict:
    """Play the remaining fixtures n_sims times. Returns finishing distributions."""
    rng = np.random.default_rng(seed)
    teams = table["team"].tolist()
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    points = np.tile(table["points"].to_numpy(np.int32)[:, None], (1, n_sims))
    gd = np.tile(table["gd"].to_numpy(np.int32)[:, None], (1, n_sims))
    gf = np.tile(table["gf"].to_numpy(np.int32)[:, None], (1, n_sims))

    max_goals = cfg.MAX_GOALS
    span = max_goals + 1

    for home, away in fixtures:
        if home not in idx or away not in idx:
            continue
        mat = model.score_matrix(home, away, allow_unknown=True)
        flat = mat.ravel()
        flat = flat / flat.sum()
        draw = rng.choice(flat.size, size=n_sims, p=flat)
        hg = (draw // span).astype(np.int32)
        ag = (draw % span).astype(np.int32)

        hi, ai = idx[home], idx[away]
        home_win = hg > ag
        away_win = ag > hg
        tie = ~home_win & ~away_win

        points[hi] += np.where(home_win, 3, np.where(tie, 1, 0))
        points[ai] += np.where(away_win, 3, np.where(tie, 1, 0))
        gd[hi] += hg - ag
        gd[ai] += ag - hg
        gf[hi] += hg
        gf[ai] += ag

    # Rank each simulation. Sort key: points, then GD, then goals scored.
    # Scaled into one integer so a single argsort does the whole job.
    key = points.astype(np.int64) * 1_000_000 + (gd + 500) * 1_000 + gf
    order = np.argsort(-key, axis=0, kind="stable")
    positions = np.empty_like(order)
    rows = np.arange(n)[:, None]
    np.put_along_axis(positions, order, np.tile(rows, (1, n_sims)), axis=0)
    positions += 1  # 1-indexed finishing place

    pos_counts = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        pos_counts[i] = np.bincount(positions[i] - 1, minlength=n)

    results = []
    for t, i in idx.items():
        dist = pos_counts[i] / n_sims
        results.append(
            {
                "team": t,
                "current_points": int(table.loc[table["team"] == t, "points"].iloc[0]),
                "played": int(table.loc[table["team"] == t, "played"].iloc[0]),
                "proj_points": round(float(points[i].mean()), 1),
                "proj_points_p10": int(np.percentile(points[i], 10)),
                "proj_points_p90": int(np.percentile(points[i], 90)),
                "avg_finish": round(float(positions[i].mean()), 2),
                "title_pct": round(float(dist[0]) * 100, 1),
                "top4_pct": round(float(dist[:4].sum()) * 100, 1),
                "top6_pct": round(float(dist[:6].sum()) * 100, 1),
                "relegation_pct": round(float(dist[-3:].sum()) * 100, 1),
                "position_distribution": [round(float(x), 4) for x in dist],
            }
        )
    results.sort(key=lambda r: r["avg_finish"])
    return {
        "n_sims": n_sims,
        "n_remaining_fixtures": len(fixtures),
        "teams": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Monte Carlo season simulation.")
    ap.add_argument("--comp", default="E0", choices=list(cfg.MAIN_LEAGUES))
    ap.add_argument("--season", default=None, help="e.g. 2026-27 (default: latest)")
    ap.add_argument("--sims", type=int, default=cfg.N_SIMS)
    ap.add_argument(
        "--as-of",
        default=None,
        help="simulate as if it were this date (YYYY-MM-DD), ignoring all "
        "later results. Used to replay a past mid-season projection.",
    )
    args = ap.parse_args()

    matches = load_matches()
    comp_matches = matches[matches["comp"] == args.comp]
    season = args.season or comp_matches["season"].iloc[-1]

    played = comp_matches[comp_matches["season"] == season]
    if played.empty:
        print(f"No matches on record for {args.comp} {season}.")
        print("Before a season starts there is no team list to simulate. Once the")
        print("first round is played, run `python -m src.collect` then retry.")
        return 1

    teams = sorted(set(played["home"]) | set(played["away"]))
    cutoff = pd.Timestamp(args.as_of) if args.as_of else None
    as_of = cutoff if cutoff is not None else played["date"].max() + pd.Timedelta(days=1)

    model = DixonColes.fit(matches, comp=args.comp, as_of=as_of)
    table = current_table(matches, args.comp, season, as_of=cutoff, teams=teams)
    fixtures = remaining_fixtures(matches, args.comp, season, teams=teams, as_of=cutoff)

    print("=" * 74)
    print(f"STAGE 5 - SEASON SIMULATION  {cfg.LEAGUES[args.comp]['name']} {season}")
    print("=" * 74)
    if cutoff is not None:
        print(f"  AS OF {cutoff:%Y-%m-%d} - later results hidden from the model")
    print(f"  played {int(table['played'].sum() // 2)} matches, "
          f"{len(fixtures)} remaining, {args.sims:,} simulations\n")

    out = simulate(model, table, fixtures, n_sims=args.sims)

    hdr = (f"  {'#':>2} {'club':22} {'pl':>3} {'pts':>4} {'proj':>6} "
           f"{'p10-p90':>9} {'title':>7} {'top4':>7} {'rel':>7}")
    print(hdr)
    print("  " + "-" * 70)
    for i, r in enumerate(out["teams"], 1):
        print(
            f"  {i:>2} {r['team'][:22]:22} {r['played']:>3} {r['current_points']:>4} "
            f"{r['proj_points']:>6.1f} {r['proj_points_p10']:>4}-{r['proj_points_p90']:<4} "
            f"{r['title_pct']:>6.1f}% {r['top4_pct']:>6.1f}% {r['relegation_pct']:>6.1f}%"
        )

    dest = cfg.OUTPUT / f"season_{args.comp}_{season}.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"comp": args.comp, "season": season, **out}, fh, indent=2)
    print(f"\n  Written to {dest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
