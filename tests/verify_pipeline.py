"""End-to-end checks across the whole pipeline.

    python -m tests.verify_pipeline

Complements tests/verify_model.py, which covers the model itself. This one
checks that the stages downstream of it hold together: the season simulator's
probabilities must be coherent, the UCL bracket must conserve teams, and the
prediction CLI must fail helpfully rather than guess.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src import config as cfg
from src.model import DixonColes, load_matches
from src.predict import resolve_team
from src.simulate import current_table, remaining_fixtures, simulate

problems: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'[ok]' if ok else '[XX]'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        problems.append(name)


def main() -> int:
    print("=" * 66)
    print("VERIFY - FULL PIPELINE")
    print("=" * 66)

    matches = load_matches()
    season = "2025-26"
    comp = "SP1"
    cutoff = pd.Timestamp("2026-01-01")

    print("\n-- season simulator --")
    played = matches[(matches["comp"] == comp) & (matches["season"] == season)]
    teams = sorted(set(played["home"]) | set(played["away"]))
    model = DixonColes.fit(matches, comp=comp, as_of=cutoff)
    table = current_table(matches, comp, season, as_of=cutoff, teams=teams)
    fixtures = remaining_fixtures(matches, comp, season, teams=teams, as_of=cutoff)
    sim = simulate(model, table, fixtures, n_sims=2000)

    rows = sim["teams"]
    check("every club appears exactly once", len(rows) == len(teams),
          f"{len(rows)} rows for {len(teams)} clubs")

    title = sum(r["title_pct"] for r in rows)
    check("title probabilities sum to 100%", abs(title - 100) < 0.6, f"sum={title:.2f}%")

    top4 = sum(r["top4_pct"] for r in rows)
    check("top-4 probabilities sum to 400%", abs(top4 - 400) < 2.0, f"sum={top4:.1f}%")

    rel = sum(r["relegation_pct"] for r in rows)
    check("relegation probabilities sum to 300%", abs(rel - 300) < 2.0, f"sum={rel:.1f}%")

    for r in rows:
        dist = sum(r["position_distribution"])
        if abs(dist - 1.0) > 1e-6:
            check(f"{r['team']} position distribution sums to 1", False, f"{dist}")
            break
    else:
        check("every position distribution sums to 1", True)

    check("nobody can both win the league and be relegated",
          not any(r["title_pct"] > 1 and r["relegation_pct"] > 1 for r in rows))

    check("projected points bracketed by p10 and p90",
          all(r["proj_points_p10"] <= r["proj_points"] <= r["proj_points_p90"]
              for r in rows))

    # The known answer: Barcelona won 2025-26. A January projection should have
    # had them clear favourites.
    barca = next(r for r in rows if r["team"] == "Barcelona")
    check("January projection made the eventual champion favourite",
          barca["title_pct"] > 50,
          f"Barcelona title={barca['title_pct']:.1f}%, proj={barca['proj_points']:.1f} "
          f"(actual finish: 94 pts, champions)")

    print("\n-- no look-ahead in the simulator --")
    check("as-of table ignores later results",
          int(table["played"].sum() // 2) < 380,
          f"{int(table['played'].sum() // 2)} of 380 matches counted")
    check("remaining fixtures complete the season",
          int(table["played"].sum() // 2) + len(fixtures) == 380,
          f"{int(table['played'].sum() // 2)} + {len(fixtures)} = "
          f"{int(table['played'].sum() // 2) + len(fixtures)}")

    print("\n-- team name resolution --")
    check("exact name resolves", resolve_team("Barcelona", model.teams) == "Barcelona")
    check("lowercase resolves", resolve_team("barcelona", model.teams) == "Barcelona")
    check("partial resolves", resolve_team("Vallecano", model.teams) == "Vallecano")
    try:
        resolve_team("Notarealclub", model.teams)
        check("unknown name rejected with a hint", False, "no error raised")
    except SystemExit as exc:
        check("unknown name rejected with a hint", "No team matching" in str(exc))

    print("\n-- Champions League output --")
    ucl_path = cfg.OUTPUT / "ucl_2026_27.json"
    if not ucl_path.exists():
        print("  [--]  skipped: run `python -m src.ucl` first")
    else:
        with open(ucl_path, encoding="utf-8") as fh:
            ucl = json.load(fh)
        u = ucl["teams"]
        check("36 clubs in the field", len(u) == 36, f"{len(u)}")
        w = sum(t["win_pct"] for t in u)
        check("winner probabilities sum to 100%", abs(w - 100) < 1.5, f"sum={w:.1f}%")
        check("survival is monotonic across rounds",
              all(t["r16_pct"] >= t["qf_pct"] >= t["sf_pct"] >= t["final_pct"]
                  >= t["win_pct"] for t in u))
        r16 = sum(t["r16_pct"] for t in u)
        check("16 clubs reach the R16", abs(r16 - 1600) < 15, f"sum={r16:.0f}%")
        check("pots are 9 clubs each",
              all(sum(1 for t in u if t["pot"] == p) == 9 for p in (1, 2, 3, 4)))

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
