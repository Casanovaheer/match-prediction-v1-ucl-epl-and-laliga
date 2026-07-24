"""Stage 8 - publishable report.

Renders the current predictions into docs/index.md, which GitHub Pages serves
as the public site. Also writes docs/predictions.json for anything that wants
to consume the numbers directly.

    python -m src.report
"""

from __future__ import annotations

import json

import pandas as pd

from . import config as cfg
from .clean import display_name
from .model import DixonColes, load_matches
from .predict import upcoming_round
from .simulate import current_table, remaining_fixtures, simulate


def build() -> dict:
    matches = load_matches()
    as_of = matches["date"].max() + pd.Timedelta(days=1)
    payload: dict = {
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "data_through": matches["date"].max().strftime("%Y-%m-%d"),
        "competitions": {},
    }

    for comp, meta in cfg.MAIN_LEAGUES.items():
        model = DixonColes.fit(matches, comp=comp, as_of=as_of)
        season = matches[matches["comp"] == comp]["season"].iloc[-1]

        fixtures = upcoming_round(matches, comp, model)
        table = current_table(matches, comp, season)
        remaining = remaining_fixtures(matches, comp, season)
        sim = simulate(model, table, remaining, n_sims=cfg.N_SIMS)

        payload["competitions"][comp] = {
            "name": meta["name"],
            "season": season,
            "as_of": as_of.strftime("%Y-%m-%d"),
            "ratings": model.ratings_table().to_dict("records"),
            "fixtures": fixtures[:200],
            "season_projection": sim,
        }
    return payload


def to_markdown(p: dict) -> str:
    L: list[str] = []
    L.append("# Match Predictions — UCL, Premier League and La Liga\n")
    L.append(f"*Generated {p['generated']} · data through {p['data_through']}*\n")
    L.append(
        "Every number below comes from a time-weighted Dixon-Coles model fitted "
        "only on matches played before the prediction date. Method and measured "
        "accuracy are in the [README](https://github.com/Casanovaheer/"
        "match-prediction-v1-ucl-epl-and-laliga).\n"
    )
    L.append("**Measured out of sample over 8,360 matches:** 52.6% accuracy, "
             "0.9845 log-loss, versus 54.6% / 0.9586 for bookmaker closing odds.\n")

    for comp, c in p["competitions"].items():
        L.append(f"\n## {c['name']} {c['season']}\n")

        L.append("### Team ratings\n")
        L.append("| Club | Attack | Defence | Strength | xG scored (H) | xG conceded (H) |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for r in c["ratings"][:12]:
            L.append(
                f"| {display_name(r['team'])} | {r['attack']:+.3f} | "
                f"{r['defence']:+.3f} | {r['strength']:+.3f} | "
                f"{r['xg_scored_home']:.2f} | {r['xg_conceded_home']:.2f} |"
            )

        L.append("\n### Season projection\n")
        L.append("| # | Club | Pl | Pts | Proj | Title | Top 4 | Relegation |")
        L.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for i, t in enumerate(c["season_projection"]["teams"], 1):
            L.append(
                f"| {i} | {display_name(t['team'])} | {t['played']} | "
                f"{t['current_points']} | {t['proj_points']:.1f} | "
                f"{t['title_pct']:.1f}% | {t['top4_pct']:.1f}% | "
                f"{t['relegation_pct']:.1f}% |"
            )

        fx = c["fixtures"]
        if fx:
            L.append(f"\n### Upcoming fixtures ({len(fx)} unplayed)\n")
            L.append("| Fixture | Home | Draw | Away | Over 2.5 | BTTS | Likeliest |")
            L.append("|---|---:|---:|---:|---:|---:|---:|")
            for f in sorted(fx, key=lambda x: -x["p_home"])[:40]:
                top = f["top_scorelines"][0]["score"]
                L.append(
                    f"| {display_name(f['home'])} v {display_name(f['away'])} | "
                    f"{f['p_home']:.1%} | {f['p_draw']:.1%} | {f['p_away']:.1%} | "
                    f"{f['p_over_2_5']:.1%} | {f['p_btts']:.1%} | {top} |"
                )

    ucl_path = cfg.OUTPUT / "ucl_2026_27.json"
    if ucl_path.exists():
        with open(ucl_path, encoding="utf-8") as fh:
            ucl = json.load(fh)
        L.append("\n## Champions League 2026-27\n")
        L.append(
            "*Provisional 36-team field; the real draw is not made until late "
            "August. Each simulation performs its own pot-respecting draw, so "
            "draw uncertainty is included in these numbers.*\n"
        )
        L.append("| Club | Pot | Elo | Proj pts | Top 8 | R16 | QF | Win |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for t in ucl["teams"][:20]:
            L.append(
                f"| {t['team']} | {t['pot']} | {t['elo']:.0f} | "
                f"{t['proj_points']:.1f} | {t['top8_pct']:.1f}% | "
                f"{t['r16_pct']:.1f}% | {t['qf_pct']:.1f}% | {t['win_pct']:.1f}% |"
            )

    L.append(
        "\n---\n\n*These are probabilities, not tips. A 70% favourite loses "
        "three times in ten — that is the model working, not failing.*\n"
    )
    return "\n".join(L)


def main() -> int:
    print("=" * 62)
    print("STAGE 8 - REPORT")
    print("=" * 62)
    payload = build()

    with open(cfg.DOCS / "predictions.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with open(cfg.DOCS / "index.md", "w", encoding="utf-8") as fh:
        fh.write(to_markdown(payload))

    for comp, c in payload["competitions"].items():
        print(f"  {comp}: {len(c['fixtures'])} fixtures, "
              f"{len(c['ratings'])} rated teams")
    print(f"\n  Written to docs/index.md and docs/predictions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
