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


HTML_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Match Predictions — UCL, Premier League, La Liga</title>
<style>
:root{--bg:#f1f3f2;--panel:#fff;--ink:#141a19;--soft:#4a5654;--faint:#7c8886;
--rule:#d6dcda;--accent:#0e6e62;--signal:#b4551c}
@media(prefers-color-scheme:dark){:root{--bg:#0e1413;--panel:#161d1c;--ink:#e4eae8;
--soft:#a3afac;--faint:#75817e;--rule:#2a3433;--accent:#4fb0a0;--signal:#d8834a}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;padding:0 1rem 5rem;
font:16px/1.6 "Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:60rem;margin:0 auto}
header{padding:3rem 0 1.5rem;border-bottom:2px solid var(--ink)}
h1{font:400 clamp(1.8rem,4vw,2.6rem)/1.15 Georgia,serif;margin:0 0 .6rem;
letter-spacing:-.02em}
.sub{color:var(--soft);margin:0}
h2{font:400 1.5rem/1.25 Georgia,serif;margin:2.5rem 0 .3rem;
border-bottom:1px solid var(--rule);padding-bottom:.4rem}
h3{font-size:1rem;font-weight:600;margin:1.8rem 0 .5rem;color:var(--soft)}
.scroll{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);
border-radius:4px;margin:.6rem 0}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:.5rem .7rem;text-align:left;border-bottom:1px solid var(--rule);
white-space:nowrap}
th{font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;
color:var(--faint);font-weight:400}
td.n{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:none}
.acc{background:var(--panel);border:1px solid var(--rule);border-radius:4px;
padding:1rem 1.2rem;margin:1.2rem 0}
.acc b{color:var(--accent)}
.note{border-left:3px solid var(--signal);padding:.7rem 1rem;margin:1.2rem 0;
color:var(--soft);font-size:.92rem}
.bar{display:flex;height:5px;border-radius:3px;overflow:hidden;min-width:90px}
.bar i{display:block;height:100%}
footer{margin-top:3.5rem;padding-top:1.2rem;border-top:2px solid var(--ink);
color:var(--faint);font-size:.82rem}
</style></head><body><div class="wrap">
"""


def _bar(h: float, d: float, a: float) -> str:
    return (
        f'<span class="bar"><i style="width:{h * 100:.0f}%;background:var(--accent)"></i>'
        f'<i style="width:{d * 100:.0f}%;background:var(--faint)"></i>'
        f'<i style="width:{a * 100:.0f}%;background:var(--signal)"></i></span>'
    )


def to_html(p: dict) -> str:
    """A self-contained page. Double-click it; no server, no internet needed."""
    H = [HTML_HEAD]
    H.append("<header><h1>Match Predictions</h1>")
    H.append(
        f'<p class="sub">UCL · Premier League · La Liga &nbsp;·&nbsp; '
        f'generated {p["generated"]} &nbsp;·&nbsp; data through '
        f'{p["data_through"]}</p></header>'
    )
    H.append(
        '<div class="acc"><b>Measured accuracy, out of sample over 8,360 '
        "matches:</b> 52.6% correct, 0.9845 log-loss. Bookmaker closing odds "
        "score 54.6% / 0.9586 on the same fixtures. Anything claiming 70%+ is "
        "lying to you.</div>"
    )

    for comp, c in p["competitions"].items():
        H.append(f"<h2>{c['name']} {c['season']}</h2>")

        fx = c["fixtures"]
        if fx:
            H.append(f"<h3>Upcoming fixtures — {len(fx)} unplayed</h3>")
            H.append('<div class="scroll"><table><thead><tr>'
                     "<th>Fixture</th><th></th><th>Home</th><th>Draw</th>"
                     "<th>Away</th><th>Over 2.5</th><th>BTTS</th>"
                     "<th>Likeliest</th></tr></thead><tbody>")
            for f in sorted(fx, key=lambda x: -x["p_home"])[:60]:
                H.append(
                    f"<tr><td>{display_name(f['home'])} v {display_name(f['away'])}</td>"
                    f"<td>{_bar(f['p_home'], f['p_draw'], f['p_away'])}</td>"
                    f"<td class='n'>{f['p_home']:.1%}</td>"
                    f"<td class='n'>{f['p_draw']:.1%}</td>"
                    f"<td class='n'>{f['p_away']:.1%}</td>"
                    f"<td class='n'>{f['p_over_2_5']:.1%}</td>"
                    f"<td class='n'>{f['p_btts']:.1%}</td>"
                    f"<td class='n'>{f['top_scorelines'][0]['score']}</td></tr>"
                )
            H.append("</tbody></table></div>")
        else:
            H.append(
                '<div class="note">No unplayed fixtures on record. The 2026-27 '
                "season has not kicked off yet — this fills in automatically "
                "from the first matchday.</div>"
            )

        H.append("<h3>Season projection</h3>")
        H.append('<div class="scroll"><table><thead><tr><th>#</th><th>Club</th>'
                 "<th>Pl</th><th>Pts</th><th>Proj</th><th>Title</th>"
                 "<th>Top 4</th><th>Relegation</th></tr></thead><tbody>")
        for i, t in enumerate(c["season_projection"]["teams"], 1):
            H.append(
                f"<tr><td class='n'>{i}</td><td>{display_name(t['team'])}</td>"
                f"<td class='n'>{t['played']}</td>"
                f"<td class='n'>{t['current_points']}</td>"
                f"<td class='n'>{t['proj_points']:.1f}</td>"
                f"<td class='n'>{t['title_pct']:.1f}%</td>"
                f"<td class='n'>{t['top4_pct']:.1f}%</td>"
                f"<td class='n'>{t['relegation_pct']:.1f}%</td></tr>"
            )
        H.append("</tbody></table></div>")

        H.append("<h3>Team ratings</h3>")
        H.append('<div class="scroll"><table><thead><tr><th>Club</th>'
                 "<th>Attack</th><th>Defence</th><th>Strength</th>"
                 "<th>xG scored (H)</th><th>xG conceded (H)</th>"
                 "</tr></thead><tbody>")
        for r in c["ratings"][:14]:
            H.append(
                f"<tr><td>{display_name(r['team'])}</td>"
                f"<td class='n'>{r['attack']:+.3f}</td>"
                f"<td class='n'>{r['defence']:+.3f}</td>"
                f"<td class='n'>{r['strength']:+.3f}</td>"
                f"<td class='n'>{r['xg_scored_home']:.2f}</td>"
                f"<td class='n'>{r['xg_conceded_home']:.2f}</td></tr>"
            )
        H.append("</tbody></table></div>")

    ucl_path = cfg.OUTPUT / "ucl_2026_27.json"
    if ucl_path.exists():
        with open(ucl_path, encoding="utf-8") as fh:
            ucl = json.load(fh)
        H.append("<h2>Champions League 2026-27</h2>")
        H.append(
            '<div class="note">Provisional 36-team field — the real draw is not '
            "made until late August. Each simulation runs its own pot-respecting "
            "draw, so draw uncertainty is already inside these numbers.</div>"
        )
        H.append('<div class="scroll"><table><thead><tr><th>Club</th><th>Pot</th>'
                 "<th>Elo</th><th>Proj pts</th><th>Top 8</th><th>R16</th>"
                 "<th>QF</th><th>SF</th><th>Win</th></tr></thead><tbody>")
        for t in ucl["teams"]:
            H.append(
                f"<tr><td>{t['team']}</td><td class='n'>{t['pot']}</td>"
                f"<td class='n'>{t['elo']:.0f}</td>"
                f"<td class='n'>{t['proj_points']:.1f}</td>"
                f"<td class='n'>{t['top8_pct']:.1f}%</td>"
                f"<td class='n'>{t['r16_pct']:.1f}%</td>"
                f"<td class='n'>{t['qf_pct']:.1f}%</td>"
                f"<td class='n'>{t['sf_pct']:.1f}%</td>"
                f"<td class='n'>{t['win_pct']:.1f}%</td></tr>"
            )
        H.append("</tbody></table></div>")

    H.append(
        "<footer>These are probabilities, not tips. A 70% favourite loses three "
        "times in ten — that is the model working, not failing.</footer>"
    )
    H.append("</div></body></html>")
    return "\n".join(H)


def main() -> int:
    print("=" * 62)
    print("STAGE 8 - REPORT")
    print("=" * 62)
    payload = build()

    with open(cfg.DOCS / "predictions.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with open(cfg.DOCS / "index.md", "w", encoding="utf-8") as fh:
        fh.write(to_markdown(payload))
    with open(cfg.DOCS / "index.html", "w", encoding="utf-8") as fh:
        fh.write(to_html(payload))

    for comp, c in payload["competitions"].items():
        print(f"  {comp}: {len(c['fixtures'])} fixtures, "
              f"{len(c['ratings'])} rated teams")
    print(f"\n  Written to docs/index.html, docs/index.md, docs/predictions.json")
    print("  Double-click docs/index.html to view the results in a browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
