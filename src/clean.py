"""Stage 1b - cleaning and normalisation.

Turns the raw football-data.co.uk dump into one tidy match table, and refuses
to hand it on if the sanity checks fail.

The single most damaging bug in a project like this is a club whose name is
spelled two ways: its history silently splits in half and both halves get
mediocre ratings. So name handling is explicit and audited, never fuzzy.

    python -m src.clean
"""

from __future__ import annotations

import difflib
import re
import sys
import warnings

import numpy as np
import pandas as pd

from . import config as cfg

warnings.simplefilter("ignore", category=FutureWarning)

# --------------------------------------------------------------- name handling

# Confirmed source-data typos. Left side is wrong, right side is canonical.
# Every entry here was verified by hand - see DO_NOT_MERGE for why that matters.
NAME_FIXES: dict[str, str] = {
    # Spain - 38 matches were filed under a missing 'r'
    "Villareal": "Villarreal",
    # Germany - capitalisation drift
    "M'Gladbach": "M'gladbach",
    # Greece - alternate transliterations of the same club
    "Levadeiakos": "Levadiakos",
    "Veroia": "Veria",
}

# Pairs that LOOK like typos to a string-similarity check but are genuinely
# different clubs. Documented so nobody "helpfully" merges them later.
DO_NOT_MERGE: list[tuple[str, str, str]] = [
    ("Reggiana", "Reggina", "Reggio Emilia vs Reggio Calabria - different clubs"),
    ("Athinaikos", "Panathinaikos", "different Athens clubs"),
]

# Display names. The source uses terse forms; these are for output only and
# never used as join keys.
DISPLAY_NAMES: dict[str, str] = {
    "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao",
    "Sociedad": "Real Sociedad",
    "Espanol": "Espanyol",
    "Betis": "Real Betis",
    "Vallecano": "Rayo Vallecano",
    "La Coruna": "Deportivo La Coruna",
    "Sp Gijon": "Sporting Gijon",
    "Celta": "Celta Vigo",
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "West Brom": "West Bromwich Albion",
    "Sheffield Weds": "Sheffield Wednesday",
    "Wolves": "Wolverhampton",
    "QPR": "Queens Park Rangers",
    "M'gladbach": "Borussia Monchengladbach",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Bayern Munich": "Bayern Munich",
    "Paris SG": "Paris Saint-Germain",
}

_WS = re.compile(r"\s+")


def normalise_name(name: object) -> str | None:
    """Canonical join key for a club. Whitespace-safe and typo-corrected."""
    if not isinstance(name, str):
        return None
    cleaned = _WS.sub(" ", name).strip()
    if not cleaned:
        return None
    return NAME_FIXES.get(cleaned, cleaned)


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


# ---------------------------------------------------------------------- odds

# Preference order for the market benchmark. "C" columns are CLOSING odds -
# the sharpest number a bookmaker publishes, and the fair thing to be judged
# against. Everything else is a fallback for older seasons.
ODDS_SETS: list[tuple[str, tuple[str, str, str]]] = [
    ("avg_closing", ("AvgCH", "AvgCD", "AvgCA")),
    ("b365_closing", ("B365CH", "B365CD", "B365CA")),
    ("pinnacle_closing", ("PSCH", "PSCD", "PSCA")),
    ("avg_prematch", ("AvgH", "AvgD", "AvgA")),
    ("betbrain_avg", ("BbAvH", "BbAvD", "BbAvA")),
    ("b365_prematch", ("B365H", "B365D", "B365A")),
    ("williamhill", ("WHH", "WHD", "WHA")),
]


def attach_market_odds(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Pick the best available odds per row and de-vig them into probabilities."""
    n = len(df)
    odds = np.full((n, 3), np.nan)
    source = np.array([""] * n, dtype=object)

    for label, cols in ODDS_SETS:
        if not all(c in raw.columns for c in cols):
            continue
        block = raw[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        # Valid only if all three prices exist and are sane (>1.01).
        ok = np.isfinite(block).all(axis=1) & (block > 1.01).all(axis=1)
        fill = ok & ~np.isfinite(odds).all(axis=1)
        odds[fill] = block[fill]
        source[fill] = label

    # Remove the bookmaker margin so the three probabilities sum to exactly 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = 1.0 / odds
        overround = implied.sum(axis=1, keepdims=True)
        fair = implied / overround
    vig = overround.ravel() - 1.0

    # A real book always carries a margin. Odds implying under 100% would be a
    # free arbitrage, so those rows are corrupt prices in the source, not value.
    # Void the odds but keep the match - the result is still a valid training row.
    corrupt = np.isfinite(vig) & (vig <= 0.0)
    fair[corrupt] = np.nan
    odds[corrupt] = np.nan
    vig[corrupt] = np.nan
    source[corrupt] = ""

    df["odds_h"], df["odds_d"], df["odds_a"] = odds[:, 0], odds[:, 1], odds[:, 2]
    df["odds_source"] = np.where(source == "", None, source)
    df["p_h_mkt"], df["p_d_mkt"], df["p_a_mkt"] = fair[:, 0], fair[:, 1], fair[:, 2]
    df["overround"] = vig
    df.attrs["corrupt_odds"] = int(corrupt.sum())
    return df


# ---------------------------------------------------------------------- build


def build() -> pd.DataFrame:
    src = cfg.DATA_RAW / "all_matches_raw.csv"
    if not src.exists():
        src = cfg.DATA_RAW / "all_matches_raw.parquet"
    if not src.exists():
        raise FileNotFoundError("Run `python -m src.collect` first.")

    raw = (
        pd.read_parquet(src)
        if src.suffix == ".parquet"
        else pd.read_csv(src, low_memory=False)
    )

    df = pd.DataFrame(
        {
            # NB: named 'comp', never 'div' - DataFrame.div is pandas' division
            # method, so df.div would silently return a function, not a column.
            "comp": raw["Div"],
            "season": raw["Season"],
            "season_start": raw["SeasonStart"].astype(int),
            "home": raw["HomeTeam"].map(normalise_name),
            "away": raw["AwayTeam"].map(normalise_name),
            "hg": pd.to_numeric(raw["FTHG"], errors="coerce"),
            "ag": pd.to_numeric(raw["FTAG"], errors="coerce"),
        }
    )

    # Mixed dd/mm/yy and dd/mm/yyyy across eras.
    df["date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce")

    df["league"] = df["comp"].map(lambda d: cfg.LEAGUES.get(d, {}).get("name", d))
    df["country"] = df["comp"].map(lambda d: cfg.LEAGUES.get(d, {}).get("country", "???"))
    df["is_main"] = df["comp"].isin(cfg.MAIN_LEAGUES)

    df = attach_market_odds(df, raw)
    corrupt_odds = df.attrs.get("corrupt_odds", 0)

    # Drop anything unusable as a training row.
    before = len(df)
    df = df.dropna(subset=["date", "home", "away", "hg", "ag"])
    df = df[df["home"] != df["away"]]
    dropped = before - len(df)

    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df["result"] = np.select(
        [df.hg > df.ag, df.hg == df.ag], ["H", "D"], default="A"
    )
    df["total_goals"] = df.hg + df.ag
    df["home_disp"] = df["home"].map(display_name)
    df["away_disp"] = df["away"].map(display_name)

    df = df.sort_values(["date", "comp", "home"]).reset_index(drop=True)
    df.attrs["dropped"] = dropped
    df.attrs["corrupt_odds"] = corrupt_odds
    return df


# ------------------------------------------------------------------ validation


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of problems. Empty list means the data is trustworthy."""
    problems: list[str] = []

    if df[["date", "home", "away", "hg", "ag"]].isna().any().any():
        problems.append("null values remain in required columns")

    if (df.hg < 0).any() or (df.ag < 0).any():
        problems.append("negative scorelines present")

    if (df.home == df.away).any():
        problems.append("a team is playing itself")

    dupes = df.duplicated(subset=["date", "comp", "home", "away"]).sum()
    if dupes:
        problems.append(f"{dupes} duplicate fixtures (same date, same two teams)")

    # Market probabilities must sum to 1 wherever odds exist.
    has_odds = df.p_h_mkt.notna()
    if has_odds.any():
        s = df.loc[has_odds, ["p_h_mkt", "p_d_mkt", "p_a_mkt"]].sum(axis=1)
        if not np.allclose(s, 1.0, atol=1e-6):
            problems.append("de-vigged market probabilities do not sum to 1")
        vig = df.loc[has_odds, "overround"]
        if (vig <= 0).any() or (vig > 0.5).any():
            problems.append("implausible bookmaker overround survived cleaning")

    # Implausible goal counts - real record for these leagues is in the low teens.
    if (df.hg > 15).any() or (df.ag > 15).any():
        problems.append("scoreline above 15 goals - likely a parsing error")

    # Re-run the near-duplicate name scan as a standing guard.
    known = {tuple(sorted((a, b))) for a, b, _ in DO_NOT_MERGE}
    for comp in df["comp"].unique():
        sub = df[df["comp"] == comp]
        names = sorted(set(sub.home) | set(sub.away))
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if tuple(sorted((a, b))) in known:
                    continue
                if difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.92:
                    problems.append(f"possible unmerged duplicate in {comp}: {a!r} / {b!r}")
    return problems


def main() -> int:
    print("=" * 62)
    print("STAGE 1b - CLEAN AND VALIDATE")
    print("=" * 62)

    df = build()
    problems = validate(df)

    out = cfg.DATA_PROC / "matches.csv"
    df.to_csv(out, index=False)

    odds_cov = df.p_h_mkt.notna().mean()
    print(f"\n  clean matches      : {len(df):,}")
    print(f"  dropped as unusable: {df.attrs['dropped']:,}")
    print(f"  odds voided as bad : {df.attrs['corrupt_odds']:,}")
    print(f"  date range         : {df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}")
    print(f"  leagues            : {df['comp'].nunique()}")
    print(f"  distinct clubs     : {len(set(df.home) | set(df.away)):,}")
    print(f"  with market odds   : {odds_cov:.1%}")
    print(f"  mean overround     : {df.overround.mean():.3f}")
    print(f"  home win rate      : {(df.result == 'H').mean():.1%}")
    print(f"  draw rate          : {(df.result == 'D').mean():.1%}")
    print(f"  away win rate      : {(df.result == 'A').mean():.1%}")
    print(f"  goals per game     : {df.total_goals.mean():.2f}")
    print(f"  saved to           : {out}")

    print("\n" + "-" * 62)
    if problems:
        print(f"  VALIDATION FAILED - {len(problems)} problem(s):")
        for p in problems:
            print(f"    x {p}")
        print("-" * 62)
        return 1
    print("  VALIDATION PASSED - all checks clean")
    print("-" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
