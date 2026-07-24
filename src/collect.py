"""Stage 1 - data collection.

Downloads every season of every configured league from football-data.co.uk and
club strength ratings from ClubElo. Both are free and need no account.

Downloads are cached on disk. Completed seasons are never re-fetched; only the
current season and the Elo snapshot refresh on each run.

    python -m src.collect            # incremental
    python -m src.collect --refresh  # force re-download of everything
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from . import config as cfg

HEADERS = {"User-Agent": "match-prediction-v1 (github.com/Casanovaheer)"}
TIMEOUT = 30


def season_code(start_year: int) -> str:
    """1993 -> '9394', 2026 -> '2627'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """1993 -> '1993-94'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def _fetch(url: str, retries: int = 3) -> bytes | None:
    """GET with retries. Returns None on 404 (season not published yet)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"    ! failed after {retries} tries: {exc}", file=sys.stderr)
                return None
            time.sleep(2**attempt)
    return None


def download_league_season(div: str, start_year: int, refresh: bool = False) -> Path | None:
    """Download one league-season CSV. Returns the local path, or None."""
    code = season_code(start_year)
    dest = cfg.DATA_RAW / f"{div}_{code}.csv"

    # Completed seasons never change - don't re-download them.
    is_current = start_year >= cfg.LAST_SEASON - 1
    if dest.exists() and not refresh and not is_current:
        return dest

    url = f"{cfg.FD_BASE}/{code}/{div}.csv"
    content = _fetch(url)
    if content is None:
        return dest if dest.exists() else None

    # Guard against the site returning an HTML error page with a 200.
    head = content[:200].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        print(f"    ! {div} {code}: got HTML, not CSV - skipping")
        return dest if dest.exists() else None

    dest.write_bytes(content)
    return dest


def collect_football_data(refresh: bool = False) -> pd.DataFrame:
    """Download all league-seasons and stack them into one frame."""
    frames = []
    for div, meta in cfg.LEAGUES.items():
        print(f"\n  {div} - {meta['name']}")
        got = 0
        for year in range(cfg.FIRST_SEASON, cfg.LAST_SEASON + 1):
            path = download_league_season(div, year, refresh=refresh)
            if path is None:
                continue
            try:
                # football-data files are latin-1 and have occasional ragged
                # trailing commas; on_bad_lines='skip' tolerates those.
                df = pd.read_csv(
                    path,
                    encoding="latin-1",
                    on_bad_lines="skip",
                )
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                print(f"    ! {season_label(year)}: unreadable ({exc})")
                continue

            df = df.dropna(how="all")
            if df.empty or "HomeTeam" not in df.columns:
                continue

            # assign() returns a de-fragmented copy - avoids the insert-by-insert
            # PerformanceWarning on these 130-column frames.
            df = df.assign(Div=div, SeasonStart=year, Season=season_label(year))
            frames.append(df)
            got += 1
        print(f"    {got} seasons")

    if not frames:
        raise RuntimeError("No data downloaded - check network access.")

    return pd.concat(frames, ignore_index=True)


def collect_clubelo(clubs: list[str]) -> pd.DataFrame:
    """Current Elo rating for each club. Missing clubs are simply skipped."""
    rows = []
    for club in clubs:
        slug = club.replace(" ", "")
        content = _fetch(f"{cfg.CLUBELO_API}/{slug}")
        if content is None:
            continue
        try:
            hist = pd.read_csv(io.BytesIO(content))
        except pd.errors.ParserError:
            continue
        if hist.empty or "Elo" not in hist.columns:
            continue
        latest = hist.iloc[-1]
        rows.append({"club": club, "elo": float(latest["Elo"]), "as_of": latest.get("From")})
        time.sleep(0.15)  # be polite to a free API

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download raw football data.")
    ap.add_argument("--refresh", action="store_true", help="re-download everything")
    args = ap.parse_args()

    print("=" * 62)
    print("STAGE 1 - DATA COLLECTION")
    print("=" * 62)

    raw = collect_football_data(refresh=args.refresh)

    out = cfg.DATA_RAW / "all_matches_raw.parquet"
    try:
        raw.to_parquet(out, index=False)
    except (ImportError, ValueError):
        out = cfg.DATA_RAW / "all_matches_raw.csv"
        raw.to_csv(out, index=False)

    print("\n" + "-" * 62)
    print(f"  matches downloaded : {len(raw):,}")
    print(f"  leagues            : {raw['Div'].nunique()}")
    print(f"  seasons            : {raw['SeasonStart'].nunique()}")
    print(f"  saved to           : {out.name}")
    print("-" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
