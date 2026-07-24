"""Central configuration. Everything tunable lives here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
DOCS = ROOT / "docs"

for _d in (DATA_RAW, DATA_PROC, OUTPUT, DOCS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data sources

FD_BASE = "https://www.football-data.co.uk/mmz4281"

# football-data.co.uk division codes.
# MAIN are the two leagues we predict match-by-match and simulate tables for.
MAIN_LEAGUES = {
    "E0": {"name": "Premier League", "country": "ENG", "teams": 20},
    "SP1": {"name": "La Liga", "country": "ESP", "teams": 20},
}

# EXTRA leagues exist only to rate clubs that appear in the Champions League.
# Without them a UCL model has nothing to say about Bayern or Inter.
EXTRA_LEAGUES = {
    "I1": {"name": "Serie A", "country": "ITA", "teams": 20},
    "D1": {"name": "Bundesliga", "country": "GER", "teams": 18},
    "F1": {"name": "Ligue 1", "country": "FRA", "teams": 18},
    "N1": {"name": "Eredivisie", "country": "NED", "teams": 18},
    "P1": {"name": "Primeira Liga", "country": "POR", "teams": 18},
    "B1": {"name": "Belgian Pro League", "country": "BEL", "teams": 16},
    "T1": {"name": "Super Lig", "country": "TUR", "teams": 19},
    "G1": {"name": "Super League Greece", "country": "GRE", "teams": 14},
    "SC0": {"name": "Scottish Premiership", "country": "SCO", "teams": 12},
}

LEAGUES = {**MAIN_LEAGUES, **EXTRA_LEAGUES}

# Season codes: "9394" .. "2627". Shots/cards data begins 2000-01;
# earlier seasons still give results, which is all the core model needs.
FIRST_SEASON = 1993
LAST_SEASON = 2026  # start year of the 2026-27 season

CLUBELO_API = "http://api.clubelo.com"

# ---------------------------------------------------------------- model params

# Exponential time decay on match weight. A match this many days old counts
# half as much as one played today. Dixon-Coles used ~0.0065/day; 180 days
# half-life (xi ~= 0.00385) tests better on modern data.
HALF_LIFE_DAYS = 180

# Only fit on matches inside this window. Keeps the team set manageable and
# stops 1990s sides polluting the parameter space.
LOOKBACK_DAYS = 1825  # 5 seasons

# Largest scoreline the matrix models. 0-10 covers >99.9% of real results.
MAX_GOALS = 10

# Monte Carlo runs for season simulation.
N_SIMS = 10000

# Reproducibility.
SEED = 1993

# ---------------------------------------------------------------- competitions

# UEFA Champions League league-phase format (since 2024-25)
UCL_TEAMS = 36
UCL_LEAGUE_PHASE_MATCHES = 8
UCL_AUTO_QUALIFY = 8  # top 8 go straight to R16
UCL_PLAYOFF_SPOTS = 16  # places 9-24 enter the knockout playoff
