"""Central configuration shared by the pipeline and the notebook."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / 'db' / 'market.db'          # single-file SQLite database
MODEL_DIR = ROOT / 'models'                  # versioned model files + latest.json pointer
LOG_DIR = ROOT / 'logs'

# ---- Universe ----
TICKERS = {
    'SPY': 'Broad Market ETF',
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'JPM': 'Financials',
    'XOM': 'Energy',
    'JNJ': 'Health Care',
}
HISTORY_START = '2010-01-01'
MACRO_START = '2008-01-01'        # extra history so 12-month CPI change exists from 2010

# ---- FRED series ----
FRED_SERIES = {
    'DGS10': '10-Year Treasury yield (%)',
    'DGS2': '2-Year Treasury yield (%)',
    'T10Y2Y': '10Y minus 2Y spread (pp)',
    'VIXCLS': 'CBOE VIX index',
    'FEDFUNDS': 'Effective Fed funds rate (%)',
    'CPIAUCSL': 'CPI, all urban consumers (index)',
    'UNRATE': 'Unemployment rate (%)',
}
MONTHLY_SERIES = ['FEDFUNDS', 'CPIAUCSL', 'UNRATE']
DAILY_SERIES = [s for s in FRED_SERIES if s not in MONTHLY_SERIES]

# ---- Incremental-load settings ----
PRICE_OVERLAP_DAYS = 10     # re-download the last N calendar days to pick up late corrections
FRED_REFRESH_DAYS = 400     # re-download ~13 months of FRED: monthly data (CPI, jobs) gets revised

# ---- Point-in-time availability (prevents look-ahead bias) ----
FRED_DAILY_LAG_BDAYS = 1                        # FRED posts daily yields / VIX the next business day
MONTHLY_RELEASE_LAG = dict(months=1, days=14)   # a monthly value is published ~6 weeks after its reference date

# ---- Data-quality thresholds ----
MAX_PRICE_STALENESS_DAYS = 5      # calendar days (covers weekends + one holiday)
MAX_DAILY_MACRO_STALENESS_DAYS = 7
MAX_MONTHLY_MACRO_STALENESS_DAYS = 75
MAX_ABS_DAILY_RETURN = 0.40       # larger one-day moves are treated as data errors

# ---- Model ----
RETRAIN_EVERY_DAYS = 7
CV_SPLITS = 5
C_GRID = [0.001, 0.01, 0.1, 1.0]
RANDOM_STATE = 42
