"""Live connectivity check: can this machine reach Yahoo Finance and FRED?

Run from the repo root:   python scripts/check_sources.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingest import fetch_fred, fetch_prices_yahoo  # noqa: E402

start = (pd.Timestamp.today() - pd.Timedelta(days=14)).strftime('%Y-%m-%d')
ok = True
for name, call in [('Yahoo Finance (SPY)', lambda: fetch_prices_yahoo('SPY', start)),
                   ('FRED (DGS10)', lambda: fetch_fred('DGS10', start))]:
    try:
        data = call()
        latest = data.dropna().iloc[-1]
        print(f'OK    {name:20s} {len(data)} rows, latest {latest["date"]:%Y-%m-%d}')
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f'FAIL  {name:20s} {type(exc).__name__}: {exc}')
sys.exit(0 if ok else 1)
