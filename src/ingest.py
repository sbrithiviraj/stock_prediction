"""Download Yahoo Finance prices and FRED macro series and load them incrementally into the database.

Incremental logic
-----------------
* Prices: download only from (last stored date - PRICE_OVERLAP_DAYS). The overlap re-checks recent rows.
  Yahoo's *adjusted* close for every past day changes after a dividend or split, so if the overlapping rows
  no longer match what is stored, that ticker's whole history is reloaded.
* FRED: re-download the last FRED_REFRESH_DAYS, because monthly figures (CPI, unemployment) are revised.
Every write is an upsert, so running the pipeline twice on the same day is harmless.
"""
import logging
import time

import pandas as pd

from . import config
from .db import upsert, utc_now

log = logging.getLogger(__name__)
PRICE_COLUMNS = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']


def _retry(fn, *args, attempts=3, wait_seconds=10):
    """Call fn, retrying on network errors (Yahoo occasionally rate-limits)."""
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - any network/parse error is retried
            if attempt == attempts:
                raise
            log.warning('%s%s failed (%s); retry %d/%d in %ss', fn.__name__, args, exc, attempt, attempts, wait_seconds)
            time.sleep(wait_seconds * attempt)


# ------------------------------------------------------------------ source adapters
def fetch_prices_yahoo(ticker, start):
    """Daily OHLCV + adjusted close for one ticker from `start` to today."""
    import yfinance as yf
    data = yf.download(ticker, start=start, auto_adjust=False, actions=False, progress=False)
    if data.empty:   # yfinance logs network errors instead of raising them; raise so _retry() tries again
        raise ConnectionError(f'Yahoo Finance returned no rows for {ticker} from {start}')
    if isinstance(data.columns, pd.MultiIndex):        # newer yfinance: (field, ticker) columns
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index().rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
                                              'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'})
    data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None).dt.normalize()
    data['ticker'] = ticker
    return data[PRICE_COLUMNS]


def fetch_fred(series_id, start):
    """One FRED series via the public CSV endpoint (no API key needed)."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}'
    data = pd.read_csv(url)
    data.columns = ['date', 'value']
    data['date'] = pd.to_datetime(data['date'])
    data['value'] = pd.to_numeric(data['value'], errors='coerce')   # FRED uses '.' for missing
    data['series_id'] = series_id
    return data[['date', 'series_id', 'value']]


# ------------------------------------------------------------------ loaders
def _history_changed(stored, fresh, tol=1e-4):
    """True if overlapping rows disagree, i.e. Yahoo re-adjusted history (dividend/split)."""
    both = stored.merge(fresh, on='date', suffixes=('_old', '_new'))
    if both.empty:
        return False
    for col in ('adj_close', 'close'):
        rel_diff = (both[f'{col}_new'] / both[f'{col}_old'] - 1).abs()
        if (rel_diff > tol).any():
            return True
    return False


def ingest_prices(con, fetch=fetch_prices_yahoo, tickers=config.TICKERS):
    total = 0
    for ticker in tickers:
        last = con.execute('SELECT MAX(date) FROM prices WHERE ticker = ?', (ticker,)).fetchone()[0]
        if last is None:
            fresh, mode = _retry(fetch, ticker, config.HISTORY_START), 'backfill'
        else:
            start = (pd.Timestamp(last) - pd.Timedelta(days=config.PRICE_OVERLAP_DAYS)).strftime('%Y-%m-%d')
            fresh, mode = _retry(fetch, ticker, start), 'incremental'
            stored = pd.read_sql('SELECT date, close, adj_close FROM prices WHERE ticker = ? AND date >= ?',
                                 con, params=(ticker, start), parse_dates=['date'])
            if _history_changed(stored, fresh[['date', 'close', 'adj_close']]):
                log.info('%s: adjusted history changed (dividend/split) -> full reload', ticker)
                fresh, mode = _retry(fetch, ticker, config.HISTORY_START), 'full reload'
                with con:
                    con.execute('DELETE FROM prices WHERE ticker = ?', (ticker,))
        fresh = fresh.dropna(subset=['close']).assign(ingested_at=utc_now())
        n = upsert(con, 'prices', fresh)
        log.info('%-5s %-11s %5d rows (latest %s)', ticker, mode, n,
                 fresh['date'].max().date() if n else 'n/a')
        total += n
    return total


def ingest_macro(con, fetch=fetch_fred, series_ids=config.FRED_SERIES):
    total = 0
    for series_id in series_ids:
        last = con.execute('SELECT MAX(date) FROM macro WHERE series_id = ?', (series_id,)).fetchone()[0]
        start = config.MACRO_START if last is None else \
            (pd.Timestamp(last) - pd.Timedelta(days=config.FRED_REFRESH_DAYS)).strftime('%Y-%m-%d')
        fresh = _retry(fetch, series_id, start).assign(ingested_at=utc_now())
        n = upsert(con, 'macro', fresh)
        log.info('%-8s %5d rows (latest %s)', series_id, n, fresh.dropna(subset=['value'])['date'].max().date())
        total += n
    return total
