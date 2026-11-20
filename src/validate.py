"""Data-quality checks run after every ingest. Any issue stops the pipeline before predictions are made."""
import numpy as np
import pandas as pd

from . import config


class DataQualityError(RuntimeError):
    pass


def check_data(con, today=None):
    """Return a list of human-readable data-quality issues (empty list = all good)."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    issues = []

    prices = pd.read_sql('SELECT * FROM prices', con, parse_dates=['date'])
    missing_tickers = set(config.TICKERS) - set(prices['ticker'])
    if missing_tickers:
        issues.append(f'No price data for: {sorted(missing_tickers)}')

    latest = prices.groupby('ticker')['date'].max()
    if latest.nunique() > 1:
        issues.append(f'Tickers end on different dates: {latest.dt.strftime("%Y-%m-%d").to_dict()}')
    if (today - latest.max()).days > config.MAX_PRICE_STALENESS_DAYS:
        issues.append(f'Price data is stale: latest {latest.max():%Y-%m-%d}')

    bad_bars = ((prices[['open', 'high', 'low', 'close', 'adj_close']] <= 0).any(axis=1)
                | (prices['high'] < prices['low'])).sum()
    if bad_bars:
        issues.append(f'{bad_bars} price rows with non-positive prices or high < low')

    prices = prices.sort_values(['ticker', 'date'])
    jumps = prices.groupby('ticker')['adj_close'].transform(lambda s: np.log(s).diff().abs())
    big = prices.loc[jumps > np.log(1 + config.MAX_ABS_DAILY_RETURN), ['ticker', 'date']]
    if not big.empty:
        issues.append(f'Suspicious >{config.MAX_ABS_DAILY_RETURN:.0%} daily moves: {big.head().to_dict("records")}')

    macro = pd.read_sql('SELECT series_id, MAX(date) AS last FROM macro WHERE value IS NOT NULL GROUP BY series_id',
                        con, parse_dates=['last']).set_index('series_id')['last']
    for series_id in config.FRED_SERIES:
        if series_id not in macro:
            issues.append(f'No data for FRED series {series_id}')
            continue
        limit = config.MAX_MONTHLY_MACRO_STALENESS_DAYS if series_id in config.MONTHLY_SERIES \
            else config.MAX_DAILY_MACRO_STALENESS_DAYS
        if (today - macro[series_id]).days > limit:
            issues.append(f'FRED {series_id} is stale: latest {macro[series_id]:%Y-%m-%d}')
    return issues


def validate(con, today=None):
    issues = check_data(con, today)
    if issues:
        raise DataQualityError('; '.join(issues))
