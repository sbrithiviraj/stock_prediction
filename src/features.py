"""Turn the raw database tables into the model-ready feature table.

The same definitions are used by the notebook's EDA and the daily pipeline, so the live model sees exactly the
features it was researched with. Everything is point-in-time: a row dated D uses only data known at D's close.
"""
import numpy as np
import pandas as pd

from . import config

NUMERIC_FEATURES = ['log_ret', 'ret_5d', 'ret_21d', 'ret_lag1', 'ret_lag2',
                    'close_vs_ema12', 'close_vs_ema26', 'macd_norm', 'macd_hist_norm', 'rsi_14',
                    'vol_21d', 'bb_width', 'bb_position', 'hl_range', 'volume_z',
                    'DGS10', 'DGS10_chg', 'T10Y2Y', 'T10Y2Y_chg', 'VIXCLS', 'VIX_chg',
                    'FEDFUNDS', 'CPI_YoY', 'UNRATE', 'UNRATE_chg_3m']
CATEGORICAL_FEATURES = ['Ticker', 'day_of_week']


def load_prices(con):
    """Prices with OHLC scaled by the dividend/split adjustment factor (adj_close / close)."""
    p = pd.read_sql('SELECT * FROM prices', con, parse_dates=['date'])
    factor = p['adj_close'] / p['close']
    out = pd.DataFrame({'Date': p['date'].astype('datetime64[ns]'), 'Ticker': p['ticker'],
                        'Open': p['open'] * factor, 'High': p['high'] * factor, 'Low': p['low'] * factor,
                        'Close': p['adj_close'], 'Volume': p['volume']})
    out['Sector'] = out['Ticker'].map(config.TICKERS)
    return out.sort_values(['Ticker', 'Date']).reset_index(drop=True)


def load_macro(con):
    """Return (daily, monthly) macro frames, each indexed by the date the value became AVAILABLE."""
    m = pd.read_sql('SELECT date, series_id, value FROM macro', con, parse_dates=['date'])
    wide = m.pivot(index='date', columns='series_id', values='value').sort_index()
    wide.index = wide.index.astype('datetime64[ns]')

    daily = wide[config.DAILY_SERIES].dropna(how='all').ffill(limit=5)
    daily['DGS10_chg'] = daily['DGS10'].diff()
    daily['T10Y2Y_chg'] = daily['T10Y2Y'].diff()
    daily['VIX_chg'] = np.log(daily['VIXCLS']).diff()
    daily.index = daily.index + pd.offsets.BDay(config.FRED_DAILY_LAG_BDAYS)   # published next business day

    monthly = wide[config.MONTHLY_SERIES].dropna(how='all').copy()
    monthly['CPI_YoY'] = monthly['CPIAUCSL'].pct_change(12, fill_method=None) * 100
    monthly['UNRATE_chg_3m'] = monthly['UNRATE'].diff(3)
    monthly = monthly.drop(columns='CPIAUCSL')
    monthly.index = monthly.index + pd.DateOffset(**config.MONTHLY_RELEASE_LAG)
    return daily, monthly


def add_technical_features(group):
    """Technical indicators for a single ticker (rows sorted by date)."""
    g = group.copy()
    close = g['Close']
    g['log_ret'] = np.log(close).diff()
    g['ret_5d'] = np.log(close / close.shift(5))
    g['ret_21d'] = np.log(close / close.shift(21))
    g['ret_lag1'] = g['log_ret'].shift(1)
    g['ret_lag2'] = g['log_ret'].shift(2)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    g['ema_12'], g['ema_26'] = ema_12, ema_26
    g['close_vs_ema12'] = close / ema_12 - 1
    g['close_vs_ema26'] = close / ema_26 - 1
    g['macd_norm'] = macd / close
    g['macd_hist_norm'] = (macd - macd_signal) / close

    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    g['rsi_14'] = 100 - 100 / (1 + avg_gain / avg_loss)

    sma_20, std_20 = close.rolling(20).mean(), close.rolling(20).std()
    g['vol_21d'] = g['log_ret'].rolling(21).std() * np.sqrt(252)
    g['bb_width'] = 4 * std_20 / sma_20
    g['bb_position'] = (close - sma_20) / (2 * std_20)
    g['hl_range'] = (g['High'] - g['Low']) / close
    g['volume_z'] = (g['Volume'] - g['Volume'].rolling(21).mean()) / g['Volume'].rolling(21).std()

    g['target_ret'] = g['log_ret'].shift(-1)                 # NaN on the latest day: that's what we predict
    g['target_up'] = (g['target_ret'] > 0).astype(float).where(g['target_ret'].notna())
    return g


def build_dataset(con):
    """Full feature table. Rows on the latest date have target_up = NaN (the rows to predict)."""
    prices = load_prices(con)
    daily, monthly = load_macro(con)
    df = pd.merge_asof(prices.sort_values('Date'), daily.rename_axis('Date').reset_index(),
                       on='Date', direction='backward')
    df = pd.merge_asof(df, monthly.rename_axis('Date').reset_index(), on='Date', direction='backward')
    df = pd.concat([add_technical_features(g) for _, g in df.groupby('Ticker')], ignore_index=True)
    df['day_of_week'] = df['Date'].dt.day_name().str[:3]
    df['vix_regime'] = pd.cut(df['VIXCLS'], bins=[0, 15, 20, 30, np.inf],
                              labels=['Calm (<15)', 'Normal (15-20)', 'Elevated (20-30)', 'Stress (>30)'])
    df['curve_inverted'] = np.where(df['T10Y2Y'] < 0, 'Inverted', 'Normal')
    return df.dropna(subset=NUMERIC_FEATURES).sort_values(['Date', 'Ticker']).reset_index(drop=True)
