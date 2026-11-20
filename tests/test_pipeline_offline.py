"""Offline end-to-end test of the daily pipeline (no internet needed).

Copies src/ into a temporary folder, replaces Yahoo Finance and FRED with simulated data, and checks:
  1. first run backfills history and predicts      4. a dividend re-adjustment triggers a full reload
  2. the next day downloads only new rows           5. the model retrains after 7 days
  3. running twice on the same day changes nothing   6. stale data stops the run before predicting
Your real db/, models/ and logs/ are never touched.

Run from the repo root:   python tests/test_pipeline_offline.py
"""
import logging, sys, shutil, sqlite3, tempfile
from pathlib import Path
import numpy as np, pandas as pd

repo_root = Path(__file__).resolve().parents[1]
work = Path(tempfile.mkdtemp(prefix='pipeline_test_'))
shutil.copytree(repo_root / 'src', work / 'src')
sys.path.insert(0, str(work))
from src import pipeline, config
from src.validate import DataQualityError
logging.basicConfig(level=logging.CRITICAL)  # check 6 fails on purpose; hide its log

rng = np.random.default_rng(0)
dates = pd.bdate_range('2010-01-04', '2025-12-31')
PRICES = {}
for t, p0 in {'SPY': 110, 'AAPL': 7, 'MSFT': 30, 'JPM': 40, 'XOM': 70, 'JNJ': 60}.items():
    r = rng.standard_t(4, len(dates)) * 0.01 + 0.0003
    c = p0 * np.exp(np.cumsum(r))
    PRICES[t] = pd.DataFrame({'date': dates, 'ticker': t, 'open': c * 1.001, 'high': c * 1.01, 'low': c * 0.99,
                              'close': c, 'adj_close': c * 0.9, 'volume': rng.integers(1e6, 1e8, len(c)).astype(float)})
d = pd.bdate_range('2008-01-01', '2025-12-31'); m = pd.date_range('2008-01-01', '2025-12-01', freq='MS')
MACRO = {'DGS10': (d, 3 + np.cumsum(rng.normal(0, .02, len(d)))), 'DGS2': (d, 1 + np.cumsum(rng.normal(0, .02, len(d)))),
         'VIXCLS': (d, 18 * np.exp(np.cumsum(rng.normal(0, .01, len(d))))), 'FEDFUNDS': (m, np.linspace(.1, 5, len(m))),
         'CPIAUCSL': (m, 210 * np.exp(np.arange(len(m)) * .002)), 'UNRATE': (m, 5 + rng.normal(0, .3, len(m)))}
MACRO['T10Y2Y'] = (d, MACRO['DGS10'][1] - MACRO['DGS2'][1])
TODAY = {'v': None}
calls = []

def fake_prices(ticker, start):
    calls.append((ticker, start))
    p = PRICES[ticker]
    return p[(p.date >= start) & (p.date <= TODAY['v'])].copy()

def fake_fred(sid, start):
    idx, vals = MACRO[sid]
    f = pd.DataFrame({'date': idx, 'series_id': sid, 'value': vals})
    return f[(f.date >= start) & (f.date <= TODAY['v'] - pd.Timedelta(days=1))]

def run(day, **kw):
    TODAY['v'] = pd.Timestamp(day)
    return pipeline.run(today=day, fetch_prices=fake_prices, fetch_macro=fake_fred, **kw)

con = lambda: sqlite3.connect(config.DB_PATH)

# 1) first run = backfill + train
p = run('2025-06-02'); print('run1 preds:', len(p), p.date.iloc[0])
assert len(p) == 6 and str(p.date.iloc[0])[:10] == '2025-06-02'
n1 = con().execute('select count(*) from prices').fetchone()[0]

# 2) next day = incremental, no retrain
calls.clear(); p = run('2025-06-03')
assert all(s != config.HISTORY_START for _, s in calls), calls
n2 = con().execute('select count(*) from prices').fetchone()[0]; assert n2 == n1 + 6, (n1, n2)
print('run2 incremental ok, rows', n1, '->', n2)

# 3) same day twice = idempotent
run('2025-06-03'); assert con().execute('select count(*) from prices').fetchone()[0] == n2
print('run3 idempotent ok')

# 4) dividend: Yahoo rewrites adj_close history for AAPL -> full reload of AAPL only
PRICES['AAPL']['adj_close'] *= 0.99
calls.clear(); run('2025-06-04')
reloads = [t for t, s in calls if s == config.HISTORY_START]; assert reloads == ['AAPL'], calls
v = con().execute("select adj_close/close from prices where ticker='AAPL' and date='2015-01-02'").fetchone()[0]
assert abs(v - 0.891) < 1e-9, v
print('run4 corporate-action reload ok')

# 5) a week later -> retrain; scorecard has data
run('2025-06-11')
meta = (config.MODEL_DIR / 'latest.json').read_text(); assert '20250610' in meta, meta
print('run5 retrain ok; models:', sorted(x.name for x in config.MODEL_DIR.iterdir()))
print('scorecard:', pipeline.live_scorecard(con()))

# 6) stale data must fail and be logged
try:
    TODAY['v'] = pd.Timestamp('2025-06-11')
    pipeline.run(today='2025-06-30', fetch_prices=fake_prices, fetch_macro=fake_fred); raise SystemExit('should fail')
except DataQualityError as e:
    print('run6 stale detected:', str(e)[:90])
shutil.rmtree(work, ignore_errors=True)
print('\nALL 6 CHECKS PASSED')
