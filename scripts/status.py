"""One-screen health report of the pipeline database.

Run from the repo root:   python scripts/status.py
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

if not config.DB_PATH.exists():
    sys.exit(f'No database at {config.DB_PATH}. Run: python -m src.pipeline')

con = sqlite3.connect(config.DB_PATH)
q = lambda sql: pd.read_sql(sql, con)  # noqa: E731
pd.set_option('display.width', 120)

print('== Last 5 pipeline runs ==')
print(q('SELECT run_id, started_at, status, price_rows, macro_rows, substr(message, 1, 60) AS message '
        'FROM pipeline_runs ORDER BY run_id DESC LIMIT 5').to_string(index=False))

print('\n== Price data ==')
print(q('SELECT ticker, COUNT(*) AS rows, MIN(date) AS first, MAX(date) AS latest FROM prices GROUP BY ticker')
      .to_string(index=False))

print('\n== Macro data ==')
print(q('SELECT series_id, COUNT(value) AS rows, MAX(CASE WHEN value IS NOT NULL THEN date END) AS latest '
        'FROM macro GROUP BY series_id').to_string(index=False))

pointer = config.MODEL_DIR / 'latest.json'
if pointer.exists():
    meta = json.loads(pointer.read_text())
    print(f"\n== Model {meta['version']} ==  trained {meta['trained_on']} on {meta['rows']:,} rows "
          f"({meta['train_start']} to {meta['train_end']}), CV ROC-AUC {meta['cv_roc_auc']}, C={meta['best_C']}")

print('\n== Latest predictions (P(up) for the next trading session) ==')
print(q('SELECT date, ticker, ROUND(proba_up, 3) AS proba_up, '
        "CASE pred_up WHEN 1 THEN 'UP' ELSE 'DOWN/FLAT' END AS signal, model_version "
        'FROM predictions WHERE date = (SELECT MAX(date) FROM predictions) ORDER BY ticker').to_string(index=False))

try:
    score = q('SELECT COUNT(*) AS scored, ROUND(AVG(p.pred_up = f.target_up), 4) AS directional_accuracy '
              'FROM predictions p JOIN features f ON date(f.Date) = p.date AND f.Ticker = p.ticker '
              'WHERE f.target_up IS NOT NULL')
    print(f"\n== Live track record ==  {int(score.scored[0])} scored predictions, "
          f"directional accuracy {score.directional_accuracy[0]}")
except Exception:  # features table not built yet
    pass
