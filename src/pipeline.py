"""Daily pipeline: ingest -> validate -> features -> (re)train -> predict -> score past predictions.

Usage:
    python -m src.pipeline                  # normal daily run
    python -m src.pipeline --force-retrain  # retrain the model even if it is recent
"""
import argparse
import logging
import sys

import pandas as pd

from . import config
from .db import connect, upsert, utc_now
from .features import build_dataset
from .ingest import fetch_fred, fetch_prices_yahoo, ingest_macro, ingest_prices
from .model import load_latest, needs_retrain, predict_latest, train
from .validate import validate

log = logging.getLogger('pipeline')


def setup_logging():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(config.LOG_DIR / 'pipeline.log')])


def live_scorecard(con):
    """Directional accuracy of past live predictions whose outcome is now known."""
    preds = pd.read_sql('SELECT date, ticker, pred_up FROM predictions', con, parse_dates=['date'])
    feats = pd.read_sql('SELECT Date AS date, Ticker AS ticker, target_up FROM features '
                        'WHERE target_up IS NOT NULL', con, parse_dates=['date'])
    scored = preds.merge(feats, on=['date', 'ticker'])
    if scored.empty:
        return None
    return {'scored_predictions': len(scored),
            'directional_accuracy': round(float((scored['pred_up'] == scored['target_up']).mean()), 4)}


def run(today=None, force_retrain=False, fetch_prices=fetch_prices_yahoo, fetch_macro=fetch_fred, db_path=None):
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    con = connect(db_path) if db_path else connect()
    run_id = con.execute('INSERT INTO pipeline_runs (started_at, status) VALUES (?, ?)',
                         (utc_now(), 'running')).lastrowid
    con.commit()
    try:
        log.info('Step 1/5 ingest prices');  n_prices = ingest_prices(con, fetch=fetch_prices)
        log.info('Step 2/5 ingest macro');   n_macro = ingest_macro(con, fetch=fetch_macro)
        log.info('Step 3/5 validate');       validate(con, today)

        log.info('Step 4/5 build features')
        dataset = build_dataset(con)
        dataset.drop(columns=['vix_regime']).to_sql('features', con, if_exists='replace', index=False)
        log.info('Feature table: %d rows, latest date %s', len(dataset), dataset['Date'].max().date())

        log.info('Step 5/5 model')
        model, meta = load_latest()
        if force_retrain or needs_retrain(meta, today):
            model, meta = train(dataset, trained_on=today)
        preds = predict_latest(model, dataset, meta['version']).assign(created_at=utc_now())
        upsert(con, 'predictions', preds)
        for row in preds.itertuples():
            log.info('Prediction for session after %s  %-5s P(up)=%.3f -> %s',
                     pd.Timestamp(row.date).date(), row.ticker, row.proba_up, 'UP' if row.pred_up else 'DOWN/FLAT')

        score = live_scorecard(con)
        if score:
            log.info('Live track record: %s', score)
        with con:
            con.execute('UPDATE pipeline_runs SET finished_at=?, status=?, price_rows=?, macro_rows=? WHERE run_id=?',
                        (utc_now(), 'success', n_prices, n_macro, run_id))
        return preds
    except Exception as exc:
        log.exception('Pipeline failed')
        with con:
            con.execute('UPDATE pipeline_runs SET finished_at=?, status=?, message=? WHERE run_id=?',
                        (utc_now(), 'failed', str(exc)[:1000], run_id))
        raise
    finally:
        con.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Daily market-data pipeline')
    parser.add_argument('--force-retrain', action='store_true')
    args = parser.parse_args()
    setup_logging()
    run(force_retrain=args.force_retrain)
