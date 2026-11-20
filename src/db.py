"""SQLite storage: schema, connection and idempotent upserts."""
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    date        TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    open        REAL, high REAL, low REAL, close REAL,
    adj_close   REAL,
    volume      REAL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS macro (
    date        TEXT NOT NULL,
    series_id   TEXT NOT NULL,
    value       REAL,
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (date, series_id)
);
CREATE TABLE IF NOT EXISTS predictions (
    date          TEXT NOT NULL,      -- feature date (market close the prediction was made at)
    ticker        TEXT NOT NULL,
    proba_up      REAL NOT NULL,      -- P(next trading day's return > 0)
    pred_up       INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (date, ticker, model_version)
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL,       -- running / success / failed
    price_rows   INTEGER,
    macro_rows   INTEGER,
    message      TEXT
);
"""


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def connect(path=DB_PATH):
    """Open (and if needed create) the database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def upsert(con, table, frame):
    """Insert rows, replacing any existing row with the same primary key. Safe to re-run."""
    if frame.empty:
        return 0
    frame = frame.copy()
    for col in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[col]):
            frame[col] = frame[col].dt.strftime('%Y-%m-%d')
    frame = frame.astype(object).where(frame.notna(), None)
    cols = ', '.join(frame.columns)
    marks = ', '.join('?' * len(frame.columns))
    with con:
        con.executemany(f'INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})',
                        frame.itertuples(index=False, name=None))
    return len(frame)
