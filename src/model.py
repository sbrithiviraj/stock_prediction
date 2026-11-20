"""Train, version, load and apply the next-day direction model."""
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

log = logging.getLogger(__name__)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to percentiles learned on the training data (saved inside the model)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_, self.high_ = np.nanquantile(X, self.lower, axis=0), np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def make_pipeline():
    preprocess = ColumnTransformer([
        ('num', Pipeline([('winsor', Winsorizer()), ('scale', StandardScaler())]), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL_FEATURES),
    ])
    return Pipeline([('prep', preprocess),
                     ('clf', LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE))])


def train(dataset, trained_on=None, model_dir=config.MODEL_DIR):
    """Fit on every row whose next-day outcome is known; save a dated version and point latest.json at it."""
    labelled = dataset.dropna(subset=['target_up']).sort_values('Date')
    X, y = labelled[NUMERIC_FEATURES + CATEGORICAL_FEATURES], labelled['target_up'].astype(int)
    search = GridSearchCV(make_pipeline(), {'clf__C': config.C_GRID},
                          cv=TimeSeriesSplit(n_splits=config.CV_SPLITS), scoring='roc_auc', n_jobs=-1)
    search.fit(X, y)

    version = labelled['Date'].max().strftime('%Y%m%d')
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, model_dir / f'model_{version}.joblib')
    meta = {'version': version,
            'trained_on': pd.Timestamp(trained_on or pd.Timestamp.today()).strftime('%Y-%m-%d'),
            'train_start': labelled['Date'].min().strftime('%Y-%m-%d'),
            'train_end': labelled['Date'].max().strftime('%Y-%m-%d'),
            'rows': int(len(labelled)),
            'best_C': search.best_params_['clf__C'],
            'cv_roc_auc': round(float(search.best_score_), 4),
            'features': NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    (model_dir / 'latest.json').write_text(json.dumps(meta, indent=2))
    log.info('Trained model %s on %d rows (CV ROC-AUC %.4f, C=%s)', version, len(labelled),
             meta['cv_roc_auc'], meta['best_C'])
    return search.best_estimator_, meta


def load_latest(model_dir=config.MODEL_DIR):
    pointer = model_dir / 'latest.json'
    if not pointer.exists():
        return None, None
    meta = json.loads(pointer.read_text())
    return joblib.load(model_dir / f"model_{meta['version']}.joblib"), meta


def needs_retrain(meta, today):
    if meta is None:
        return True
    return (pd.Timestamp(today) - pd.Timestamp(meta['trained_on'])).days >= config.RETRAIN_EVERY_DAYS


def predict_latest(model, dataset, version):
    """Predict next-day direction for each ticker's most recent date."""
    latest = dataset[dataset['Date'] == dataset['Date'].max()]
    proba = model.predict_proba(latest[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    return pd.DataFrame({'date': latest['Date'].values, 'ticker': latest['Ticker'].values,
                         'proba_up': proba, 'pred_up': (proba > 0.5).astype(int), 'model_version': version})
