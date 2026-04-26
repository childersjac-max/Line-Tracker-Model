# models/model.py

import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from configs.config import XGB_PARAMS, XGB_OVERRIDES, MODELS_DIR
from features.movement import FEATURE_COLS

logger = logging.getLogger(__name__)


class LineMovementModel:
    def __init__(self, sport_key, market):
        self.sport_key = sport_key
        self.market    = market
        params = {**XGB_PARAMS}
        params.update(XGB_OVERRIDES.get(sport_key, {}))
        self.params      = params
        self._clf        = None
        self._cal        = None
        self._cv_metrics = {}
        self.is_trained  = False

    def train(self, df, n_splits=5):
        X, y = self._prep(df)
        logger.info(f"  [{self.sport_key}/{self.market}] Training: {len(y)} samples")
        skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf_cv = XGBClassifier(**self.params)
        auc    = cross_val_score(clf_cv, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        ll     = cross_val_score(clf_cv, X, y, cv=skf, scoring="neg_log_loss", n_jobs=-1)
        self._cv_metrics = {
            "auc_mean": float(auc.mean()), "auc_std": float(auc.std()),
            "logloss_mean": float(-ll.mean()), "n_samples": int(len(y)), "n_wins": int(y.sum()),
        }
        logger.info(f"      AUC: {self._cv_metrics['auc_mean']:.4f}")
        self._clf = XGBClassifier(**self.params)
        self._clf.fit(X, y)
        method = "isotonic" if len(y) >= 1000 else "sigmoid"
        self._cal = CalibratedClassifierCV(XGBClassifier(**self.params), method=method, cv=min(n_splits, 5))
        self._cal.fit(X, y)
        self.is_trained = True
        return self._cv_metrics

    def predict_proba(self, df):
        self._require_trained()
        X, _ = self._prep(df, require_label=False)
        return self._cal.predict_proba(X)[:, 1]

    def save(self):
        self._require_trained()
        Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "cal": self._cal}, self._path())
        with open(self._path().with_suffix(".json"), "w") as f:
            json.dump({"sport_key": self.sport_key, "market": self.market,
                       "cv_metrics": self._cv_metrics}, f, indent=2)

    def load(self):
        bundle = joblib.load(self._path())
        self._clf = bundle["clf"]
        self._cal = bundle["cal"]
        self.is_trained = True
        return self

    def _path(self):
        return Path(MODELS_DIR) / f"{self.sport_key}__{self.market}.joblib"

    def _prep(self, df, require_label=True):
        for c in FEATURE_COLS:
            if c not in df.columns:
                df = df.copy()
                df[c] = 0.0
        X = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
        y = df["outcome"].values.astype(int) if require_label and "outcome" in df.columns else None
        return X, y

    def _require_trained(self):
        if not self.is_trained:
            raise RuntimeError(f"Model {self.sport_key}/{self.market} not trained.")


def load_model(sport_key, market):
    try:
        return LineMovementModel(sport_key, market).load()
    except FileNotFoundError:
        return None


def load_all_models(sport_key, markets):
    return {m: model for m in markets if (model := load_model(sport_key, m)) is not None}
