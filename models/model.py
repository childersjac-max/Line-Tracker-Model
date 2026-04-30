# models/model.py

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from xgboost import XGBClassifier
from configs.config import (
    XGB_PARAMS, XGB_OVERRIDES, MODELS_DIR,
    TIMESERIES_CV_SPLITS, CALIBRATION_TAIL_FRACTION,
)
from features.movement import FEATURE_COLS

logger = logging.getLogger(__name__)


def _detect_n_features(clf, cal, fallback: int) -> int:
    """Return the number of features the model was trained on.

    Checks `n_features_in_` on the raw classifier first, then on the
    calibrated wrapper, then falls back to the XGBoost booster's num_feature.
    Used only for legacy model files that pre-date feature_cols persistence.
    """
    for obj in (clf, cal):
        if obj is None:
            continue
        # scikit-learn sets n_features_in_ on fit
        if hasattr(obj, "n_features_in_"):
            return int(obj.n_features_in_)
        # CalibratedClassifierCV wraps the estimator
        if hasattr(obj, "estimator") and hasattr(obj.estimator, "n_features_in_"):
            return int(obj.estimator.n_features_in_)
        # Iterate calibrated classifiers list (sklearn ≥ 1.2)
        if hasattr(obj, "calibrated_classifiers_"):
            for cc in obj.calibrated_classifiers_:
                if hasattr(cc, "estimator") and hasattr(cc.estimator, "n_features_in_"):
                    return int(cc.estimator.n_features_in_)
    # Last resort: ask the XGBoost booster directly
    try:
        return int(clf.get_booster().num_features())
    except Exception:
        pass
    return fallback


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
        self.trained_on  = None      # "real" or "synthetic" — set by train.py
        self.trained_at  = None
        self.feature_cols = list(FEATURE_COLS)

    # ────────────────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────────────────
    def train(self, df, n_splits=None, time_col="commence_time", trained_on="real"):
        """
        Train with proper TIME-SERIES cross-validation and PREFIT calibration
        on a held-out chronological tail. (Tier 1 fixes #2 and #3.)

        df must contain `time_col` so rows can be sorted chronologically.
        """
        if time_col not in df.columns:
            raise ValueError(
                f"train() requires a '{time_col}' column for time-series CV. "
                f"Make sure label_histories preserves commence_time and that "
                f"build_feature_dataframe surfaces it."
            )

        # Sort chronologically — never shuffle line-movement data.
        df = df.copy()
        df["_t"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.sort_values("_t", kind="mergesort").reset_index(drop=True)

        # Address class imbalance via scale_pos_weight (Tier 3 freebie)
        y_full = df["outcome"].astype(int).values
        n_pos = int(y_full.sum())
        n_neg = int(len(y_full) - n_pos)
        if n_pos > 0 and n_neg > 0:
            self.params = {**self.params, "scale_pos_weight": n_neg / n_pos}

        # ── Honest out-of-time CV metrics ───────────────────────────────────
        n_splits = n_splits or TIMESERIES_CV_SPLITS
        n_splits = max(2, min(n_splits, max(2, len(y_full) // 30)))

        X_full = self._prep(df, require_label=False)[0]
        tss = TimeSeriesSplit(n_splits=n_splits)

        aucs, lls, briers = [], [], []
        for fold_idx, (tr, te) in enumerate(tss.split(X_full)):
            y_tr, y_te = y_full[tr], y_full[te]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            clf = XGBClassifier(**self.params)
            clf.fit(X_full[tr], y_tr)
            p = clf.predict_proba(X_full[te])[:, 1]
            try:
                aucs.append(roc_auc_score(y_te, p))
            except ValueError:
                pass
            lls.append(log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6)))
            briers.append(brier_score_loss(y_te, p))

        self._cv_metrics = {
            "auc_mean":      float(np.mean(aucs))   if aucs   else float("nan"),
            "auc_std":       float(np.std(aucs))    if aucs   else float("nan"),
            "logloss_mean":  float(np.mean(lls))    if lls    else float("nan"),
            "brier_mean":    float(np.mean(briers)) if briers else float("nan"),
            "n_splits_used": len(aucs),
            "n_samples":     int(len(y_full)),
            "n_wins":        int(n_pos),
            "n_losses":      int(n_neg),
            "cv_strategy":   "TimeSeriesSplit",
        }
        logger.info(
            f"  [{self.sport_key}/{self.market}] "
            f"OOT AUC: {self._cv_metrics['auc_mean']:.4f} "
            f"(±{self._cv_metrics['auc_std']:.4f}) "
            f"| Brier: {self._cv_metrics['brier_mean']:.4f} "
            f"| n={self._cv_metrics['n_samples']}"
        )

        # ── Production fit + PREFIT calibration on chronological tail ──────
        cal_frac = CALIBRATION_TAIL_FRACTION
        n_cal = max(20, int(len(df) * cal_frac))

        if len(df) < 60 or n_cal >= len(df) - 20:
            # Not enough data to safely hold out — fall back to a quick sigmoid
            # calibration via 3-fold (still better than nothing, but flag it).
            logger.warning(
                f"  [{self.sport_key}/{self.market}] "
                f"Only {len(df)} samples — using full-fit + sigmoid CV calibration. "
                f"Probabilities will be less reliable."
            )
            X = X_full
            self._clf = XGBClassifier(**self.params)
            self._clf.fit(X, y_full)
            self._cal = CalibratedClassifierCV(
                XGBClassifier(**self.params), method="sigmoid", cv=3,
            )
            self._cal.fit(X, y_full)
        else:
            df_fit = df.iloc[:-n_cal].copy()
            df_cal = df.iloc[-n_cal:].copy()
            X_fit, y_fit = self._prep(df_fit)
            X_cal, y_cal = self._prep(df_cal)

            self._clf = XGBClassifier(**self.params)
            self._clf.fit(X_fit, y_fit)

            # PREFIT calibration — calibrator never sees the data the model
            # was fit on. Isotonic when we have enough cal samples, else sigmoid.
            method = "isotonic" if len(y_cal) >= 200 else "sigmoid"
            self._cal = CalibratedClassifierCV(self._clf, method=method, cv="prefit")
            self._cal.fit(X_cal, y_cal)
            self._cv_metrics["calibration_method"]  = method
            self._cv_metrics["n_calibration"]       = int(len(y_cal))
            self._cv_metrics["n_fit"]               = int(len(y_fit))

        self.trained_on = trained_on
        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.is_trained = True
        return self._cv_metrics

    # ────────────────────────────────────────────────────────────────────────
    # Inference
    # ────────────────────────────────────────────────────────────────────────
    def predict_proba(self, df):
        self._require_trained()
        X, _ = self._prep(df, require_label=False)
        return self._cal.predict_proba(X)[:, 1]

    # ────────────────────────────────────────────────────────────────────────
    # Persistence
    # ────────────────────────────────────────────────────────────────────────
    def save(self):
        self._require_trained()
        Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clf": self._clf,
                "cal": self._cal,
                "feature_cols": self.feature_cols,
            },
            self._path(),
        )
        with open(self._path().with_suffix(".json"), "w") as f:
            json.dump({
                "sport_key":    self.sport_key,
                "market":       self.market,
                "trained_on":   self.trained_on,
                "trained_at":   self.trained_at,
                "feature_cols": self.feature_cols,
                "cv_metrics":   self._cv_metrics,
            }, f, indent=2)

    def load(self):
        bundle = joblib.load(self._path())
        self._clf = bundle["clf"]
        self._cal = bundle["cal"]

        saved_cols = bundle.get("feature_cols")
        if saved_cols is not None:
            # Modern model file — trust the saved list exactly.
            self.feature_cols = saved_cols
        else:
            # Older model file saved before feature_cols was persisted.
            # Detect the training width from the booster so we slice the
            # current FEATURE_COLS to exactly the right length. This is
            # safe because new features are always appended to the END of
            # FEATURE_COLS, so the first N are the same as at train time.
            n = _detect_n_features(self._clf, self._cal, fallback=len(FEATURE_COLS))
            self.feature_cols = list(FEATURE_COLS)[:n]
            logger.debug(
                f"[{self.sport_key}/{self.market}] legacy model: "
                f"detected {n} training features → using first {n} of FEATURE_COLS"
            )

        # Pull metadata sidecar if present so callers can read trained_on
        sidecar = self._path().with_suffix(".json")
        if sidecar.exists():
            try:
                with open(sidecar) as f:
                    meta = json.load(f)
                self.trained_on  = meta.get("trained_on")
                self.trained_at  = meta.get("trained_at")
                self._cv_metrics = meta.get("cv_metrics", {})
            except Exception:
                pass
        self.is_trained = True
        return self

    def _path(self):
        return Path(MODELS_DIR) / f"{self.sport_key}__{self.market}.joblib"

    # ────────────────────────────────────────────────────────────────────────
    # Internals
    # ────────────────────────────────────────────────────────────────────────
    def _prep(self, df, require_label=True):
        cols = self.feature_cols
        for c in cols:
            if c not in df.columns:
                df = df.copy()
                df[c] = 0.0
        X = df[cols].fillna(0).values.astype(np.float32)
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
