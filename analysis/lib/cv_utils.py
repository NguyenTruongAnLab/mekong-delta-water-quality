"""Cross-validation utilities for spatial/temporal grouped validation.

Provides leakage-safe grouped CV helpers: LOSO, LOYO, LOBO, and
spatial-block CV. All preprocessing (scaling, orthogonalization) must
happen inside folds when using these helpers.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from scipy.stats import spearmanr


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Spearman rho robust to edge cases."""
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan, np.nan
    res = spearmanr(y_true, y_pred)
    try:
        return float(res.statistic), float(res.pvalue)
    except AttributeError:
        return float(res[0]), float(res[1])


def grouped_cv_predict(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    preprocess_fn: Callable | None = None,
) -> np.ndarray:
    """Leave-one-group-out CV with optional in-fold preprocessing.

    Parameters
    ----------
    model : sklearn estimator (will be cloned per fold)
    X : feature matrix
    y : target vector
    groups : group labels (station, year, branch, etc.)
    preprocess_fn : optional callable(X_train, X_test) -> (X_train_proc, X_test_proc)
        Applied inside each fold to avoid leakage.

    Returns
    -------
    y_pred : array of out-of-fold predictions aligned with y
    """
    logo = LeaveOneGroupOut()
    y_pred = np.full_like(y, np.nan, dtype=float)

    for train_idx, test_idx in logo.split(X, y, groups):
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        if preprocess_fn is not None:
            X_train, X_test = preprocess_fn(X_train, X_test)

        m = clone(model)
        m.fit(X_train, y_train)
        y_pred[test_idx] = m.predict(X_test)

    return y_pred


def cv_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R², RMSE, and Spearman rho from CV predictions."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 3:
        return {"R2": np.nan, "RMSE": np.nan, "Spearman_rho": np.nan, "Spearman_p": np.nan, "N": 0}
    rho, pval = _safe_spearman(yt, yp)
    return {
        "R2": float(r2_score(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "Spearman_rho": rho,
        "Spearman_p": pval,
        "N": int(len(yt)),
    }


def naive_baselines(y_true: np.ndarray, groups_station: np.ndarray,
                    groups_branch: np.ndarray | None = None) -> dict:
    """Compute naive baseline predictions: global mean, station mean, branch mean."""
    results = {}
    # Global mean
    gm = np.nanmean(y_true)
    results["global_mean"] = cv_metrics(y_true, np.full_like(y_true, gm))

    # Station mean (LOSO: predicted = mean of all OTHER stations)
    logo = LeaveOneGroupOut()
    y_pred_station = np.full_like(y_true, np.nan, dtype=float)
    for train_idx, test_idx in logo.split(y_true, y_true, groups_station):
        y_pred_station[test_idx] = np.nanmean(y_true[train_idx])
    results["other_station_mean"] = cv_metrics(y_true, y_pred_station)

    # Branch mean (LOBO)
    if groups_branch is not None:
        y_pred_branch = np.full_like(y_true, np.nan, dtype=float)
        for train_idx, test_idx in logo.split(y_true, y_true, groups_branch):
            y_pred_branch[test_idx] = np.nanmean(y_true[train_idx])
        results["other_branch_mean"] = cv_metrics(y_true, y_pred_branch)

    return results


def evaluate_model_cv(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    station_groups: np.ndarray,
    branch_groups: np.ndarray | None = None,
    year_groups: np.ndarray | None = None,
    preprocess_fn: Callable | None = None,
    label: str = "",
) -> list[dict]:
    """Evaluate model under multiple CV schemes."""
    rows = []

    # LOSO
    pred_loso = grouped_cv_predict(model, X, y, station_groups, preprocess_fn)
    m = cv_metrics(y, pred_loso)
    m.update({"CV_scheme": "LOSO", "model_label": label})
    rows.append(m)

    # LOBO
    if branch_groups is not None:
        pred_lobo = grouped_cv_predict(model, X, y, branch_groups, preprocess_fn)
        m = cv_metrics(y, pred_lobo)
        m.update({"CV_scheme": "LOBO", "model_label": label})
        rows.append(m)

    # LOYO
    if year_groups is not None:
        pred_loyo = grouped_cv_predict(model, X, y, year_groups, preprocess_fn)
        m = cv_metrics(y, pred_loyo)
        m.update({"CV_scheme": "LOYO", "model_label": label})
        rows.append(m)

    return rows
