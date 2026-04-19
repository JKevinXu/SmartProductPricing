"""Metrics used by SmartProductPricing models."""

from __future__ import annotations

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return symmetric mean absolute percentage error as a percentage."""
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    errors = np.divide(
        np.abs(predicted - actual),
        denominator,
        out=np.zeros_like(actual, dtype=np.float64),
        where=denominator != 0,
    )
    return float(np.mean(errors) * 100.0)
