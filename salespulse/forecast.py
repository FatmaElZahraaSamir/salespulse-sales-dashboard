"""Monthly revenue forecasting with an honest backtest.

Three candidate models are compared on the most recent months (walk-forward
holdout); the one with the lowest error forecasts the future:

* Seasonal naive  - "same month last year" (the baseline every model must beat)
* Holt-Winters    - exponential smoothing with damped trend + yearly seasonality
* Trend regression - log-linear trend + month-of-year effects (ridge-regularised)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import last_complete_month

SEASON = 12


def monthly_series(df: pd.DataFrame, value: str = "sales") -> pd.Series:
    """Revenue per calendar month, complete months only, gaps filled with 0."""
    end = last_complete_month(df)
    months = df["order_date"].dt.to_period("M")
    series = df.groupby(months)[value].sum()
    series = series[series.index <= end]
    full = pd.period_range(series.index.min(), end, freq="M")
    series = series.reindex(full, fill_value=0.0)
    series.index = series.index.to_timestamp()
    series.index.freq = "MS"
    return series.astype(float)


# ------------------------------------------------------------------ models
def seasonal_naive(train: pd.Series, horizon: int) -> np.ndarray:
    if len(train) < SEASON:
        return np.repeat(train.iloc[-1], horizon)
    last_year = train.iloc[-SEASON:].to_numpy()
    return np.array([last_year[i % SEASON] for i in range(horizon)])


def holt_winters(train: pd.Series, horizon: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seasonal = SEASON if len(train) >= 2 * SEASON else None
        model = ExponentialSmoothing(
            train.clip(lower=1.0),
            trend="add",
            damped_trend=True,
            seasonal="mul" if seasonal else None,
            seasonal_periods=seasonal,
            initialization_method="estimated",
        ).fit(optimized=True)
        return np.clip(model.forecast(horizon).to_numpy(), 0, None)


def trend_regression(train: pd.Series, horizon: int, alpha: float = 1.0) -> np.ndarray:
    """revenue ~ linear time trend + month-of-year effects (ridge-regularised)."""
    from sklearn.linear_model import Ridge

    def features(index: pd.DatetimeIndex, start: pd.Timestamp) -> np.ndarray:
        t = ((index.year - start.year) * 12 + index.month - start.month).to_numpy()[:, None] / 12.0
        months = np.eye(12)[index.month - 1]
        return np.hstack([t, months])

    start = train.index[0]
    future = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    scale = max(float(train.mean()), 1.0)  # fit in "average month" units for a well-conditioned ridge
    model = Ridge(alpha=alpha).fit(features(train.index, start), train.to_numpy() / scale)
    return np.clip(model.predict(features(future, start)) * scale, 0, None)


MODELS = {
    "Seasonal naive (baseline)": seasonal_naive,
    "Holt-Winters": holt_winters,
    "Trend + seasonality": trend_regression,
}


# ---------------------------------------------------------------- evaluation
def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    mask = actual > 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else float("nan")


def backtest(series: pd.Series, horizon: int = 6, folds: int = 3) -> pd.DataFrame:
    """Rolling-origin backtest: forecast `horizon` months from `folds` different cut-off points."""
    rows = []
    for name, fn in MODELS.items():
        errors, abs_errors = [], []
        for fold in range(folds, 0, -1):
            cut = len(series) - horizon - (fold - 1)
            train, test = series.iloc[:cut], series.iloc[cut: cut + horizon]
            if len(train) < SEASON + 3 or test.empty:
                continue
            pred = fn(train, len(test))
            errors.append(mape(test.to_numpy(), pred))
            abs_errors.append(float(np.mean(np.abs(test.to_numpy() - pred))))
        if errors:
            rows.append({"model": name, "MAPE %": round(float(np.nanmean(errors)), 1),
                         "MAE": round(float(np.mean(abs_errors)), 0), "folds": len(errors)})
    result = pd.DataFrame(rows).sort_values("MAPE %").reset_index(drop=True)
    return result


@dataclass
class Forecast:
    history: pd.Series
    table: pd.DataFrame          # month, forecast, lower, upper
    model: str
    backtest: pd.DataFrame

    @property
    def improvement_vs_baseline(self) -> float | None:
        bt = self.backtest.set_index("model")["MAPE %"]
        base = bt.get("Seasonal naive (baseline)")
        best = bt.get(self.model)
        if base is None or best is None or base == 0:
            return None
        return round((base - best) / base * 100, 1)


def forecast(series: pd.Series, horizon: int = 6, interval: float = 0.8) -> Forecast:
    """Pick the best model on the backtest and forecast `horizon` months ahead.

    The prediction band comes from the chosen model's own backtest errors (the
    80th percentile of absolute percentage errors by default), so it widens
    when the series is hard to predict.
    """
    bt = backtest(series, horizon=horizon)
    best = bt["model"].iloc[0] if not bt.empty else "Seasonal naive (baseline)"
    point = MODELS[best](series, horizon)

    # empirical relative errors of the chosen model on the backtest folds
    rel_errors = []
    for fold in range(3, 0, -1):
        cut = len(series) - horizon - (fold - 1)
        train, test = series.iloc[:cut], series.iloc[cut: cut + horizon]
        if len(train) < SEASON + 3:
            continue
        pred = MODELS[best](train, len(test))
        ok = test.to_numpy() > 0
        rel_errors.extend(((test.to_numpy()[ok] - pred[ok]) / np.maximum(pred[ok], 1)).tolist())
    # band half-width = the `interval` quantile of absolute percentage errors seen in the backtest
    width = float(np.quantile(np.abs(rel_errors), interval)) if len(rel_errors) >= 4 else 0.2

    months = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    table = pd.DataFrame({
        "month": months,
        "forecast": point,
        "lower": np.clip(point * (1 - width), 0, None),
        "upper": point * (1 + width),
    })
    return Forecast(series, table, best, bt)
