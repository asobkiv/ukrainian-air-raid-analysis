"""Small, predeclared model set; all output constraints are explicit and audited."""
from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

BASELINES = ["last", "weekly_naive", "mean_7"]
CANDIDATES = ["arima", "sarima_7", "ses", "ets_7"]


def bounded(raw: float, target: str) -> float:
    if not np.isfinite(raw):
        raise ValueError("Non-finite forecast")
    return float(max(0, raw)) if target == "count" else float(np.clip(raw, 0, 1))


def forecast_one(y, steps: int, model: str) -> tuple[float, str, str]:
    values = np.asarray(y, dtype=float)
    if len(values) < 7 or not np.isfinite(values).all() or steps < 1:
        raise ValueError("Invalid training series or horizon")
    if model == "last":
        return float(values[-1]), "", ""
    if model == "mean_7":
        return float(values[-7:].mean()), "", ""
    if model == "weekly_naive":
        # Recursive weekly forecast when source latency spans multiple days.
        return float(values[-7 + ((steps - 1) % 7)]), "", ""
    if np.ptp(values) < 1e-12:
        return float(values[-1]), "constant_series", ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if model in {"arima", "sarima_7"}:
                seasonal = (1, 0, 0, 7) if model == "sarima_7" else (0, 0, 0, 0)
                fitted = ARIMA(values, order=(1, 0, 1), seasonal_order=seasonal, trend="c").fit(
                    method_kwargs={"maxiter": 80})
                if not fitted.mle_retvals.get("converged", True):
                    raise RuntimeError("optimizer_not_converged")
            elif model == "ses":
                fitted = SimpleExpSmoothing(values, initialization_method="estimated").fit(optimized=True)
            elif model == "ets_7":
                fitted = ExponentialSmoothing(values, trend=None, seasonal="add", seasonal_periods=7,
                                              initialization_method="estimated").fit(optimized=True)
            else:
                raise ValueError(f"Unknown model {model}")
            if model in {"ses", "ets_7"}:
                ret = getattr(fitted, "mle_retvals", None)
                if ret is not None and not ret.get("success", True):
                    raise RuntimeError("optimizer_not_converged")
            prediction = float(np.asarray(fitted.forecast(steps))[-1])
            if not np.isfinite(prediction):
                raise RuntimeError("non_finite_model_output")
            status = ""
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            prediction = float(values[-7:].mean())
            status = f"fallback_mean_7:{type(exc).__name__}:{str(exc)[:120]}"
    notes = "; ".join(sorted(set(str(w.message)[:140] for w in caught)))
    return prediction, status, notes


def forecast_task(task: dict) -> list[dict]:
    result = []
    for target in ["count", "hours"]:
        y = task["count"] if target == "count" else task["share"]
        for model in task["models"]:
            raw, status, notes = forecast_one(y, task["steps"], model)
            limited = bounded(raw, target)
            scale = task["day_hours"] if target == "hours" else 1.0
            record = {k: task[k] for k in ["region", "date", "origin", "snapshot_sha", "snapshot_available_at",
                                           "train_start", "train_end", "train_n", "steps", "day_hours"]}
            record.update({"target": target, "model": model, "raw_model_output": raw,
                           "model_output": limited, "raw_prediction": raw * scale,
                           "prediction": limited * scale, "was_clipped": abs(raw - limited) > 1e-12,
                           "status": status, "warnings": notes})
            result.append(record)
    return result
