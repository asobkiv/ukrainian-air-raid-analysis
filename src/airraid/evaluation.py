"""Daily rolling-origin evaluation with validation labels frozen before the test."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .models import BASELINES, CANDIDATES, forecast_task
from .series import TZ, aggregate, coverage_at, day_hours, mark_censored, midnight
from .sources import as_of, snapshot_at


def design(ledger: pd.DataFrame, snapshots: list, config: dict) -> dict:
    latest = snapshots[-1]
    # Latest completed local calendar day; unresolved source intervals are excluded per label.
    test_end = pd.Timestamp(latest["committed_at"]).tz_convert(TZ).date() - timedelta(days=1)
    test_start = test_end - timedelta(days=config["test_days"] - 1)
    selection = midnight(test_start)
    validation_end = test_start - timedelta(days=3)  # publication/closure guard before model selection
    validation_start = validation_end - timedelta(days=config["validation_days"] - 1)
    selection_snapshot = snapshot_at(snapshots, selection, config["snapshot_max_age_hours"])
    assessment = coverage_at(as_of(ledger, selection), selection, config)
    regions = assessment.loc[assessment.eligible, "region"].tolist()
    if not regions:
        raise ValueError("No eligible regions")
    return {"test_start": str(test_start), "test_end": str(test_end), "selection_at": selection.isoformat(),
            "validation_start": str(validation_start), "validation_end": str(validation_end),
            "selection_snapshot": selection_snapshot["sha"], "regions": regions,
            "coverage": assessment.to_dict("records"), "config": config}


def make_tasks(ledger, snapshots, config, dates, regions):
    tasks, omissions = [], []
    for d in dates:
        day = pd.Timestamp(d).date()
        origin = midnight(day)
        snap = snapshot_at(snapshots, origin, config["snapshot_max_age_hours"])
        events = as_of(ledger, origin)
        if not (events.available_at <= origin).all():
            raise AssertionError("Future version in training")
        known_end = pd.Timestamp(snap["committed_at"]).tz_convert(TZ).date() - timedelta(days=1)
        window_start = day - timedelta(days=config["train_days"] + 10)
        daily = mark_censored(aggregate(events, window_start, known_end, regions), snap)
        coverage = coverage_at(events, origin, config).set_index("region")
        for region in regions:
            if region not in coverage.index or not coverage.loc[region, "eligible"]:
                omissions.append({"region": region, "date": str(day), "reason": "source_coverage_proxy_failed"})
                continue
            g = daily.loc[(daily.region == region) & daily.label_ready].sort_values("date")
            if len(g) < config["minimum_train_days"]:
                omissions.append({"region": region, "date": str(day), "reason": "insufficient_ready_training_days"})
                continue
            g = g.tail(config["train_days"])
            last = pd.Timestamp(g.date.iloc[-1]).date()
            if len(g) > 1 and not (pd.to_datetime(g.date).diff().dropna() == pd.Timedelta(days=1)).all():
                raise ValueError("Training series has internal gaps; cannot silently compress calendar")
            steps = (day - last).days
            if steps < 1 or steps > 14:
                omissions.append({"region": region, "date": str(day), "reason": "training_history_too_stale"})
                continue
            tasks.append({"region": region, "date": str(day), "origin": origin.isoformat(),
                          "snapshot_sha": snap["sha"], "snapshot_available_at": snap["available_at"],
                          "train_start": g.date.iloc[0], "train_end": str(last), "train_n": len(g),
                          "steps": steps, "day_hours": day_hours(day), "count": g["count"].tolist(),
                          "share": g.share.tolist(), "models": config["models"]})
    return tasks, omissions


def score(predictions: pd.DataFrame) -> pd.DataFrame:
    d = predictions.loc[predictions.label_ready & predictions.actual.notna()].copy()
    d["absolute_error"] = (d.prediction - d.actual).abs()
    d["squared_error"] = (d.prediction - d.actual) ** 2
    result = d.groupby(["region", "target", "model"], observed=True).agg(
        n=("actual", "size"), mae=("absolute_error", "mean"), mse=("squared_error", "mean"),
        clipping_rate=("was_clipped", "mean")).reset_index()
    result["rmse"] = np.sqrt(result.pop("mse"))
    return result


def attach_labels(predictions, daily):
    labels = daily.melt(id_vars=["region", "date", "label_ready", "label_reason"],
                        value_vars=["count", "hours"], var_name="target", value_name="actual")
    return predictions.merge(labels, on=["region", "date", "target"], how="left", validate="many_to_one")


def choose_models(validation_scores):
    selections = []
    for (region, target), g in validation_scores.groupby(["region", "target"]):
        for kind, names in [("selected", CANDIDATES), ("baseline", BASELINES)]:
            subset = g.loc[g.model.isin(names)].sort_values(["mae", "rmse", "model"])
            if subset.empty:
                raise ValueError("Missing validation candidate")
            best = subset.iloc[0]
            selections.append({"region": region, "target": target, "kind": kind,
                               "model": best.model, "validation_mae": best.mae, "validation_n": int(best.n)})
    return pd.DataFrame(selections)


def run_predictions(tasks, workers, output: Path):
    rows = []
    if workers == 1:
        iterator = map(forecast_task, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterator = pool.map(forecast_task, tasks, chunksize=2)
    try:
        for i, result in enumerate(iterator, 1):
            rows.extend(result)
            if i % 25 == 0 or i == len(tasks):
                print(f"Forecast tasks: {i}/{len(tasks)}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    return df


def evaluate(ledger, snapshots, protocol, output: Path, workers):
    output.mkdir(parents=True, exist_ok=True)
    config, regions = protocol["config"], protocol["regions"]
    validation_dates = pd.date_range(protocol["validation_start"], protocol["validation_end"])
    test_dates = pd.date_range(protocol["test_start"], protocol["test_end"])
    tasks, omitted = make_tasks(ledger, snapshots, config, list(validation_dates) + list(test_dates), regions)
    print(f"Ready: {len(tasks)} regional origins, {len(omitted)} omitted", flush=True)
    # No model-selection work depends on test labels. All candidate forecasts are saved for comparison.
    predictions = run_predictions(tasks, workers, output / "predictions_unlabelled.csv")
    selection_at = pd.Timestamp(protocol["selection_at"])
    val_snap = snapshot_at(snapshots, selection_at, config["snapshot_max_age_hours"])
    val_daily = mark_censored(aggregate(as_of(ledger, selection_at), validation_dates[0].date(),
                                       validation_dates[-1].date(), regions), val_snap)
    val_pred = attach_labels(predictions.loc[predictions.date.isin(validation_dates.strftime("%Y-%m-%d"))], val_daily)
    val_scores = score(val_pred)
    selection = choose_models(val_scores)
    # Selection must be complete before exposing final evaluation labels to scoring.
    selection.to_csv(output / "model_selection.csv", index=False)
    latest = snapshots[-1]
    truth = mark_censored(aggregate(as_of(ledger, pd.Timestamp(latest["available_at"])),
                                   test_dates[0].date(), test_dates[-1].date(), regions), latest)
    test_pred = attach_labels(predictions.loc[predictions.date.isin(test_dates.strftime("%Y-%m-%d"))], truth)
    val_pred.to_csv(output / "validation_predictions.csv", index=False)
    test_pred.to_csv(output / "test_predictions.csv", index=False)
    val_scores.to_csv(output / "validation_metrics.csv", index=False)
    score(test_pred).to_csv(output / "test_metrics_all_candidates.csv", index=False)
    picked = test_pred.merge(selection[["region", "target", "model", "kind"]],
                             on=["region", "target", "model"], how="inner", validate="many_to_many")
    picked.to_csv(output / "test_predictions_selected.csv", index=False)
    selected_scores = score(picked.rename(columns={"model": "source_model", "kind": "model"}))
    selected_scores.to_csv(output / "test_metrics_selected.csv", index=False)
    macro = selected_scores.groupby(["target", "model"]).agg(
        macro_mae=("mae", "mean"), macro_rmse=("rmse", "mean"), regions=("region", "size"),
        forecast_days=("n", "sum")).reset_index()
    macro.to_csv(output / "test_macro_metrics.csv", index=False)
    pd.DataFrame(omitted, columns=["region", "date", "reason"]).to_csv(output / "forecast_omissions.csv", index=False)
    truth.to_csv(output / "test_label_status.csv", index=False)
    audit = {"forecast_rows": len(predictions), "all_finite": bool(np.isfinite(predictions.prediction).all()),
             "nonnegative": bool((predictions.prediction >= 0).all()),
             "hours_bounded": bool((predictions.loc[predictions.target == "hours", "model_output"] <= 1).all()),
             "all_snapshots_before_origin": bool((pd.to_datetime(predictions.snapshot_available_at, utc=True)
                                                   <= pd.to_datetime(predictions.origin, utc=True)).all()),
             "max_steps": int(predictions.steps.max()), "min_steps": int(predictions.steps.min()),
             "fallback_forecasts": int(predictions.status.fillna("").str.startswith("fallback").sum()),
             "clipped_forecasts": int(predictions.was_clipped.sum()),
             "test_labels_excluded": int((~truth.label_ready).sum()),
             "validation_labels_excluded": int((~val_daily.label_ready).sum()),
             "selection_label_snapshot": val_snap["sha"], "selection_label_available_at": val_snap["available_at"]}
    if not all(audit[k] for k in ["all_finite", "nonnegative", "hours_bounded", "all_snapshots_before_origin"]):
        raise AssertionError(audit)
    (output / "forecast_audit.json").write_text(json.dumps(audit, indent=2))
    return predictions, val_scores, macro
