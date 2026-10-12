"""Explicit fixed-baseline comparisons and calendar semantics from frozen forecasts.

These are reporting diagnostics requested after the test was opened. They do not
change the validation selection, forecasts, test period or original criterion.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .models import BASELINES
from .series import midnight


def summarize_baselines(all_metrics, selected_metrics):
    selected = selected_metrics.loc[selected_metrics.model == "selected", ["region", "target", "n", "mae", "rmse"]]
    baselines = all_metrics.loc[all_metrics.model.isin(BASELINES)].copy()
    historical = selected_metrics.loc[selected_metrics.model == "baseline"].copy()
    historical["model"] = "validation_selected_baseline"
    baselines = pd.concat([baselines, historical], ignore_index=True)
    paired = baselines.merge(selected, on=["region", "target"], suffixes=("_baseline", "_selected"),
                             validate="many_to_one")
    if len(paired) != len(selected) * (len(BASELINES) + 1) or not (paired.n_baseline == paired.n_selected).all():
        raise ValueError("Baseline comparisons must cover the same regional cases")
    paired["relative_mae_improvement"] = np.where(paired.mae_baseline > 0,
                                                   1 - paired.mae_selected / paired.mae_baseline, np.nan)
    macro = paired.groupby(["target", "model"]).agg(
        baseline_mae=("mae_baseline", "mean"), selected_mae=("mae_selected", "mean"),
        baseline_rmse=("rmse_baseline", "mean"), selected_rmse=("rmse_selected", "mean"),
        regions=("region", "size"), forecast_days=("n_selected", "sum")).reset_index()
    # Ratio of macro MAEs, rather than an average of unstable regional ratios.
    macro["relative_mae_improvement"] = np.where(macro.baseline_mae > 0,
                                                  1 - macro.selected_mae / macro.baseline_mae, np.nan)
    macro["meets_5_percent"] = macro.relative_mae_improvement >= .05
    return paired, macro


def annotate_timing(predictions):
    columns = ["region", "date", "origin", "train_end", "steps", "snapshot_sha", "snapshot_available_at"]
    timing = predictions[columns].drop_duplicates().copy()
    if timing.duplicated(["region", "date"]).any():
        raise ValueError("Conflicting timing metadata for the same forecast")
    starts, ends, intermediate = [], [], []
    for row in timing.itertuples():
        target = midnight(row.date)
        origin = pd.Timestamp(row.origin)
        distance = (target.date() - pd.Timestamp(row.train_end).date()).days
        if origin != target or distance != row.steps:
            raise ValueError("Origin, target date and history gap disagree")
        starts.append(target.isoformat())
        ends.append(midnight(target.date() + timedelta(days=1)).isoformat())
        intermediate.append(distance - 1)
    timing["target_start"] = starts
    timing["target_end"] = ends
    timing["intermediate_unobserved_days"] = intermediate
    return timing


def independent_baseline_audit(reports: Path, expected: pd.DataFrame):
    """Use stdlib CSV arithmetic, independently from saved score tables."""
    selected, baselines = {}, {}
    with (reports / "test_predictions_selected.csv").open() as f:
        for row in csv.DictReader(f):
            if row["kind"] == "selected" and row["label_ready"].lower() == "true":
                selected[(row["region"], row["date"], row["target"])] = (float(row["prediction"]), float(row["actual"]))
    with (reports / "test_predictions.csv").open() as f:
        for row in csv.DictReader(f):
            if row["model"] in BASELINES and row["label_ready"].lower() == "true":
                key = row["region"], row["date"], row["target"]
                baselines[(key, row["model"])] = (float(row["prediction"]), float(row["actual"]))
    for model in BASELINES:
        if {key for key, name in baselines if name == model} != set(selected):
            raise AssertionError("Baseline and selected model use different dates")
    regional = defaultdict(list)
    for key, (prediction, actual) in selected.items():
        for model in BASELINES:
            baseline, other_actual = baselines[(key, model)]
            assert actual == other_actual
            regional[(key[0], key[2], model)].append((abs(prediction - actual), abs(baseline - actual)))
    for row in expected.loc[expected.model.isin(BASELINES)].itertuples():
        groups = [errors for (_, target, model), errors in regional.items() if target == row.target and model == row.model]
        selected_mae = sum(sum(e[0] for e in errors) / len(errors) for errors in groups) / len(groups)
        baseline_mae = sum(sum(e[1] for e in errors) / len(errors) for errors in groups) / len(groups)
        assert abs(selected_mae - row.selected_mae) < 1e-10
        assert abs(baseline_mae - row.baseline_mae) < 1e-10
    result = {"passed": True, "paired_forecast_cases": len(selected), "fixed_baselines_checked": BASELINES,
              "macro_mae_checked_rows": len(BASELINES) * 2,
              "method": "independent stdlib CSV paired-date checks and regional/macro absolute errors"}
    (reports / "baseline_comparison_audit.json").write_text(json.dumps(result, indent=2))
    return result


def baseline_comparison(reports: Path):
    all_metrics = pd.read_csv(reports / "test_metrics_all_candidates.csv")
    selected_metrics = pd.read_csv(reports / "test_metrics_selected.csv")
    paired, macro = summarize_baselines(all_metrics, selected_metrics)
    paired.to_csv(reports / "baseline_comparison_by_region.csv", index=False)
    macro.to_csv(reports / "baseline_comparison.csv", index=False)
    all_macro = all_metrics.groupby(["target", "model"]).agg(
        macro_mae=("mae", "mean"), macro_rmse=("rmse", "mean"), regions=("region", "size"), forecast_days=("n", "sum")).reset_index()
    selected_macro = pd.read_csv(reports / "test_macro_metrics.csv").replace({"model": {
        "selected": "validation_selected_model", "baseline": "validation_selected_baseline"}})
    all_macro = pd.concat([all_macro, selected_macro], ignore_index=True)
    all_macro.to_csv(reports / "test_macro_all_models.csv", index=False)
    selection = pd.read_csv(reports / "model_selection.csv")
    selection.query("kind == 'baseline'").groupby(["target", "model"]).size().rename("regions").reset_index().to_csv(
        reports / "baseline_selection_summary.csv", index=False)
    timing = annotate_timing(pd.read_csv(reports / "predictions_unlabelled.csv"))
    timing.to_csv(reports / "forecast_timing.csv", index=False)
    audit = {"passed": True, "regional_origins": len(timing), "target_start_equals_origin": True,
             "steps_equal_calendar_distance_from_train_end": True,
             "minimum_steps": int(timing.steps.min()), "maximum_steps": int(timing.steps.max()),
             "october_1_example": timing.loc[(timing.date == "2026-10-01") & (timing.region == "Київ")].to_dict("records")}
    (reports / "timing_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    independent_baseline_audit(reports, macro)
    return macro, all_macro
