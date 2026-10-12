"""Guard against conflating history gap, target day and benchmark choice."""
from pathlib import Path

import pandas as pd
import pytest

from airraid.comparison import annotate_timing, summarize_baselines


def metrics():
    all_models = pd.DataFrame([
        {"region": "Київ", "target": "count", "model": name, "n": 28, "mae": error, "rmse": error}
        for name, error in [("last", 10.), ("mean_7", 2.), ("weekly_naive", 8.)]])
    selected = pd.DataFrame([
        {"region": "Київ", "target": "count", "model": name, "n": 28, "mae": error, "rmse": error}
        for name, error in [("selected", 1.9), ("baseline", 10.)]])
    return all_models, selected


def test_mean7_comparison_does_not_inherit_weak_baseline_gain():
    all_models, selected = metrics()
    _, macro = summarize_baselines(all_models, selected)
    result = macro.set_index("model")
    assert result.loc["last", "relative_mae_improvement"] == pytest.approx(.81)
    assert result.loc["mean_7", "relative_mae_improvement"] == pytest.approx(.05)
    assert result.loc["validation_selected_baseline", "relative_mae_improvement"] == pytest.approx(.81)


def test_unpaired_regional_case_counts_are_rejected():
    all_models, selected = metrics()
    all_models.loc[all_models.model == "mean_7", "n"] = 27
    with pytest.raises(ValueError, match="same regional cases"):
        summarize_baselines(all_models, selected)


def test_saved_october1_forecast_uses_september29_and_targets_october1():
    root = Path(__file__).resolve().parents[1]
    forecasts = pd.read_csv(root / "reports/predictions_unlabelled.csv")
    example = forecasts.query("date == '2026-10-01' and region == 'Київ'")
    timing = annotate_timing(example)
    assert len(timing) == 1
    row = timing.iloc[0]
    assert row.train_end == "2026-09-29" and row.steps == 2
    assert row.intermediate_unobserved_days == 1
    assert row.target_start == row.origin == "2026-10-01T00:00:00+03:00"
    assert row.target_end == "2026-10-02T00:00:00+03:00"


def test_inconsistent_target_day_cannot_be_silently_relabeled():
    data = pd.DataFrame([{"region": "Київ", "date": "2026-10-02", "origin": "2026-10-01T00:00:00+03:00",
                          "train_end": "2026-09-29", "steps": 2, "snapshot_sha": "a",
                          "snapshot_available_at": "2026-09-30T04:44:09Z"}])
    with pytest.raises(ValueError, match="disagree"):
        annotate_timing(data)
