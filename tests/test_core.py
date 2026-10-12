from datetime import datetime, timezone
import pickle

import numpy as np
import pandas as pd
import pytest

from airraid.models import bounded, forecast_one
from airraid.series import aggregate, day_hours, mark_censored, midnight
from airraid.sources import as_of, decode_active, snapshot_at


def events(rows):
    d = pd.DataFrame(rows, columns=["region", "started_at", "finished_at", "naive"])
    for c in ["started_at", "finished_at"]:
        d[c] = pd.to_datetime(d[c], utc=True)
    return d


def test_midnight_split_and_union():
    d = events([("Київ", "2026-01-01T21:00Z", "2026-01-02T00:00Z", False),
                ("Київ", "2026-01-01T22:30Z", "2026-01-02T01:00Z", False)])
    a = aggregate(d, pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-02").date(), ["Київ"])
    assert a["count"].tolist() == [1, 1]
    assert a.hours.tolist() == [1, 3]


@pytest.mark.parametrize("day,expected", [("2026-03-29", 23), ("2026-10-25", 25), ("2026-07-01", 24)])
def test_dst_day_lengths(day, expected):
    assert day_hours(day) == expected


@pytest.mark.parametrize("day", ["2026-03-29", "2026-10-25"])
def test_full_day_alert_respects_dst(day):
    s = midnight(day)
    e = midnight((s + pd.DateOffset(days=1)).date())
    d = events([("Київ", s.tz_convert("UTC").isoformat(), e.tz_convert("UTC").isoformat(), False)])
    a = aggregate(d, s.date(), s.date(), ["Київ"]).iloc[0]
    assert a.hours == day_hours(day)
    assert a.share == 1


def test_naive_30_minutes_unchanged():
    d = events([("Київ", "2026-01-01T10:00Z", "2026-01-01T10:30Z", True)])
    a = aggregate(d, pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-01").date(), ["Київ"]).iloc[0]
    assert a.hours == .5 and a["count"] == 1 and a.naive_starts == 1


def test_real_zero_and_unresolved_are_distinguished():
    d = events([("Київ", "2026-01-01T10:00Z", "2026-01-01T10:30Z", True)])
    a = aggregate(d, pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-03").date(), ["Київ"])
    s = {"committed_at": "2026-01-04T05:00Z", "active": {"Київ": {"started_at": "2026-01-03T10:00Z"}}}
    a = mark_censored(a, s)
    assert a.label_ready.tolist() == [True, True, False]
    assert a.hours.iloc[1] == 0


def test_negative_duration_rejected():
    d = events([("Київ", "2026-01-01T10:00Z", "2026-01-01T09:00Z", False)])
    with pytest.raises(ValueError, match="Negative"):
        aggregate(d, pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-01").date(), ["Київ"])


def test_late_naive_event_is_not_known_at_artificial_finish():
    d = events([("Київ", "2026-01-01T10:00Z", "2026-01-01T10:30Z", True)])
    d["available_at"] = pd.Timestamp("2026-01-02T05:00Z")
    d["unavailable_at"] = pd.NaT
    assert as_of(d, pd.Timestamp("2026-01-01T23:00Z")).empty
    assert len(as_of(d, pd.Timestamp("2026-01-02T05:00Z"))) == 1


def test_future_changes_do_not_change_past_input_or_prediction():
    past = events([("Київ", "2026-01-01T10:00Z", "2026-01-01T10:30Z", False)])
    past["available_at"] = pd.Timestamp("2026-01-02T05:00Z")
    past["unavailable_at"] = pd.NaT
    future = past.copy()
    future["available_at"] = pd.Timestamp("2026-01-10T05:00Z")
    future["finished_at"] = pd.Timestamp("2026-01-09T20:00Z")
    origin = pd.Timestamp("2026-01-03T00:00Z")
    a = as_of(past, origin)
    b = as_of(pd.concat([past, future]), origin)
    pd.testing.assert_frame_equal(a, b)
    y = np.array([0, 1, 0, 2, 0, 0, len(a)] * 4)
    z = np.array([0, 1, 0, 2, 0, 0, len(b)] * 4)
    assert forecast_one(y, 2, "arima")[0] == forecast_one(z, 2, "arima")[0]


def test_versions_respect_deletions_and_replacements():
    d = pd.DataFrame({"available_at": pd.to_datetime(["2026-01-01", "2026-01-03"], utc=True),
                      "unavailable_at": pd.to_datetime(["2026-01-03", None], utc=True), "value": [1, 2]})
    assert as_of(d, pd.Timestamp("2026-01-02", tz="UTC")).value.tolist() == [1]
    assert as_of(d, pd.Timestamp("2026-01-03", tz="UTC")).value.tolist() == [2]


def test_snapshot_availability_and_staleness():
    snapshots = [{"sha": "a", "available_at": "2026-01-01T05:00Z"},
                 {"sha": "b", "available_at": "2026-01-02T05:00Z"}]
    assert snapshot_at(snapshots, pd.Timestamp("2026-01-02T00:00Z"))["sha"] == "a"
    with pytest.raises(ValueError, match="stale"):
        snapshot_at(snapshots, pd.Timestamp("2026-01-05T00:00Z"))


def test_forecast_limits_and_non_finite():
    assert bounded(-.3, "count") == 0
    assert bounded(1.4, "count") == 1.4
    assert bounded(-.2, "hours") == 0
    assert bounded(1.2, "hours") == 1
    for x in [float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            bounded(x, "hours")


def test_weekly_baseline_accounts_for_latency():
    y = np.arange(1, 15)
    assert forecast_one(y, 1, "weekly_naive")[0] == 8
    assert forecast_one(y, 3, "weekly_naive")[0] == 10
    assert forecast_one(y, 8, "weekly_naive")[0] == 8


def test_constant_and_all_zero_models():
    for model in ["arima", "sarima_7", "ses", "ets_7"]:
        assert forecast_one(np.zeros(60), 3, model)[0] == 0


def test_smoothing_nonconvergence_has_auditable_fallback(monkeypatch):
    import airraid.models as models
    class FailedFit:
        mle_retvals = {"success": False}
    class FailedSmoothing:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, **kwargs):
            return FailedFit()
    monkeypatch.setattr(models, "SimpleExpSmoothing", FailedSmoothing)
    y = np.arange(60, dtype=float)
    prediction, status, _ = forecast_one(y, 2, "ses")
    assert prediction == y[-7:].mean()
    assert status.startswith("fallback_mean_7") and "optimizer_not_converged" in status


def test_safe_pickle_interpreter():
    original = {"Київ": {"started_at": datetime(2026, 1, 1, tzinfo=timezone.utc), "finished_at": None, "naive": False}}
    parsed = decode_active(pickle.dumps(original, protocol=4))
    assert parsed["Київ"]["started_at"] == "2026-01-01T00:00:00+00:00"
    class Unsupported:
        def __reduce__(self):
            return (eval, ("1 + 1",))
    with pytest.raises(ValueError, match="Unsupported pickle symbol"):
        decode_active(pickle.dumps(Unsupported(), protocol=4))


def test_whole_training_task_invariant_to_future_snapshot_and_correction():
    import json
    from pathlib import Path
    from airraid.evaluation import make_tasks
    from airraid.sources import load_ledger
    root = Path(__file__).resolve().parents[1]
    ledger = load_ledger(root / "data/event_ledger.csv.gz")
    snapshots = json.loads((root / "data/snapshots.json").read_text())
    config = json.loads((root / "config.json").read_text())
    dates = [pd.Timestamp("2026-08-25")]
    first, omitted = make_tasks(ledger, snapshots, config, dates, ["Київ"])
    future = ledger.iloc[[-1]].copy()
    future["available_at"] = pd.Timestamp("2026-12-01", tz="UTC")
    future["finished_at"] = pd.Timestamp("2026-11-30", tz="UTC")
    changed = pd.concat([ledger, future], ignore_index=True)
    changed.loc[0, "unavailable_at"] = pd.Timestamp("2026-12-01", tz="UTC")
    future_snap = {**snapshots[-1], "available_at": "2026-12-01T00:00Z", "active": {}}
    second, omitted_again = make_tasks(changed, snapshots + [future_snap], config, dates, ["Київ"])
    assert first == second and omitted == omitted_again
