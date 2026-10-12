"""Replay selected ARIMAX and matched ARIMA forecasts from frozen inputs, offline."""
from __future__ import annotations

import hashlib
import json
import sys
import argparse
from pathlib import Path

# Match run.py: verification must also work in an uninstalled source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from airraid.evaluation import make_tasks
from airraid.models import bounded, forecast_one
from airraid.sources import load_ledger, verify_checksums
from airraid.telegram_experiment import independent_feature_audit, independent_metric_audit, text_forecast
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-9,
                        help="Absolute forecast tolerance; CI allows 1e-4 across macOS ARM / Linux x86")
    args = parser.parse_args()
    if not 0 < args.atol <= 1e-4:
        parser.error("Tolerance must be positive and no larger than 1e-4")
    root = Path.cwd()
    data, reports = root / "data", root / "reports"
    output = reports / "telegram/publication_assumed"
    verify_checksums(data)
    for name, expected in json.loads((data / "telegram/checksums.json").read_text()).items():
        assert hashlib.sha256((data / "telegram" / name).read_bytes()).hexdigest() == expected
    protocol = json.loads((reports / "protocol.json").read_text())
    telegram_protocol = json.loads((output / "protocol.json").read_text())
    assert hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest() == (output / "protocol.sha256").read_text().strip()
    for name, expected in telegram_protocol["original_experiment_hashes"].items():
        assert hashlib.sha256((reports / name).read_bytes()).hexdigest() == expected
    ledger = load_ledger(data / "event_ledger.csv.gz")
    snapshots = json.loads((data / "snapshots.json").read_text())
    families = pd.read_csv(output / "family_selection.csv")
    lookup = {(r.region, r.target): r.model for r in families.itertuples()}
    features = pd.read_csv(output / "daily_features.csv").set_index("date")
    saved = pd.read_csv(output / "predictions_unlabelled.csv")
    without = pd.read_csv(reports / "predictions_unlabelled.csv")
    dates = [protocol[k] for k in ["validation_start", "validation_end", "test_start", "test_end"]]
    tasks, _ = make_tasks(ledger, snapshots, protocol["config"], dates, ["Київ", "Харківська область"])
    fallback = saved.loc[saved.status.fillna("").str.startswith("fallback")].iloc[0]
    extra, _ = make_tasks(ledger, snapshots, protocol["config"], [fallback.date], [fallback.region])
    tasks += extra
    maximum, checked, differences = 0.0, 0, []
    from airraid.telegram import FEATURES
    for task in tasks:
        train_days = pd.date_range(task["train_start"], task["train_end"]).strftime("%Y-%m-%d")
        future_days = pd.date_range(pd.Timestamp(task["train_end"]) + pd.Timedelta(days=1), task["date"]).strftime("%Y-%m-%d")
        task.update({"families": {t: lookup[task["region"], t] for t in ["count", "hours"]},
                     "train_x": features.loc[train_days, FEATURES].to_numpy().tolist(),
                     "future_x": features.loc[future_days, FEATURES].to_numpy().tolist(),
                     "max_feature_origin": features.loc[future_days[-1], "origin"]})
        for r in text_forecast(task):
            old = saved.loc[(saved.region == r["region"]) & (saved.date == r["date"]) & (saved.target == r["target"])].iloc[0]
            difference = float(abs(r["prediction"] - old.prediction))
            maximum = max(maximum, difference)
            differences.append({"region": r["region"], "date": r["date"], "target": r["target"], "model": "with_text",
                                "difference": difference, "status_matches": r["status"] == (old.status if pd.notna(old.status) else "")})
            family = task["families"][r["target"]]
            raw, status, _ = forecast_one(task["count"] if r["target"] == "count" else task["share"], task["steps"], family)
            point = bounded(raw, r["target"]) * (task["day_hours"] if r["target"] == "hours" else 1)
            old_without = without.loc[(without.region == r["region"]) & (without.date == r["date"]) &
                                      (without.target == r["target"]) & (without.model == family)].iloc[0]
            difference = float(abs(point - old_without.prediction))
            maximum = max(maximum, difference)
            differences.append({"region": r["region"], "date": r["date"], "target": r["target"], "model": "without_text",
                                "difference": difference, "status_matches": status == (old_without.status if pd.notna(old_without.status) else "")})
            checked += 2
    independent_feature_audit(data / "telegram/message_versions.csv.gz", output / "daily_features.csv", output)
    independent_metric_audit(output)
    passed = bool(maximum < args.atol and all(r["status_matches"] for r in differences))
    evidence = {"offline_replay_passed": passed, "forecast_rows_recomputed": checked,
                "regional_origins": len(tasks), "maximum_prediction_difference": maximum,
                "absolute_tolerance": args.atol, "all_statuses_match": all(r["status_matches"] for r in differences),
                "validation_and_test_boundaries": True, "fallback_case_replayed": True,
                "source_and_original_baseline_hashes_verified": True,
                "historical_text_availability_proven": False}
    (output / "reproduction_audit.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
    if not passed:
        print(json.dumps(differences, ensure_ascii=False, indent=2))
        raise AssertionError("Forecast replay exceeds declared tolerance or optimizer/fallback status changed")


if __name__ == "__main__":
    main()
