"""Explicit frozen reproduction, source preparation, reporting and daily forecasting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .audit import independent_aggregation_check, git_replay_check
from .diagnostics import seasonality, source_audit
from .evaluation import design, evaluate, make_tasks, run_predictions
from .reporting import build_report, descriptive
from .sources import load_ledger, prepare_bundle, verify_checksums

# Data are checkout artifacts, not files embedded in the installed package.
ROOT = Path.cwd()


def load(data):
    verify_checksums(data)
    return load_ledger(data / "event_ledger.csv.gz"), json.loads((data / "snapshots.json").read_text())


def daily_forecast(ledger, snapshots, protocol, reports, date, workers=1):
    tasks, omitted = make_tasks(ledger, snapshots, protocol["config"], [date], protocol["regions"])
    output = reports / f"forecast_{date}"
    output.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(reports / "model_selection.csv")
    predictions = run_predictions(tasks, workers, output / "all_candidates.csv")
    result = predictions.merge(selection[["region", "target", "kind", "model"]], on=["region", "target", "model"], validate="many_to_many")
    result.to_csv(output / "selected.csv", index=False)
    pd.DataFrame(omitted, columns=["region", "date", "reason"]).to_csv(output / "omissions.csv", index=False)
    print(f"Saved {len(result.query('kind == \"selected\"'))} selected forecasts for {date}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "report", "audit", "prepare", "forecast", "telegram-capture", "telegram-experiment"])
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--repo", type=Path, help="Local source Git history for prepare/independent replay audit")
    parser.add_argument("--runs", type=Path, help="GitHub Actions workflow_runs JSON for prepare")
    parser.add_argument("--date", help="Kyiv calendar date YYYY-MM-DD for forecast")
    parser.add_argument("--since", default="2026-02-19", help="Earliest UTC publication date for Telegram capture")
    parser.add_argument("--max-pages", type=int, default=1200)
    parser.add_argument("--text-mode", choices=["observed", "publication_assumed"], default="observed",
                        help="observed enforces capture time; publication_assumed is explicitly exploratory")
    parser.add_argument("--experiment-output", type=Path, help="Separate directory for a new Telegram protocol")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "telegram-capture":
        from .telegram import capture
        capture(args.data / "telegram", args.since, args.max_pages)
        return
    if args.command == "telegram-experiment":
        from .telegram_experiment import experiment
        experiment(args.data, args.output, args.experiment_output or args.output / "telegram" / args.text_mode, args.text_mode,
                   args.workers if args.workers is not None else 4)
        return
    if args.command == "prepare":
        if args.repo is None or args.runs is None:
            parser.error("prepare requires --repo and --runs")
        print(json.dumps(prepare_bundle(args.repo, args.runs, args.data), indent=2))
        return
    ledger, snapshots = load(args.data)
    config = json.loads(args.config.read_text())
    if config["timezone"] != "Europe/Kyiv":
        parser.error("Only Europe/Kyiv is supported")
    from .models import BASELINES, CANDIDATES
    if set(config["models"]) != set(BASELINES + CANDIDATES):
        parser.error("The locked comparison requires all seven declared models")
    workers = args.workers if args.workers is not None else config["workers"]
    if workers < 1:
        parser.error("workers must be positive")
    if args.command == "run":
        protocol = design(ledger, snapshots, config)
        path = args.output / "protocol.json"
        serialized = json.dumps(protocol, ensure_ascii=False, indent=2)
        # A changed experiment must use a separate output directory.
        if path.exists() and json.loads(path.read_text()) != protocol:
            parser.error("Existing protocol differs; use a new --output directory")
        path.write_text(serialized)
        (args.output / "protocol.sha256").write_text(hashlib.sha256(serialized.encode()).hexdigest() + "\n")
        source_audit(ledger, snapshots, protocol, args.output)
        seasonality(ledger, snapshots, protocol, args.output)
        print("Source and seasonality stage audits passed", flush=True)
        evaluate(ledger, snapshots, protocol, args.output, workers)
        independent_aggregation_check(args.data, args.output)
        if args.repo:
            git_replay_check(args.repo, args.data, args.output)
        descriptive(ledger, snapshots, protocol, args.output)
        verification = {"pipeline_stage_audits_passed": True,
                        "frozen_data_checksums_passed": True,
                        "unit_tests": "Run python -m pytest -q; separate from pipeline",
                        "clean_environment_reproduction": "Not independently asserted by a single pipeline run"}
        (args.output / "verification.json").write_text(json.dumps(verification, indent=2))
        print(build_report(args.data, args.output))
    elif args.command == "forecast":
        if not args.date:
            parser.error("forecast requires --date")
        protocol = json.loads((args.output / "protocol.json").read_text())
        if pd.Timestamp(args.date).date() <= pd.Timestamp(protocol["test_end"]).date():
            parser.error("forecast requires a date after the locked test; historical results are in test_predictions.csv")
        daily_forecast(ledger, snapshots, protocol, args.output, args.date, workers)
    elif args.command == "report":
        protocol = json.loads((args.output / "protocol.json").read_text())
        descriptive(ledger, snapshots, protocol, args.output)
        print(build_report(args.data, args.output))
    else:
        protocol = json.loads((args.output / "protocol.json").read_text())
        source_audit(ledger, snapshots, protocol, args.output)
        independent_aggregation_check(args.data, args.output)
        if args.repo:
            git_replay_check(args.repo, args.data, args.output)
        from .reporting import independent_metrics_check
        print(independent_metrics_check(args.output))
        from .comparison import baseline_comparison
        comparisons, _ = baseline_comparison(args.output)
        print(comparisons.query("model == 'mean_7'").to_string(index=False))


if __name__ == "__main__":
    main()
