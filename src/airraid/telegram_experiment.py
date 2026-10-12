"""Paired ARIMA/ARIMAX ablation; historical public-web replay is exploratory.

The original baseline experiment remains immutable. Strict observed-time mode
refuses retrospective scores when no captured training history exists.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .evaluation import make_tasks, score
from .models import bounded
from .series import midnight
from .sources import load_ledger, verify_checksums
from .telegram import FEATURES, KEYWORDS, feature_table, load_messages


def fit_with_text(y, train_x, future_x, family):
    values = np.asarray(y, dtype=float)
    x, future = np.asarray(train_x, dtype=float), np.asarray(future_x, dtype=float)
    if len(values) != len(x) or len(values) < 60 or not len(future):
        raise ValueError("Misaligned or insufficient exogenous training data")
    if not all(np.isfinite(a).all() for a in [values, x, future]):
        raise ValueError("Nonfinite model inputs")
    # Transform is fixed; means/stds and the constant-column mask use training only.
    x, future = np.log1p(x), np.log1p(future)
    means, stds = x.mean(axis=0), x.std(axis=0)
    keep = stds > 1e-10
    if np.ptp(values) < 1e-12:
        return float(values[-1]), "constant_series", ""
    if not keep.any():
        x = future = None
    else:
        x, future = (x[:, keep] - means[keep]) / stds[keep], (future[:, keep] - means[keep]) / stds[keep]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = ARIMA(values, exog=x, order=(1, 0, 1), trend="c",
                        seasonal_order=(1, 0, 0, 7) if family == "sarima_7" else (0, 0, 0, 0)).fit(
                            method_kwargs={"maxiter": 80})
            if not fit.mle_retvals.get("converged", True):
                raise RuntimeError("optimizer_not_converged")
            raw = float(np.asarray(fit.forecast(len(future_x), exog=future))[-1])
            if not np.isfinite(raw):
                raise RuntimeError("non_finite_output")
            status = ""
        except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
            raw, status = float(values[-7:].mean()), f"fallback_mean_7:{type(exc).__name__}:{str(exc)[:120]}"
    return raw, status, "; ".join(sorted(set(str(w.message)[:140] for w in caught)))


def text_forecast(task):
    rows = []
    for target in ["count", "hours"]:
        family = task["families"][target]
        raw, status, notes = fit_with_text(task["count"] if target == "count" else task["share"],
                                         task["train_x"], task["future_x"], family)
        point = bounded(raw, target)
        scale = task["day_hours"] if target == "hours" else 1
        rows.append({**{k: task[k] for k in ["region", "date", "origin", "train_start", "train_end", "train_n",
                                            "steps", "day_hours", "snapshot_available_at", "max_feature_origin"]},
                     "target": target, "family": family, "model": "arima_with_telegram_assumed",
                     "raw_prediction": raw * scale, "prediction": point * scale,
                     "model_output": point, "was_clipped": abs(raw - point) > 1e-12,
                     "status": status, "warnings": notes, "historical_availability_proven": False})
    return rows


def paired_metrics(d):
    scores = score(d)
    macro = scores.groupby(["target", "model"]).agg(macro_mae=("mae", "mean"), macro_rmse=("rmse", "mean"),
                                                       regions=("region", "size"), forecast_days=("n", "sum")).reset_index()
    return scores, macro


def block_comparison(paired, seed=42, repetitions=2000):
    """Circular seven-calendar-day block bootstrap; all regions stay together."""
    output = []
    for target, d in paired.groupby("target"):
        errors = d.assign(error=(d.prediction - d.actual).abs()).pivot(
            index=["date", "region"], columns="model", values="error")
        models = ["arima_without_telegram", "arima_with_telegram_assumed", "mean_7", "original_selected"]
        if errors[models].isna().any().any():
            raise ValueError("Unpaired evaluation origins")
        dates, regions = sorted(d.date.unique()), sorted(d.region.unique())
        cubes = {m: errors[m].unstack("region").reindex(index=dates, columns=regions).to_numpy() for m in models}
        point = {m: float(np.nanmean(np.nanmean(a, axis=0))) for m, a in cubes.items()}
        generator = np.random.default_rng(seed)
        boot = {m: [] for m in ["arima_without_telegram", "mean_7", "original_selected"]}
        for _ in range(repetitions):
            starts = generator.integers(0, len(dates), size=math.ceil(len(dates) / 7))
            indices = np.concatenate([(s + np.arange(7)) % len(dates) for s in starts])[:len(dates)]
            mae = {m: float(np.nanmean(np.nanmean(a[indices], axis=0))) for m, a in cubes.items()}
            for m in boot:
                boot[m].append(mae[m] - mae["arima_with_telegram_assumed"])
        for comparator, values in boot.items():
            lower, upper = np.quantile(values, [.025, .975])
            gain = 1 - point["arima_with_telegram_assumed"] / point[comparator] if point[comparator] else None
            output.append({"target": target, "comparator": comparator, "comparator_mae": point[comparator],
                           "telegram_mae": point["arima_with_telegram_assumed"], "relative_mae_improvement": gain,
                           "mae_difference_ci_low": lower, "mae_difference_ci_high": upper,
                           "meets_numeric_5_percent": bool(gain is not None and gain >= .05),
                           "certified_success": False, "evidence": "publication_assumed_exploratory"})
    return pd.DataFrame(output)


def independent_metric_audit(output):
    """Separate csv/math calculation; also verify identical label sets, no silent model drops."""
    groups, keys, truths, seen = {}, {}, {}, set()
    with (output / "paired_test_predictions.csv").open() as f:
        for r in csv.DictReader(f):
            if r["label_ready"] != "True":
                continue
            group = (r["region"], r["target"], r["model"])
            row_key = (r["region"], r["date"], r["target"], r["model"])
            if row_key in seen:
                raise AssertionError("Duplicate regional forecast")
            seen.add(row_key)
            truth_key = (r["region"], r["date"], r["target"])
            actual = float(r["actual"])
            if truth_key in truths and truths[truth_key] != actual:
                raise AssertionError("Models have different evaluation labels")
            truths[truth_key] = actual
            groups.setdefault(group, []).append(abs(float(r["prediction"]) - float(r["actual"])))
            keys.setdefault((r["region"], r["target"]), {}).setdefault(r["model"], []).append(r["date"])
            if float(r["prediction"]) < 0 or (r["target"] == "hours" and float(r["prediction"]) > float(r["day_hours"]) + 1e-9):
                raise AssertionError("Invalid physical range")
    for models in keys.values():
        if len(models) != 4 or len({tuple(sorted(v)) for v in models.values()}) != 1:
            raise AssertionError("Unpaired model dates")
    maximum = 0
    with (output / "metrics_by_region.csv").open() as f:
        for r in csv.DictReader(f):
            errors = groups[(r["region"], r["target"], r["model"])]
            delta = abs(sum(errors) / len(errors) - float(r["mae"]))
            maximum = max(maximum, delta)
            rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
            if delta > 1e-10 or abs(rmse - float(r["rmse"])) > 1e-10 or len(errors) != int(r["n"]):
                raise AssertionError("Independent Telegram metric mismatch")
    audit = {"independent_csv_math": True, "regional_metric_rows": len(groups), "same_dates_and_labels": True,
             "paired_region_days_per_target": len(truths) // 2, "mae_and_rmse_checked": True,
             "physical_bounds_passed": True, "maximum_mae_difference": maximum,
             "strict_no_leakage_certified": False, "reason": "Current web archive cannot prove historical versions"}
    (output / "metrics_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def independent_feature_audit(messages_path, features_path, output):
    """stdlib CSV/datetime replay, separate from pandas feature selection."""
    import gzip
    with gzip.open(messages_path, "rt") as f:
        rows = list(csv.DictReader(f))
    latest = {}
    for r in sorted(rows, key=lambda r: r["observed_at"]):
        latest[r["message_id"]] = r
    candidates = [r for r in latest.values() if r["edited_marker"] == "False" and r["has_text"] == "True"]
    from datetime import timedelta
    checked = 0
    with features_path.open() as f:
        for day in csv.DictReader(f):
            origin = datetime.fromisoformat(day["origin"])
            recent = [r for r in candidates if origin - timedelta(hours=24) <= datetime.fromisoformat(r["published_at"]) < origin]
            expected = [len(recent), sum(r["threat_keyword"] == "True" for r in recent),
                        sum(datetime.fromisoformat(r["published_at"]) >= origin - timedelta(hours=6) for r in recent)]
            if expected != [int(day[c]) for c in FEATURES]:
                raise AssertionError("Independent Telegram feature counts disagree")
            checked += 1
    audit = {"stdlib_independent_counts_match": True, "days_checked": checked,
             "edited_posts_excluded": True, "strict_availability_proven": False}
    (output / "independent_features_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def experiment(data: Path, reports: Path, output: Path, mode="observed", workers=4):
    output.mkdir(parents=True, exist_ok=True)
    verify_checksums(data)
    checksum_path = data / "telegram" / "checksums.json"
    if not checksum_path.exists():
        raise ValueError("Telegram checksums missing; finish and audit capture first")
    for name, expected in json.loads(checksum_path.read_text()).items():
        if hashlib.sha256((data / "telegram" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Telegram checksum mismatch: {name}")
    source_schema = json.loads((data / "telegram" / "source_audit.json").read_text())
    if source_schema["threat_substrings"] != KEYWORDS:
        raise ValueError("Frozen Telegram keyword flags use a different dictionary")
    messages_path = data / "telegram" / "message_versions.csv.gz"
    messages = load_messages(messages_path)
    base_protocol = json.loads((reports / "protocol.json").read_text())
    frozen_files = [reports / n for n in ["protocol.json", "predictions_unlabelled.csv", "model_selection.csv", "test_predictions.csv"]]
    frozen_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_files}
    protocol = {"channel": "war_monitor", "mode": mode, "registered_at": datetime.now(timezone.utc).isoformat(),
                "original_experiment_hashes": frozen_hashes, "messages_sha256": hashlib.sha256(messages_path.read_bytes()).hexdigest(),
                "features": FEATURES, "threat_substrings": KEYWORDS, "windows": "24 and 6 elapsed hours before origin",
                "transform": "log1p then training-only standardization", "baseline_protocol": base_protocol,
                "family_selection": "ARIMA vs SARIMA7 selected using frozen no-text validation MAE only",
                "edited_marker": "excluded in exploratory mode, including hidden-edit risk",
                "test_status": "already examined; exploratory reuse, not an untouched holdout",
                "success_requires": ">=5% paired macro MAE improvement for both targets, positive block CI and proven availability on a new untouched test",
                "bootstrap": {"block_calendar_days": 7, "repetitions": 2000, "seed": 42},
                "historical_availability_proven": False}
    lock = output / "protocol.json"
    if lock.exists():
        previous = json.loads(lock.read_text())
        comparable = {k: v for k, v in previous.items() if k != "registered_at"}
        if comparable != {k: v for k, v in protocol.items() if k != "registered_at"}:
            raise ValueError("Telegram protocol changed; use another output directory")
        protocol = previous
    lock.write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    (output / "protocol.sha256").write_text(hashlib.sha256(lock.read_bytes()).hexdigest() + "\n")
    dates = pd.date_range(base_protocol["validation_start"], base_protocol["test_end"])
    readiness = {"first_observed_at": str(messages.observed_at.min()), "requested_first_origin": midnight(dates[0].date()).isoformat(),
                 "historical_origins_with_any_observed_messages": sum(bool((messages.observed_at < midnight(d.date())).any()) for d in dates),
                 "strict_historical_comparison_possible": False,
                 "reason": "All fetched text versions were first observed after historical forecast origins; absence is unknown, not zero activity.",
                 "next_step": "Accumulate timestamped captures before origins, then freeze a new future test after enough labeled training history."}
    (output / "availability_audit.json").write_text(json.dumps(readiness, indent=2))
    if mode == "observed":
        if readiness["historical_origins_with_any_observed_messages"] == 0:
            (output / "status.json").write_text(json.dumps({"status": "insufficient_point_in_time_text_history", "scores_produced": False}, indent=2))
            print("Strict Telegram comparison: insufficient observed-time text history; no fabricated zero features or scores", flush=True)
            return
        raise ValueError("A new prospective protocol is required; the old test cannot certify strict improvement")
    if mode != "publication_assumed":
        raise ValueError("Unknown Telegram evidence mode")
    # Make family choice and persist it BEFORE accessing test labels in this experiment.
    val = pd.read_csv(reports / "validation_metrics.csv").query("model in ['arima', 'sarima_7']")
    families = val.sort_values(["mae", "rmse", "model"]).groupby(["region", "target"]).first().reset_index()[["region", "target", "model"]]
    families.to_csv(output / "family_selection.csv", index=False)
    ledger = load_ledger(data / "event_ledger.csv.gz")
    snapshots = json.loads((data / "snapshots.json").read_text())
    tasks, omissions = make_tasks(ledger, snapshots, base_protocol["config"],
                                 list(pd.date_range(base_protocol["validation_start"], base_protocol["validation_end"])) +
                                 list(pd.date_range(base_protocol["test_start"], base_protocol["test_end"])), base_protocol["regions"])
    earliest = min(t["train_start"] for t in tasks)
    if messages.published_at.min() > midnight(earliest) - pd.Timedelta(hours=24):
        raise ValueError("Telegram archive does not span training feature windows")
    x = feature_table(messages, pd.date_range(earliest, base_protocol["test_end"]), mode)
    x.to_csv(output / "daily_features.csv", index=False)
    independent_feature_audit(messages_path, output / "daily_features.csv", output)
    indexed = x.set_index("date")
    publication_ok = all(not r.max_publication_used or pd.Timestamp(r.max_publication_used) < pd.Timestamp(r.origin) for r in x.itertuples())
    if not publication_ok:
        raise AssertionError("Future publication included")
    source_stage = {"publication_before_feature_origin": True, "strict_observation_time_passed": False,
                    "edited_marker_excluded": True, "historical_version_proof": False, "feature_days": len(x),
                    "source": "current public web messages, incomplete/deletion and hidden-edit risks"}
    (output / "features_audit.json").write_text(json.dumps(source_stage, indent=2))
    family_lookup = {(r.region, r.target): r.model for r in families.itertuples()}
    for task in tasks:
        training_days = pd.date_range(task["train_start"], task["train_end"]).strftime("%Y-%m-%d")
        future_days = pd.date_range(pd.Timestamp(task["train_end"]) + pd.Timedelta(days=1), task["date"]).strftime("%Y-%m-%d")
        if len(training_days) != task["train_n"] or len(future_days) != task["steps"]:
            raise AssertionError("Calendar/exogenous alignment mismatch")
        task.update({"train_x": indexed.loc[training_days, FEATURES].to_numpy().tolist(),
                     "future_x": indexed.loc[future_days, FEATURES].to_numpy().tolist(),
                     "families": {target: family_lookup[(task["region"], target)] for target in ["count", "hours"]},
                     "max_feature_origin": indexed.loc[future_days[-1], "origin"]})
        if pd.Timestamp(task["max_feature_origin"]) > pd.Timestamp(task["origin"]):
            raise AssertionError("Future exogenous origin")
    print(f"Telegram feature audit saved; {len(tasks)} paired regional origins", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(text_forecast, tasks, chunksize=2), 1):
            rows.extend(result)
            if i % 50 == 0 or i == len(tasks):
                print(f"Telegram ARIMAX {i}/{len(tasks)}", flush=True)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / "predictions_unlabelled.csv", index=False)
    (output / "model_audit.json").write_text(json.dumps({"rows": len(predictions), "finite": bool(np.isfinite(predictions.prediction).all()),
        "fallbacks": int(predictions.status.str.startswith("fallback").sum()), "clipped": int(predictions.was_clipped.sum()),
        "training_scaler_only": True, "exogenous_intermediate_days_known_at_origin": True}, indent=2))
    for partition in ["validation", "test"]:
        baseline = pd.read_csv(reports / f"{partition}_predictions.csv")
        labels = baseline[["region", "date", "target", "label_ready", "label_reason", "actual"]].drop_duplicates()
        with_text = predictions.merge(labels, on=["region", "date", "target"], validate="many_to_one")
        without = baseline.merge(families, on=["region", "target", "model"], validate="many_to_one").copy()
        without["family"] = without["model"]
        without["model"] = "arima_without_telegram"
        mean = baseline.query("model == 'mean_7'").copy()
        chosen = pd.read_csv(reports / "model_selection.csv").query("kind == 'selected'")
        original = baseline.merge(chosen[["region", "target", "model"]], on=["region", "target", "model"], validate="many_to_one")
        original["model"] = "original_selected"
        paired = pd.concat([with_text, without, mean, original], ignore_index=True)
        paired.to_csv(output / f"paired_{partition}_predictions.csv", index=False)
        regional, macro = paired_metrics(paired)
        regional.to_csv(output / ("metrics_by_region.csv" if partition == "test" else "validation_metrics.csv"), index=False)
        macro.to_csv(output / f"{partition}_macro_metrics.csv", index=False)
        if partition == "test":
            usable = paired.loc[paired.label_ready & paired.actual.notna()]
            comparison = block_comparison(usable)
            comparison.to_csv(output / "comparison.csv", index=False)
    independent_metric_audit(output)
    if frozen_hashes != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_files}:
        raise AssertionError("Original experiment was modified")
    (output / "baseline_integrity_audit.json").write_text(json.dumps({"unchanged": True, "sha256": frozen_hashes}, indent=2))
    (output / "status.json").write_text(json.dumps({"status": "exploratory_complete", "strict_success": False,
        "note": "Scores are conditional on unproven publication-time availability; no deployment/model promotion."}, indent=2))
    plot_experiment(output)
    print(comparison.to_string(index=False), flush=True)


def plot_experiment(output):
    from .diagnostics import plt
    daily = pd.read_csv(output / "daily_features.csv")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    axes[0].plot(pd.to_datetime(daily.date), daily.posts_24h, label="Text posts / previous 24h", linewidth=1)
    axes[0].plot(pd.to_datetime(daily.date), daily.threat_posts_24h, label="Keyword posts / previous 24h", linewidth=1)
    axes[0].set_ylabel("Retrieved archive posts")
    axes[0].legend()
    macro = pd.read_csv(output / "test_macro_metrics.csv")
    for axis, target, unit in zip(axes[1:], ["count", "hours"], ["alert starts", "hours"]):
        d = macro.query("target == @target").set_index("model").loc[["mean_7", "original_selected", "arima_without_telegram", "arima_with_telegram_assumed"]]
        bars = axis.barh(["Mean 7", "Original selected", "Matched ARIMA", "ARIMA + Telegram (assumed)"], d.macro_mae,
                         color=["#89969d", "#19858b", "#326dac", "#d18531"])
        axis.bar_label(bars, fmt="%.3f", padding=5)
        axis.set_xlim(0, d.macro_mae.max() * 1.12)
        axis.set_xlabel(f"Macro MAE ({unit}); lower is better")
    fig.suptitle("Telegram exploratory replay — historical text versions NOT proven", fontsize=14)
    fig.savefig(output / "experiment.png", dpi=140)
    plt.close(fig)


def report_section(data, reports):
    """Render explicit evidence limitations alongside the numerical ablation."""
    output = reports / "telegram" / "publication_assumed"
    if not (output / "comparison.csv").exists():
        return None
    source = json.loads((data / "telegram" / "source_audit.json").read_text())
    model_audit = json.loads((output / "model_audit.json").read_text())
    comparison = pd.read_csv(output / "comparison.csv")
    table = comparison.copy()
    table["relative_mae_improvement"] *= 100
    table = table[["target", "comparator", "comparator_mae", "telegram_mae", "relative_mae_improvement", "mae_difference_ci_low", "mae_difference_ci_high"]]
    table.columns = ["Ціль", "Орієнтир", "MAE орієнтира", "MAE з Telegram", "Покращення, %", "Різниця MAE: CI 2,5%", "CI 97,5%"]
    direct = comparison.query("comparator == 'arima_without_telegram'").set_index("target")
    words = []
    for target, label in [("count", "кількість"), ("hours", "години")]:
        gain = float(direct.loc[target, "relative_mae_improvement"])
        words.append(f"{label}: MAE {'зменшився' if gain >= 0 else 'зріс'} на {abs(gain):.2%}")
    text = (
        "**Це умовний дослідницький replay, а не підтверджений результат без витоку.** "
        "Строга перевірка за фактичним observed_at не має жодного історичного origin із доступними текстами: "
        "їх зібрано вже після тесту. Вона не підставляє нулі й не видає метрики. Нижче — окремий сценарій "
        "із непідтвердженим припущенням, що нинішні тексти без видимої позначки редагування були доступні на published_at. "
        "У Telegram існує edit_hide, тому відсутність позначки не доводить незмінності тексту. Видалені дописи також не відновлені. "
        "[Документація Telegram](https://core.telegram.org/constructor/message).\n\n"
        f"Архів: {source['unique_messages']:,} доступних дописів, {source['pages']} збережених сторінок, "
        f"публікації {source['first_published']} — {source['last_published']}; фактично отримано "
        f"{source['first_observed']} — {source['last_observed']}. Видима позначка edited: {source['edited_marker_messages']}; "
        f"без тексту: {source['no_text_messages']}; згадок технічних робіт: {source['technical_pause_messages']}; "
        f"пропущених ID у діапазоні: {source['missing_ids_in_span']} (причину кожного не встановлено). "
        "Це не повний архів каналу від його створення. Нуль ознаки означає нуль придатних дописів у зібраному архіві, "
        "а не доведену відсутність публікацій чи загроз.\n\n"
        "Три ознаки: кількість текстових дописів за попередні 24 години, кількість дописів із фіксованими "
        "підрядками загроз за 24 години, кількість текстових дописів за 6 годин. Це минулі години до origin, "
        "не тексти цільової доби. Словник не розрізняє загрозу та відбій. Ознаки спільні для каналу; кожен регіон має "
        "окрему модель, географічне розпізнавання й аналіз медіа не застосовано. log1p і стандартизація тільки за навчальним "
        "вікном. Лічильники переглядів, реакції та цитати відповідей не використовуються.\n\n"
        "Парне порівняння ARIMA без тексту та ARIMA з зовнішніми ознаками (ARIMAX): однакові 180 готових днів, "
        "порядок, регіони, цілі, origin і затримка джерела. Сезонний варіант обрано лише за MAE замороженої валідації "
        "ARIMA без тексту. Для проміжних кроків використовуються лише Telegram-вікна, що закінчилися до origin; "
        "майбутні значення тривог не підставляються. Mean_7 та початкова обрана модель додані як орієнтири.\n\n"
        f"Навчання: {model_audit['rows']} прогнозів обох періодів; {model_audit['fallbacks']} явних резервних mean_7 "
        f"({model_audit['fallbacks']/model_audit['rows']:.2%}), {model_audit['clipped']} обмежень виходу "
        f"({model_audit['clipped']/model_audit['rows']:.2%}). Усі виходи скінченні; кількість ≥0, години в межах доби. "
        "Резервні прогнози включено до метрик.\n\n"
        f"За умовного припущення доступності, проти парної ARIMA: {'; '.join(words)}. "
        "Додатна різниця MAE в таблиці означає користь Telegram. Інтервали — circular block bootstrap по 7 календарних "
        "днів, 2000 повторів; усі регіони переміщуються разом. На 28 днях це лише груба оцінка невизначеності, "
        "без поправки на численні порівняння. Переглянутий раніше тест повторно використано лише для дослідження. "
        "**Підтверджене покращення й критерій успіху не встановлені; базова модель не замінена.**\n\n"
        "Критерій наступної суворої перевірки: новий недоторканий період, тексти збережені до origin, щонайменше 5% "
        "зниження парного macro MAE для обох цілей і додатна нижня межа блочного інтервалу. Для цього треба накопичити "
        "історію захоплень та готових міток і зафіксувати новий протокол до відкриття його тесту.\n\n"
        "[Протокол](telegram/publication_assumed/protocol.json) · [Строга доступність](telegram/observed/availability_audit.json) · "
        "[Ознаки](telegram/publication_assumed/daily_features.csv) · [Валідація](telegram/publication_assumed/validation_metrics.csv) · "
        "[Метрики за регіонами](telegram/publication_assumed/metrics_by_region.csv) · "
        "[Парні прогнози](telegram/publication_assumed/paired_test_predictions.csv) · "
        "[Незалежний аудит ознак](telegram/publication_assumed/independent_features_audit.json) · "
        "[Незалежний аудит метрик](telegram/publication_assumed/metrics_audit.json) · "
        "[Незмінність базового експерименту](telegram/publication_assumed/baseline_integrity_audit.json).\n\n"
        "![Telegram експеримент](telegram/publication_assumed/experiment.png)"
    )
    from .reporting import md_table
    # Relative links in this standalone appendix need to point back to the main reports directory.
    appendix = text.replace("](telegram/", "](../../telegram/")
    (output / "experiment.md").write_text("# Telegram: умовний історичний експеримент\n\n" + appendix + "\n\n" + md_table(table))
    return "Telegram: порівняння з текстами та без", text, table
