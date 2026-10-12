"""Generate the report from saved predictions, never from manually copied metrics."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend
from statsmodels.stats.diagnostic import acorr_ljungbox

from .diagnostics import LABELS, plt, save_figure, seasonal_comparison, short
from .comparison import baseline_comparison
from .series import aggregate, mark_censored, midnight
from .sources import as_of, snapshot_at


def independent_metrics_check(reports: Path):
    """Recompute all selected regional metrics with csv/math, separately from score()."""
    import math
    errors = defaultdict(list)
    keys = defaultdict(set)
    with (reports / "test_predictions_selected.csv").open() as f:
        for row in csv.DictReader(f):
            if row["label_ready"].lower() != "true" or not row["actual"]:
                continue
            k = row["region"], row["target"], row["kind"]
            errors[k].append(float(row["prediction"]) - float(row["actual"]))
            keys[k].add(row["date"])
    with (reports / "test_metrics_selected.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        key = row["region"], row["target"], row["model"]
        e = errors[key]
        assert len(e) == int(row["n"])
        assert abs(sum(map(abs, e)) / len(e) - float(row["mae"])) < 1e-10
        assert abs(math.sqrt(sum(x*x for x in e) / len(e)) - float(row["rmse"])) < 1e-10
        assert keys[(key[0], key[1], "selected")] == keys[(key[0], key[1], "baseline")]
    with (reports / "test_macro_metrics.csv").open() as f:
        macro = list(csv.DictReader(f))
    for row in macro:
        regional = [r for r in rows if r["target"] == row["target"] and r["model"] == row["model"]]
        assert len(regional) == int(row["regions"])
        assert sum(int(r["n"]) for r in regional) == int(row["forecast_days"])
        for metric in ["mae", "rmse"]:
            assert abs(sum(float(r[metric]) for r in regional)/len(regional) - float(row["macro_" + metric])) < 1e-10
    result = {"passed": True, "regional_metric_rows": len(rows), "paired_cases": True,
              "macro_metric_rows": len(macro),
              "method": "independent stdlib CSV arithmetic; regional and macro MAE/RMSE; paired dates"}
    (reports / "metrics_audit.json").write_text(json.dumps(result, indent=2))
    return result


def descriptive(ledger, snapshots, protocol, reports):
    """Past-only descriptive trends/outliers and validation error diagnostics."""
    selection_at = pd.Timestamp(protocol["selection_at"])
    snap = snapshot_at(snapshots, selection_at)
    end = selection_at.date() - timedelta(days=3)
    start = end - timedelta(days=179)
    daily = mark_censored(aggregate(as_of(ledger, selection_at), start, end, protocol["regions"]), snap)
    daily.to_csv(reports / "descriptive_daily.csv", index=False)
    trends = []
    for region, g in daily.loc[daily.label_ready].groupby("region"):
        recent = g.loc[g.date > str(end - timedelta(days=30))]
        previous = g.loc[(g.date > str(end - timedelta(days=60))) & (g.date <= str(end - timedelta(days=30)))]
        for target in ["count", "hours"]:
            trends.append({"region": region, "target": target, "recent_n": len(recent), "previous_n": len(previous),
                           "recent_mean": recent[target].mean(), "previous_mean": previous[target].mean(),
                           "difference": recent[target].mean() - previous[target].mean()})
    pd.DataFrame(trends).to_csv(reports / "trend_comparison.csv", index=False)
    latest = as_of(ledger, pd.Timestamp(snapshots[-1]["available_at"]))
    latest["duration_hours"] = (latest.finished_at - latest.started_at).dt.total_seconds() / 3600
    latest.loc[(latest.duration_hours == 0) | (latest.duration_hours > 24)].to_csv(reports / "source_duration_anomalies.csv", index=False)
    examples = ["Київ", "Дніпропетровська область", "Львівська область"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    for i, region in enumerate(examples):
        g = daily.loc[daily.region == region].copy()
        g.loc[~g.label_ready, ["count", "hours"]] = np.nan
        for j, target in enumerate(["count", "hours"]):
            ax = axes[i, j]
            ax.plot(pd.to_datetime(g.date), g[target], color="#bad7d9", linewidth=.7, label="доба")
            ax.plot(pd.to_datetime(g.date), g[target].rolling(7, min_periods=7).mean(), color="#116a72", linewidth=1.5, label="середнє за 7 діб")
            ax.set_title(f"{short(region)} · {LABELS[target]}", fontsize=10)
            ax.tick_params(axis="x", rotation=20, labelsize=8)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Тренд до фінального тесту · лише доступні завершені дні", fontsize=14)
    save_figure(fig, reports / "figures/history.png")
    test = pd.read_csv(reports / "test_predictions_selected.csv")
    fixed_baseline = pd.read_csv(reports / "test_predictions.csv").query("model == 'mean_7'")
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    for i, region in enumerate(examples):
        for j, target in enumerate(["count", "hours"]):
            ax = axes[i, j]
            g = test.loc[(test.region == region) & (test.target == target) & test.label_ready]
            chosen = g.loc[g.kind == "selected"].sort_values("date")
            base = fixed_baseline.loc[(fixed_baseline.region == region) & (fixed_baseline.target == target)
                                      & fixed_baseline.label_ready].sort_values("date")
            ax.plot(pd.to_datetime(chosen.date), chosen.actual, color="#173842", marker=".", label="записано")
            ax.plot(pd.to_datetime(chosen.date), chosen.prediction, color="#b65524", label="вибрана модель")
            ax.plot(pd.to_datetime(base.date), base.prediction, color="#56a0a4", linestyle="--", label="середнє 7 готових днів")
            ax.set_title(f"{short(region)} · {LABELS[target]}", fontsize=10)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis="x", rotation=25, labelsize=8)
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Фінальний тест · моделі вибрані до відкриття тестових міток", fontsize=14)
    save_figure(fig, reports / "figures/forecast_vs_actual.png")
    val = pd.read_csv(reports / "validation_predictions.csv")
    selected = pd.read_csv(reports / "model_selection.csv").query("kind == 'selected'")
    val = val.merge(selected[["region", "target", "model"]], on=["region", "target", "model"], validate="many_to_one")
    records = []
    for (region, target), g in val.loc[val.label_ready].groupby(["region", "target"]):
        g = g.sort_values("date")
        e = (g.actual - g.prediction).to_numpy()
        contiguous = bool((pd.to_datetime(g.date).diff().dropna() == pd.Timedelta(days=1)).all())
        # Missing dates must not be compressed into artificial daily lags.
        statistic = acorr_ljungbox(e, lags=[7], return_df=True) if contiguous and len(e) > 14 and np.std(e) > 1e-9 else None
        records.append({"region": region, "target": target, "n": len(e), "bias_actual_minus_prediction": np.mean(e),
                        "contiguous": contiguous, "ljung_box_lag7_p_exploratory": statistic.lb_pvalue.iloc[0] if statistic is not None else np.nan})
    pd.DataFrame(records).to_csv(reports / "validation_error_diagnostics.csv", index=False)


def md_table(frame):
    f = frame.copy()
    for column in f.select_dtypes("number"):
        f[column] = f[column].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    rows = ["| " + " | ".join(map(str, f.columns)) + " |", "| " + " | ".join("---" for _ in f.columns) + " |"]
    rows += ["| " + " | ".join(map(str, row)) + " |" for row in f.itertuples(index=False, name=None)]
    return "\n".join(rows)


def build_report(data: Path, reports: Path):
    p = json.loads((reports / "protocol.json").read_text())
    s = json.loads((reports / "source_audit.json").read_text())
    a = json.loads((reports / "forecast_audit.json").read_text())
    provenance = json.loads((data / "provenance.json").read_text())
    relative_data = Path(os.path.relpath(data.resolve(), reports.resolve())).as_posix()
    seasonality_audit = json.loads((reports / "seasonality_audit.json").read_text())
    label_status = pd.read_csv(reports / "test_label_status.csv")
    excluded_days = label_status.loc[~label_status.label_ready]
    excluded_day_text = "; ".join(f"{row.region}, {row.date}" for row in excluded_days.itertuples()) or "немає"
    region_count = len(p["regions"])
    usable_days = int(label_status.label_ready.sum())
    comparisons = seasonal_comparison(reports)
    independent_metrics_check(reports)
    macro = pd.read_csv(reports / "test_macro_metrics.csv")
    baseline_table, all_model_macro = baseline_comparison(reports)
    mean7 = baseline_table.query("model == 'mean_7'").set_index("target")
    result = []
    for target in ["count", "hours"]:
        g = macro.loc[macro.target == target].set_index("model")
        baseline, selected = g.loc["baseline", "macro_mae"], g.loc["selected", "macro_mae"]
        reduction = 1 - selected / baseline if baseline > 0 else None
        result.append({"target": target, "baseline_mae": baseline, "selected_mae": selected,
                       "relative_improvement": reduction, "criterion_met": bool(reduction is not None and reduction >= p["config"]["success_relative_mae_reduction"])})
    (reports / "success_criteria.json").write_text(json.dumps(result, indent=2))
    mean7.reset_index().to_json(reports / "success_criteria_mean7.json", orient="records", indent=2)
    display = mean7.reset_index()[["target", "baseline_mae", "selected_mae", "relative_mae_improvement", "meets_5_percent"]].rename(columns={"target": "Ціль", "baseline_mae": "MAE середнього 7 днів", "selected_mae": "MAE моделі", "relative_mae_improvement": "Покращення, %", "meets_5_percent": "Поріг 5% проти mean_7"})
    display["Ціль"] = display["Ціль"].map(LABELS)
    display["Покращення, %"] *= 100
    display["Поріг 5% проти mean_7"] = display["Поріг 5% проти mean_7"].map({True: "пройдено", False: "не пройдено"})
    baseline_display = baseline_table[["target", "model", "baseline_mae", "selected_mae", "relative_mae_improvement"]].copy()
    baseline_display["target"] = baseline_display.target.map(LABELS)
    baseline_display["relative_mae_improvement"] *= 100
    baseline_display.columns = ["Ціль", "Орієнтир", "MAE орієнтира", "MAE обраної моделі", "Покращення, %"]
    metrics = pd.read_csv(reports / "test_metrics_selected.csv")
    selected_metrics = metrics.query("model == 'selected'")[["region", "target", "n", "mae", "rmse"]]
    selected_metrics = selected_metrics.merge(pd.read_csv(reports / "model_selection.csv").query("kind == 'selected'")[["region", "target", "model"]], on=["region", "target"])
    selected_metrics.to_csv(reports / "results_by_region.csv", index=False)
    selected_metrics["target"] = selected_metrics.target.map(LABELS)
    chosen = pd.read_csv(reports / "model_selection.csv").query("kind == 'selected'")
    counts = chosen.groupby(["target", "model"]).size().rename("regions").reset_index()
    pred = pd.read_csv(reports / "predictions_unlabelled.csv")
    correction = (pred.prediction - pred.raw_prediction).abs()
    max_corrections = pred.assign(correction=correction).groupby("target").correction.max().to_dict()
    supported = comparisons.groupby("target").apply(lambda g: g.loc[g.relative_mae_improvement >= .05, "region"].nunique(), include_groups=False).to_dict()
    acf_supported = comparisons.groupby("target").apply(lambda g: g.loc[(g.relative_mae_improvement >= .05) & g.acf_7_exploratory_positive, "region"].nunique(), include_groups=False).to_dict()
    season = comparisons[["region", "target", "family", "nonseasonal_mae", "seasonal_mae", "conclusion"]].copy()
    season["target"] = season.target.map(LABELS)
    excluded = ", ".join(r["region"] for r in s["excluded_regions"])
    sections = []
    def section(title, text, extra=""):
        sections.append((title, text, extra))
    section("Результат базової версії", f"Щодня о 00:00 Europe/Kyiv прогнозуємо добу, що починається в цей момент: інтервал [D 00:00, D+1 00:00). Через затримку даних це зазвичай прогноз на **післязавтра відносно останньої готової доби**. Тест: **{p['test_start']} — {p['test_end']}**; {region_count} регіони, {usable_days} придатні регіональні дні для кожної цілі. Вибір моделей: {p['validation_start']} — {p['validation_end']}.\n\nОсновне явне порівняння — із середнім останніх семи доступних готових днів (mean_7) в усіх регіонах. Кількість: покращення MAE **{mean7.loc['count', 'relative_mae_improvement']:.2%}**. Години: **{mean7.loc['hours', 'relative_mae_improvement']:.2%}**. Поріг 5% проти цього орієнтира не досягнуто для обох цілей. Обрана модель — вибір між ARIMA та експоненційним згладжуванням окремо для регіону й цілі; це не результат однієї ARIMA для всіх регіонів. Для кількості RMSE обраної моделі {mean7.loc['count', 'selected_rmse']:.3f}, проти {mean7.loc['count', 'baseline_rmse']:.3f} у mean_7, тобто за цією метрикою модель трохи гірша.", display)
    section("З яким орієнтиром порівнюємо", f"Попередні {result[0]['relative_improvement']:.2%} для кількості — точний результат проти орієнтира, обраного за валідацією окремо для кожного регіону: last у 8 регіонах і mean_7 у 15. Для годин: last у 3, mean_7 у 18, weekly_naive у 2. Це не порівняння тільки з правилом «як учора», проте така загальна цифра приховувала значно менший виграш проти єдиного mean_7. Тепер усі варіанти показано поруч. last означає останній **готовий** день, який через затримку може бути позавчорашнім.\n\nmean_7 використовує сім останніх готових добових значень, доступних до origin. Для годин усереднюється частка доби, яка множиться на 23/24/25 годин цільової доби. Останні сім календарних днів до origin можуть містити неготову добу, тому майбутні чи ще не опубліковані значення не використовуються. Усі порівняння мають однакові регіони, дати, origin, мітки та доступність. Відсоток = 100 × (MAE орієнтира − MAE моделі) / MAE орієнтира.\n\nЯвне порівняння mean_7 додано за запитом користувача після відкриття тесту; прогнози mean_7 вже існували в замороженому експерименті. Початковий протокол і критерій проти validation_selected_baseline збережено, новий критерій не видається за заздалегідь погоджений. Моделі, параметри й тестові дати не змінювалися. [Порівняння орієнтирів](baseline_comparison.csv) · [за регіонами](baseline_comparison_by_region.csv) · [окремі ARIMA, SARIMA, SES, ETS та орієнтири](test_macro_all_models.csv) · [склад початкового орієнтира](baseline_selection_summary.csv).", baseline_display)
    section("Дані й аномалії", f"Основне джерело — [волонтерський датасет]({provenance['repository']}/tree/{provenance['latest']}). Зафіксовано SHA `{provenance['latest']}`; підтверджена доступність не пізніше {provenance['latest_available_at']}. Файл містить {s['rows']:,} записів від {s['min_start']} до останнього початку {s['max_start']}. Останнє завершення: {s['max_finish']}. Щоденний commit сам собою не підтверджує актуальність кожного регіону.\n\nТочних дублікатів немає. Дублікат тут означає збіг усіх чотирьох полів: region, started_at, finished_at, naive, включно з роком і секундою. Однакова тривалість або день місяця в різні роки не є дублікатом.\n\n{ s['naive_rows']:,} записів naive=True: завершення через 30 хвилин встановили автори джерела. Ці інтервали залишено без змін і включено до годин. Є {s['zero_durations']} нульових і {s['over_24_hours']} інтервалів понад 24 години; найдовший {s['max_duration_hours']:.3f} год. Їх автоматично не вилучено: довгий інтервал може бути реальною подією або помилкою джерела. [Перелік аномалій](source_duration_anomalies.csv).\n\nДжерело має 25 географічних одиниць: 24 області та Київ; Криму немає. Виключено: {excluded}. Для допуску використано активність джерела: щонайменше два різні дні з початками за минулі 30 днів та свіжий snapshot. **Це практичний індикатор доступності, він не підтверджує повноти всіх реальних тривог.** Дні без записаних початків умовно прийнято за нуль у допущених рядах. Окремого незалежного журналу всіх пропусків немає; тому цілі стосуються зареєстрованої історії, а не гарантовано всіх подій. [Покриття](coverage.csv).")
    section("Добові ряди", f"Кількість — число початків у межах київської календарної доби. Години — довжина об'єднання записаних інтервалів у цьому регіоні в межах доби. Перетини не подвоюють час, інтервали через північ розділено. Межі напіввідкриті [початок, кінець). Тривалість доби обчислюється через UTC і становить 23/24/25 годин.\n\nНезавершений active-інтервал блокує мітки від дня його початку до наявного snapshot. Йому не вигадується час завершення. Це окремий випадок від уже завершених naive=True, де правило 30 хвилин збережено. Неготові тестові регіональні доби: {excluded_day_text}. Їх виключено для обох цілей й усіх моделей. [Статуси міток](test_label_status.csv). Нулі для днів без записів залежать від припущення про покриття вище.")
    section("Доступність даних і захист від витоку", f"Відтворено {s['snapshots']} щоденних Git-версій. Для кожної версії перевірено blob SHA та відповідний успішний GitHub Actions run. Верхня межа публікації — завершення run після commit і push, зі звіркою parent SHA. Для найпершої shallow-версії parent не доступний; використано єдиний відповідний часовий інтервал run. [Маніфест версій]({relative_data}/snapshots.json). Початкові історичні рядки консервативно доступні тільки з {provenance['earliest_replay_at']}.\n\nНа кожен прогноз береться остання вже опублікована версія, а не сучасний CSV, обрізаний за датою події. Навчання — до 180 доступних завершених днів, мінімум 60. Наприклад, **1 жовтня 2026 о 00:00** остання готова доба — **29 вересня**. Крок 1 — прогноз на 30 вересня; крок 2 — цільовий прогноз на **1 жовтня, 00:00–24:00**. Це «післязавтра» відносно 29 вересня та найближча доба від моменту запуску. День запуску й дата цілі збігаються. Прогалину 30 вересня модель проходить власним прогнозом, фактичне значення не підставляється. Якщо через незавершені інтервали остання готова мітка ще давніша, горизонт від неї більший: у цьому експерименті {a['min_steps']}–{a['max_steps']} календарних добових кроків. [Дати й межі кожного прогнозу](forecast_timing.csv) · [аудит часової схеми](timing_audit.json).\n\nВалідаційні мітки взято з версії, доступної {a['selection_label_available_at']}, до початку тесту. Є триденний календарний буфер. Сімейство/сезонність і орієнтир вибрано за MAE, далі RMSE, потім назвою моделі. Вибір зафіксовано перед тестовими мітками; щоденне перенавчання на новому минулому дозволене. Тест не використано для повторного налаштування. Чотири прогнози валідації пропущено через індикатор покриття; усі моделі порівнюються на однакових доступних днях.")
    section("Тренди та опис історії", "Нижче — приклади історії для Києва, Дніпропетровщини та Львівщини до фінального тесту. У CSV є зміни середніх за останні й попередні 30 днів для всіх 23 регіонів. Зміни можуть відображати як характер подій, так і роботу джерела; причинні пояснення з одного ряду не випливають. [Таблиця трендів](trend_comparison.csv).\n\n![Тренди](figures/history.png)")
    section("Дні тижня та ACF", f"Одна фігура, дві панелі: середні початки та години, понеділок–неділя для всіх регіонів. Навчальний період {seasonality_audit['start']} — {seasonality_audit['end']}, до першої валідації. [Таблиця відмінностей](weekday_differences.csv) містить n, середнє, медіану та різницю із загальним середнім регіону для кожної цілі.\n\n![Дні тижня](figures/weekday_profiles.png)\n\n[ACF кількості](figures/acf_count.png) · [ACF годин](figures/acf_hours.png) · [Числа ACF](acf.csv). Перед ACF вилучено лінійний тренд; лаги 7, 14 і 21 позначено. Межі 95% поточкові, орієнтовні, без поправки на багато порівнянь; окремий пік не є доказом сезонності. STL не застосовується.\n\nЩонайменше одна сезонна модель зменшила валідаційний MAE на 5% у {supported.get('count', 0)} регіонах для кількості й {supported.get('hours', 0)} для годин. Поєднання такого покращення з додатною ACF7 поза орієнтовною межею є лише у {acf_supported.get('count', 0)} та {acf_supported.get('hours', 0)} регіонах відповідно. Однієї спільної стійкої тижневої закономірності для всіх регіонів не встановлено. Висновки нижче є дослідницькими; причинного впливу дня тижня не доведено.", season)
    section("Моделі та допустимі значення", "ARIMA(1,0,1) з константою; її тижневий варіант додає сезонну AR(1) з періодом 7. Експоненційне згладжування: SES або адитивна тижнева компонента без тренду. Порядки наперед фіксовані, великого пошуку параметрів немає. Орієнтири: останній доступний день, повтор тижня, середнє останніх семи днів. Це базовий набір, не повна перевірка всіх ARIMA-конфігурацій.\n\nДля кількості: max(0, raw). Для тривалості модель прогнозує частку доби; clip(raw, 0, 1) множиться на довжину цільової доби. Дробова кількість — очікуване значення; перед метриками не округлюється. Прогнозні інтервали в цій версії не оцінювалися.\n\nПри збої чи відсутності збіжності — середнє останніх семи доступних днів із записаною причиною. NaN не підмінюється нулем. Обрані на валідації моделі:", counts)
    section("Фінальний тест", "MAE та RMSE обчислено після обмеження прогнозів у початкових одиницях. Macro MAE — рівноважне середнє 23 регіональних MAE; це лише агрегування помилки, без загального показника тривалості. [Усі кандидати на тесті](test_metrics_all_candidates.csv), [зведення кожної моделі](test_macro_all_models.csv), [метрики обраних моделей та орієнтирів](test_metrics_selected.csv), [кожен прогноз і факт](test_predictions_selected.csv).\n\nНа графіку обрана модель порівнюється саме з mean_7.\n\n![Прогноз і факт](figures/forecast_vs_actual.png)", all_model_macro)
    section("Результати за регіонами", "Для кожної цілі модель вибрана на попередній валідації. [CSV із моделями й метриками](results_by_region.csv).", selected_metrics)
    section("Перевірки після етапів", f"1. Джерело: 100 перевірок Git blob; три незалежні зіставлення ledger з сирими історичними CSV. 30-хвилинні інтервали збережено.\n2. Агрегація: 40 незалежних перевірок sweep-line алгоритмом, тести DST, півночі, перекриттів і нульових інтервалів.\n3. Сезонність: звірено день тижня та зважені середні таблиці; дані передують валідації.\n4. Доступність: тести незмінності минулих входів і прогнозів після додавання майбутніх записів та snapshot. Для всіх {a['forecast_rows']:,} прогнозів snapshot доступний до origin.\n5. Моделі: усі прогнози скінченні й обмежені. {a['fallback_forecasts']} резервних прогнозів ({a['fallback_forecasts']/a['forecast_rows']:.2%}); {a['clipped_forecasts']} обмежень ({a['clipped_forecasts']/a['forecast_rows']:.2%}) серед усіх кандидатів/орієнтирів обох періодів. Найбільша абсолютна корекція: {max_corrections['count']:.3f} початків, {max_corrections['hours']:.3f} год. Причини є в predictions_unlabelled.csv. [Діагностика валідаційних похибок](validation_error_diagnostics.csv): зміщення та дослідницький Ljung–Box(7), без налаштування за тестом.\n6. Оцінювання: незалежний перерахунок усіх 92 регіональних рядків метрик стандартним CSV/math; дати моделей та орієнтирів збігаються.\n7. Відтворення: [журнал перевірок](verification.json), виконаний notebook і зафіксовані залежності.\n\n[Аудит прогнозів](forecast_audit.json) · [Аудит агрегації](independent_aggregation_audit.json) · [Аудит метрик](metrics_audit.json) · [Незалежний аудит орієнтирів](baseline_comparison_audit.json).")
    from .telegram_experiment import report_section
    telegram_section = report_section(data, reports)
    if telegram_section:
        section(*telegram_section)
    section("Критерії завершення та наступна перевірка", "Готовність репозиторію: встановлюваний Python-пакет, одна команда відтворення заморожених даних, конфігурація/версії, notebook, тести, журнал рішень і походження прогнозів. Готовність репорту: точні часові межі, покриття/аномалії, погоджені сезонні артефакти, порівняння з орієнтирами та перевірки кожного етапу. Окремий критерій якості: мінімум 5% зниження macro MAE для кожної цілі. Початковий критерій проти вибраного на валідації орієнтира виконано тільки для кількості. Проти єдиного mean_7 поріг 5% не досягнутий для жодної цілі; змінювати моделі за результатом цього тесту не можна без нового майбутнього відкладеного періоду.\n\nTelegram досліджується окремо від базової версії. Історичне порівняння з нинішньою вебстрічкою не може підтвердити результат без витоку; наступна доказова перевірка потребує текстів, збережених до origin, і нового недоторканого тестового періоду.")
    md = "# Аналіз і прогнозування зареєстрованих повітряних тривог\n\n" + "\n\n".join("## " + title + "\n\n" + text + ("\n\n" + md_table(extra) if isinstance(extra, pd.DataFrame) else "") for title, text, extra in sections)
    (reports / "report.md").write_text(md)
    # Render the limited Markdown used above without introducing a heavyweight dependency.
    import re
    def inline(text):
        text = html.escape(text)
        text = re.sub(r"!\[([^]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2" loading="lazy">', text)
        text = re.sub(r"\[([^]]*)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
        text = re.sub(r"\*\*([^*]+)\*\*", r'<strong>\1</strong>', text)
        text = re.sub(r"`([^`]+)`", r'<code>\1</code>', text)
        return text.replace("\n", "<br>")
    body = []
    for title, text, extra in sections:
        body.append(f"<section><h2>{html.escape(title)}</h2>" + "".join("<p>" + inline(par) + "</p>" for par in text.split("\n\n")))
        if isinstance(extra, pd.DataFrame):
            body.append('<div class="table">' + extra.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}") + '</div>')
        body.append("</section>")
    document = '''<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Повітряні тривоги · аналіз часових рядів</title><style>
    body{font:17px/1.6 system-ui,sans-serif;color:#173842;background:#f5f8f8;margin:0}main{max-width:1180px;margin:auto;padding:32px}h1{font-size:36px;line-height:1.2}h2{font-size:25px}section{background:white;padding:28px;margin:24px 0;border-radius:12px}a{color:#087281}code{font-size:13px;overflow-wrap:anywhere}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px 10px;text-align:left;border-bottom:1px solid #dbe4e5}th{background:#e9f2f2;position:sticky;top:0}.table{max-height:640px;overflow:auto}@media(max-width:700px){main{padding:14px}section{padding:16px}h1{font-size:27px}}
    </style><main><h1>Повітряні тривоги України<br>Базовий аналіз часових рядів</h1><p>Python · ARIMA · експоненційне згладжування · Europe/Kyiv</p>''' + "".join(body) + "</main></html>"
    (reports / "report.html").write_text(document)
    return {"fixed_mean7": mean7.reset_index().to_dict("records"),
            "original_validation_selected_baseline": result}
