"""Training-only weekday profiles and detrended ACF; no STL or text features."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airraid-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from statsmodels.tsa.stattools import acf

from .series import TZ, aggregate, mark_censored, midnight
from .sources import as_of, snapshot_at

WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
LABELS = {"count": "Кількість початків", "hours": "Години під тривогою"}


def short(region):
    return region.replace(" область", "")


def save_figure(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def source_audit(ledger, snapshots, protocol, output):
    latest = snapshots[-1]
    current = as_of(ledger, pd.Timestamp(latest["available_at"]))
    duration = (current.finished_at - current.started_at).dt.total_seconds() / 60
    ids = [s["last_processed_message_id"] for s in snapshots]
    checks = {"rows": len(current), "duplicate_rows": int(current.duplicated(["region", "started_at", "finished_at", "naive"]).sum()),
              "missing_timestamps": int(current[["started_at", "finished_at"]].isna().sum().sum()),
              "negative_durations": int((duration < 0).sum()), "zero_durations": int((duration == 0).sum()),
              "over_24_hours": int((duration > 1440).sum()), "max_duration_hours": float(duration.max() / 60),
              "naive_rows": int(current.naive.sum()), "naive_30_minutes_preserved": bool((duration[current.naive] == 30).all()),
              "finished_before_publication": bool((current.finished_at <= current.available_at).all()),
              "source_message_ids_nondecreasing": bool(all(b >= a for a, b in zip(ids, ids[1:]))),
              "snapshots": len(snapshots), "min_start": current.started_at.min().isoformat(),
              "max_start": current.started_at.max().isoformat(), "max_finish": current.finished_at.max().isoformat(),
              "selected_regions": len(protocol["regions"]),
              "coverage_policy": "Activity proxy at model selection; >=2 distinct start days in preceding 30 days. This does not prove complete real-world coverage.",
              "excluded_regions": [r for r in protocol["coverage"] if not r["eligible"]],
              "ongoing_latest": latest["active"]}
    assert checks["duplicate_rows"] == 0 and checks["missing_timestamps"] == 0 and checks["negative_durations"] == 0
    assert checks["naive_30_minutes_preserved"] and checks["finished_before_publication"] and checks["source_message_ids_nondecreasing"]
    (output / "source_audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
    pd.DataFrame(protocol["coverage"]).to_csv(output / "coverage.csv", index=False)
    return checks


def seasonality(ledger, snapshots, protocol, output):
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    origin = midnight(protocol["validation_start"])
    snap = snapshot_at(snapshots, origin)
    end = pd.Timestamp(snap["committed_at"]).tz_convert(TZ).date() - timedelta(days=1)
    start = end - timedelta(days=protocol["config"]["train_days"] - 1)
    daily = mark_censored(aggregate(as_of(ledger, origin), start, end, protocol["regions"]), snap)
    daily.to_csv(output / "seasonality_training_daily.csv", index=False)
    ready = daily.loc[daily.label_ready].copy()
    long = ready.melt(id_vars=["region", "date", "weekday"], value_vars=["count", "hours"],
                      var_name="target", value_name="value")
    table = long.groupby(["region", "target", "weekday"]).agg(
        n=("value", "size"), mean=("value", "mean"), median=("value", "median")).reset_index()
    overall = long.groupby(["region", "target"]).value.mean().rename("region_mean").reset_index()
    table = table.merge(overall, on=["region", "target"], validate="many_to_one")
    table["difference_from_region_mean"] = table["mean"] - table.region_mean
    table["weekday_name"] = table.weekday.map(dict(enumerate(WEEKDAYS)))
    table.to_csv(output / "weekday_differences.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), constrained_layout=True)
    for ax, target in zip(axes, ["count", "hours"]):
        pivot = table.loc[table.target == target].pivot(index="region", columns="weekday", values="mean").reindex(protocol["regions"])
        picture = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(7), WEEKDAYS)
        ax.set_yticks(range(len(pivot)), [short(r) for r in pivot.index], fontsize=9)
        ax.set_title(LABELS[target])
        for i in range(len(pivot)):
            for j in range(7):
                v = pivot.iloc[i, j]
                color = "white" if v > np.nanmax(pivot.to_numpy()) * .62 else "#142e3b"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(picture, ax=ax, shrink=.65, label="початків / день" if target == "count" else "годин / день")
    fig.suptitle(f"Дні тижня: навчальні дані {start} — {end}\nОкремі шкали для кількості та годин", fontsize=14)
    save_figure(fig, figures / "weekday_profiles.png")
    acf_rows = []
    for target in ["count", "hours"]:
        fig, axes = plt.subplots(6, 4, figsize=(15, 17), constrained_layout=True)
        for ax, region in zip(axes.flat, protocol["regions"]):
            g = ready.loc[ready.region == region].sort_values("date")
            if len(g) < 35 or not (pd.to_datetime(g.date).diff().dropna() == pd.Timedelta(days=1)).all():
                ax.set_title(short(region) + " — недостатньо даних", fontsize=9)
                continue
            y = detrend(g[target].to_numpy(dtype=float), type="linear")
            if np.std(y) < 1e-12:
                vals, ci = np.zeros(29), np.zeros((29, 2))
                vals[0] = 1
            else:
                vals, ci = acf(y, nlags=28, alpha=.05, fft=True, result_object=False)
            ax.vlines(np.arange(1, len(vals)), 0, vals[1:], color="#176b78", linewidth=1.4)
            ax.fill_between(np.arange(1, len(vals)), ci[1:, 0] - vals[1:], ci[1:, 1] - vals[1:], color="#176b78", alpha=.13)
            for lag in [7, 14, 21]:
                ax.axvline(lag, color="#c27038", alpha=.35, linestyle="--", linewidth=.8)
            ax.axhline(0, color="gray", linewidth=.6)
            ax.set_xticks([1, 7, 14, 21, 28])
            ax.set_ylim(-.5, 1)
            ax.set_title(short(region), fontsize=9)
            for lag in range(1, len(vals)):
                acf_rows.append({"region": region, "target": target, "lag": lag, "acf": vals[lag],
                                 "ci_low": ci[lag, 0], "ci_high": ci[lag, 1], "n": len(g)})
        for ax in list(axes.flat)[len(protocol["regions"]):]:
            ax.axis("off")
        fig.suptitle(f"ACF після вилучення лінійного тренду: {LABELS[target]}\nНавчальний період; межі 95% є орієнтовними й поточковими", fontsize=14)
        save_figure(fig, figures / f"acf_{target}.png")
    pd.DataFrame(acf_rows).to_csv(output / "acf.csv", index=False)
    audit = {"analysis_origin": origin.isoformat(), "source_available_at": snap["available_at"],
             "snapshot_sha": snap["sha"], "start": str(start), "end": str(end),
             "only_pre_validation_data": bool(end < origin.date()), "training_rows": len(ready),
             "censored_rows": int((~daily.label_ready).sum()),
             "weekday_mapping_checked": bool((pd.to_datetime(ready.date).dt.dayofweek == ready.weekday).all()),
             "table_mean_reconciled": bool(np.allclose(overall.region_mean, table.groupby(["region", "target"]).apply(
                 lambda g: np.average(g["mean"], weights=g.n), include_groups=False).to_numpy())),
             "stl_used": False}
    assert audit["only_pre_validation_data"] and audit["weekday_mapping_checked"] and audit["table_mean_reconciled"]
    (output / "seasonality_audit.json").write_text(json.dumps(audit, indent=2))
    return daily


def seasonal_comparison(output):
    scores = pd.read_csv(output / "validation_metrics.csv")
    acfs = pd.read_csv(output / "acf.csv")
    records = []
    for (region, target), g in scores.groupby(["region", "target"]):
        for family, regular, seasonal in [("ARIMA", "arima", "sarima_7"), ("Exponential smoothing", "ses", "ets_7")]:
            a, b = g.loc[g.model == regular].iloc[0], g.loc[g.model == seasonal].iloc[0]
            delta = float(a.mae - b.mae)
            lag = acfs.loc[(acfs.region == region) & (acfs.target == target) & (acfs.lag == 7)]
            acf7 = float(lag.acf.iloc[0]) if len(lag) else np.nan
            supported = bool(len(lag) and lag.ci_low.iloc[0] > 0)
            ratio = delta / a.mae if a.mae > 0 else np.nan
            records.append({"region": region, "target": target, "family": family,
                            "nonseasonal_mae": a.mae, "seasonal_mae": b.mae,
                            "nonseasonal_rmse": a.rmse, "seasonal_rmse": b.rmse,
                            "mae_improvement": delta, "relative_mae_improvement": ratio,
                            "acf_7": acf7, "acf_7_exploratory_positive": supported,
                            "conclusion": "ознаки сезонності підтримані валідацією" if supported and ratio >= .05
                            else "сезонність покращує прогноз, ACF не дає переконливого підтвердження" if ratio >= .05
                            else "переконливої користі сезонності не виявлено"})
    result = pd.DataFrame(records)
    result.to_csv(output / "seasonal_comparison.csv", index=False)
    return result
