"""Calendar-day aggregation with interval union and explicit censoring."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

TZ = "Europe/Kyiv"


def midnight(day: str | date) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(TZ)


def day_hours(day: str | date) -> float:
    start = midnight(day)
    end = midnight(start.date() + timedelta(days=1))
    return (end.tz_convert("UTC") - start.tz_convert("UTC")).total_seconds() / 3600


def merged_intervals(intervals):
    """Merge half-open intervals [start, end), including adjacent boundaries."""
    result = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Negative alert duration")
        if start == end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def aggregate(events: pd.DataFrame, start: date, end: date, regions: list[str]) -> pd.DataFrame:
    if end < start:
        raise ValueError("Invalid aggregation range")
    lower, upper = midnight(start).tz_convert("UTC"), midnight(end + timedelta(days=1)).tz_convert("UTC")
    d = events.loc[(events.started_at < upper) & (events.finished_at >= lower)
                   & events.region.isin(regions)].copy()
    if d[["started_at", "finished_at"]].isna().any().any():
        raise ValueError("Missing event timestamp")
    if (d.finished_at < d.started_at).any():
        raise ValueError("Negative alert duration")
    counts, naive_counts, intervals = defaultdict(int), defaultdict(int), defaultdict(list)
    for r in d.itertuples(index=False):
        local_day = r.started_at.tz_convert(TZ).date()
        if start <= local_day <= end:
            counts[(r.region, local_day)] += 1
            naive_counts[(r.region, local_day)] += int(r.naive)
        intervals[r.region].append((max(lower, r.started_at), min(upper, r.finished_at)))
    hours = defaultdict(float)
    for region, values in intervals.items():
        for s, e in merged_intervals(values):
            while s < e:
                local_day = s.tz_convert(TZ).date()
                next_midnight = midnight(local_day + timedelta(days=1)).tz_convert("UTC")
                stop = min(e, next_midnight)
                hours[(region, local_day)] += (stop - s).total_seconds() / 3600
                s = stop
    records = []
    for region in regions:
        for stamp in pd.date_range(start, end):
            day = stamp.date()
            length = day_hours(day)
            h = hours[(region, day)]
            if h < -1e-9 or h > length + 1e-9:
                raise AssertionError("Invalid aggregate hours")
            records.append({"region": region, "date": str(day), "weekday": day.weekday(),
                            "count": counts[(region, day)], "hours": h, "share": h / length,
                            "day_hours": length, "naive_starts": naive_counts[(region, day)]})
    return pd.DataFrame(records)


def mark_censored(daily: pd.DataFrame, snapshot: dict) -> pd.DataFrame:
    """Exclude unresolved active starts/durations; never assume their end or persistence.

    Completed naive=True intervals remain exactly as supplied by the source.
    """
    d = daily.copy()
    d["label_ready"] = True
    d["label_reason"] = ""
    for region, alert in snapshot["active"].items():
        started_day = pd.Timestamp(alert["started_at"]).tz_convert(TZ).date().isoformat()
        mask = (d.region == region) & (d.date >= started_day)
        d.loc[mask, "label_ready"] = False
        d.loc[mask, "label_reason"] = "unresolved_active_interval"
    cutoff = pd.Timestamp(snapshot["committed_at"]).tz_convert(TZ).date().isoformat()
    incomplete = d.date >= cutoff
    d.loc[incomplete, "label_ready"] = False
    d.loc[incomplete, "label_reason"] = "incomplete_local_day"
    return d


def coverage_at(events: pd.DataFrame, origin: pd.Timestamp, config: dict) -> pd.DataFrame:
    """A source-coverage proxy, not an assertion that all real-world events are captured."""
    end = origin.tz_convert(TZ).date()
    lower = origin - pd.Timedelta(days=config["coverage_lookback_days"])
    result = []
    for region, g in events.groupby("region"):
        recent = g.loc[g.started_at >= lower]
        n = recent.started_at.dt.tz_convert(TZ).dt.date.nunique()
        ready = n >= config["coverage_min_event_days"] and region != "Луганська область"
        result.append({"region": region, "eligible": bool(ready), "recent_event_days": int(n),
                       "last_recorded_start": g.started_at.max().isoformat(),
                       "reason": "recent_source_activity_proxy" if ready else "insufficient_recent_source_activity",
                       "assessed_at": origin.isoformat()})
    return pd.DataFrame(result)
