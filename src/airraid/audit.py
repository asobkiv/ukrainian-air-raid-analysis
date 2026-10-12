"""Independent arithmetic checks against the frozen raw CSV and source Git history."""
from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .series import aggregate, midnight
from .sources import CSV_PATH, as_of, git, load_ledger


def independent_aggregation_check(data: Path, reports: Path) -> dict:
    """Sweep line in stdlib datetime, separate from the production interval-merging code."""
    with gzip.open(data / "volunteer_data_uk.csv.gz", "rt") as f:
        raw = list(csv.DictReader(f))
    ledger = load_ledger(data / "event_ledger.csv.gz")
    snapshots = json.loads((data / "snapshots.json").read_text())
    current = as_of(ledger, pd.Timestamp(snapshots[-1]["available_at"]))
    p = json.loads((reports / "protocol.json").read_text())
    zones = ZoneInfo("Europe/Kyiv")
    candidates = [(r, d.date()) for r in p["regions"] for d in pd.date_range(p["validation_start"], p["test_end"])]
    selected = np.random.default_rng(p["config"]["seed"]).choice(len(candidates), size=40, replace=False)
    cases = []
    for index in selected:
        region, day = candidates[index]
        begin = datetime.combine(day, datetime.min.time(), tzinfo=zones).astimezone(timezone.utc)
        finish = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=zones).astimezone(timezone.utc)
        points, count = [], 0
        for row in raw:
            if row["region"] != region:
                continue
            s, e = datetime.fromisoformat(row["started_at"]), datetime.fromisoformat(row["finished_at"])
            count += int(begin <= s < finish)
            s, e = max(s, begin), min(e, finish)
            if s < e:
                points.extend([(s, 1), (e, -1)])
        balance, seconds, previous = 0, 0.0, None
        for t, delta in sorted(points):
            if previous is not None and balance > 0:
                seconds += (t - previous).total_seconds()
            balance += delta
            previous = t
        actual = aggregate(current, day, day, [region]).iloc[0]
        if count != actual["count"] or abs(seconds / 3600 - actual.hours) > 1e-9:
            raise AssertionError((region, day, count, seconds / 3600, actual.to_dict()))
        cases.append({"region": region, "date": str(day), "count": count, "hours": seconds / 3600})
    summary = {"passed": True, "cases": len(cases), "method": "stdlib CSV + independent sweep-line interval union", "checked": cases}
    (reports / "independent_aggregation_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def git_replay_check(repo: Path, data: Path, reports: Path) -> dict:
    ledger = load_ledger(data / "event_ledger.csv.gz")
    snapshots = json.loads((data / "snapshots.json").read_text())
    records = []
    for index in [0, len(snapshots) // 2, len(snapshots) - 1]:
        snap = snapshots[index]
        actual = list(csv.DictReader(io.StringIO(git(repo, "show", f"{snap['sha']}:{CSV_PATH}").decode())))
        replay = as_of(ledger, pd.Timestamp(snap["available_at"]))
        canonical_actual = Counter((r["region"], pd.Timestamp(r["started_at"]), pd.Timestamp(r["finished_at"]), r["naive"].lower() == "true") for r in actual)
        canonical_replay = Counter((r.region, r.started_at, r.finished_at, r.naive) for r in replay.itertuples())
        if canonical_actual != canonical_replay:
            raise AssertionError("Replay differs from actual Git snapshot")
        records.append({"sha": snap["sha"], "rows": len(actual), "passed": True})
    result = {"passed": True, "snapshots": records}
    (reports / "git_replay_audit.json").write_text(json.dumps(result, indent=2))
    return result
