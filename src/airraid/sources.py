"""Build a replayable ledger from public Git snapshots, never executing pickle code."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import pickletools
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CSV_PATH = "datasets/volunteer_data_uk.csv"
ACTIVE_PATH = "processors/pkl/volunteer_active_alerts.pkl"
REPOSITORY = "https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset"


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def decode_active(data: bytes) -> dict:
    """Interpret a narrow pickle grammar as data; no imports, eval or pickle.loads.

    Only the author's alert record and UTC datetime representations are supported.
    Unexpected constructors/opcodes fail closed.
    """
    stack, memo = [], {}
    mark = object()
    for op, arg, _ in pickletools.genops(data):
        name = op.name
        if name in {"PROTO", "FRAME"}:
            continue
        if name in {"SHORT_BINUNICODE", "BINUNICODE", "SHORT_BINBYTES", "BINBYTES", "BININT", "BININT1", "BININT2"}:
            stack.append(arg)
        elif name == "NONE":
            stack.append(None)
        elif name in {"NEWTRUE", "NEWFALSE"}:
            stack.append(name == "NEWTRUE")
        elif name == "EMPTY_DICT":
            stack.append({})
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "MARK":
            stack.append(mark)
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name in {"BINGET", "LONG_BINGET"}:
            stack.append(memo[arg])
        elif name == "STACK_GLOBAL":
            symbol = stack.pop()
            module = stack.pop()
            allowed = {("processors.tg_dataclasses", "ETryvogaChannelAlert"),
                       ("datetime", "datetime"), ("datetime", "timezone"), ("datetime", "timedelta")}
            if (module, symbol) not in allowed:
                raise ValueError(f"Unsupported pickle symbol: {module}.{symbol}")
            stack.append((module, symbol))
        elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
            n = int(name[-1])
            values = tuple(stack[-n:])
            del stack[-n:]
            stack.append(values)
        elif name == "REDUCE":
            args, symbol = stack.pop(), stack.pop()
            if symbol == ("datetime", "timedelta") and args == (0, 0, 0):
                value = "zero_offset"
            elif symbol == ("datetime", "timezone") and args == ("zero_offset",):
                value = "UTC"
            elif symbol == ("datetime", "datetime") and len(args) == 2 and args[1] == "UTC":
                b = args[0]
                if not isinstance(b, bytes) or len(b) != 10:
                    raise ValueError("Unexpected datetime encoding")
                value = datetime(int.from_bytes(b[:2], "big"), b[2] & 127, b[3], b[4], b[5], b[6],
                                 int.from_bytes(b[7:], "big"), tzinfo=timezone.utc).isoformat()
            else:
                raise ValueError(f"Unsupported pickle reduction: {symbol}")
            stack.append(value)
        elif name == "NEWOBJ":
            args, symbol = stack.pop(), stack.pop()
            if symbol != ("processors.tg_dataclasses", "ETryvogaChannelAlert") or args != ():
                raise ValueError("Unsupported pickle object")
            stack.append({})
        elif name == "SETITEMS":
            i = next(i for i in range(len(stack) - 1, -1, -1) if stack[i] is mark)
            values = stack[i + 1:]
            del stack[i:]
            stack[-1].update(zip(values[::2], values[1::2]))
        elif name == "SETITEM":
            value, key = stack.pop(), stack.pop()
            stack[-1][key] = value
        elif name == "BUILD":
            state = stack.pop()
            stack[-1].update(state)
        elif name == "STOP":
            if len(stack) != 1 or not isinstance(stack[0], dict):
                raise ValueError("Unexpected pickle result")
            return stack[0]
        else:
            raise ValueError(f"Unsupported pickle opcode: {name}")
    raise ValueError("Incomplete pickle")


def prepare_bundle(repo: Path, runs_file: Path, output: Path) -> dict:
    """Replay snapshots and record publication upper bounds from successful Actions runs."""
    output.mkdir(parents=True, exist_ok=True)
    runs = json.loads(runs_file.read_text())["workflow_runs"]
    commits = git(repo, "log", "--first-parent", "--reverse", "--format=%H|%cI|%P").decode().splitlines()
    live, versions, manifest, checks = {}, [], [], []
    previous = set()
    header = None
    latest_bytes = None
    for line in commits:
        sha, committed, parents = line.split("|", 2)
        t = pd.Timestamp(committed)
        matching = [r for r in runs if r["conclusion"] == "success"
                    and pd.Timestamp(r["run_started_at"]) <= t <= pd.Timestamp(r["updated_at"])
                    and (not parents or r["head_sha"] == parents.split()[0])]
        if len(matching) != 1:
            raise ValueError(f"Cannot prove a publication upper bound for {sha}: {len(matching)} matching runs")
        run = matching[0]
        available = run["updated_at"]
        raw = git(repo, "show", f"{sha}:{CSV_PATH}")
        blob_sha = git(repo, "rev-parse", f"{sha}:{CSV_PATH}").decode().strip()
        calculated = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if calculated != blob_sha:
            raise ValueError("Git blob checksum mismatch")
        lines = raw.decode().splitlines()
        if header is None:
            header = lines[0]
        if lines[0] != header:
            raise ValueError("CSV schema changed")
        current = set(lines[1:])
        if len(current) != len(lines) - 1:
            raise ValueError("Exact duplicate rows require explicit policy")
        for row in previous - current:
            record = live.pop(row)
            record["unavailable_at"] = available
            versions.append(record)
        for row in current - previous:
            parsed = next(csv.DictReader(io.StringIO(header + "\n" + row)))
            live[row] = {**parsed, "available_at": available, "unavailable_at": "", "first_seen_commit": sha}
        active = decode_active(git(repo, "show", f"{sha}:{ACTIVE_PATH}"))
        last_id = int(git(repo, "show", f"{sha}:processors/pkl/volunteer_last_processed_id.txt"))
        manifest.append({"sha": sha, "committed_at": committed, "available_at": available,
                         "run_url": run["html_url"], "csv_blob_sha": blob_sha,
                         "csv_sha256": hashlib.sha256(raw).hexdigest(), "csv_rows": len(current),
                         "last_processed_message_id": last_id, "active": active})
        checks.append({"sha": sha, "added": len(current - previous), "removed": len(previous - current),
                       "rows": len(current), "blob_verified": True})
        previous, latest_bytes = current, raw
    versions.extend(live.values())
    ledger = pd.DataFrame(versions).sort_values(["started_at", "region", "available_at"])
    ledger.to_csv(output / "event_ledger.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    with gzip.GzipFile(output / "volunteer_data_uk.csv.gz", "wb", mtime=0) as f:
        f.write(latest_bytes)
    (output / "snapshots.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (output / "history_checks.json").write_text(json.dumps(checks, indent=2))
    (output / "workflow_runs.json").write_bytes(runs_file.read_bytes())
    summary = {"repository": REPOSITORY, "snapshot_count": len(manifest), "latest": manifest[-1]["sha"],
               "latest_available_at": manifest[-1]["available_at"], "latest_rows": manifest[-1]["csv_rows"],
               "ledger_versions": len(ledger), "earliest_replay_at": manifest[0]["available_at"],
               "publication_policy": "successful GitHub Actions run completion after commit-and-push; exact parent head SHA match except shallow root",
               "note": "Base rows are conservatively available only at the earliest bundled snapshot; earlier forecasts are unsupported."}
    (output / "provenance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_checksums(output)
    return summary


def write_checksums(data: Path):
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.iterdir())
                if p.is_file() and p.name != "checksums.json"}
    (data / "checksums.json").write_text(json.dumps(manifest, indent=2))


def verify_checksums(data: Path):
    manifest = json.loads((data / "checksums.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen data checksum mismatch: {name}")
    return len(manifest)


def load_ledger(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["started_at", "finished_at", "available_at", "unavailable_at"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    if df.naive.dtype != bool:
        df["naive"] = df.naive.astype(str).str.lower().map({"true": True, "false": False})
    if df.naive.isna().any():
        raise ValueError("Invalid naive flag")
    return df


def as_of(ledger: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    if origin.tzinfo is None:
        raise ValueError("Forecast origin must be timezone-aware")
    available = pd.to_datetime(ledger.available_at, utc=True)
    unavailable = pd.to_datetime(ledger.unavailable_at, utc=True)
    return ledger.loc[(available <= origin)
                      & (unavailable.isna() | (unavailable > origin))].copy()


def snapshot_at(snapshots: list, origin: pd.Timestamp, max_age_hours=48) -> dict:
    eligible = [s for s in snapshots if pd.Timestamp(s["available_at"]) <= origin]
    if not eligible:
        raise ValueError("No snapshot available before forecast origin")
    result = eligible[-1]
    if (origin - pd.Timestamp(result["available_at"])).total_seconds() > max_age_hours * 3600:
        raise ValueError("Available source snapshot is stale")
    return result
