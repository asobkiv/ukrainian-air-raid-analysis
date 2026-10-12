"""Public-channel capture and point-in-time features, with explicit evidence modes.

Web publication times do NOT prove historical text availability: Telegram has
edit_hide. Strict mode uses observed_at; publication_assumed is exploratory only.
No views, reactions, forwards totals, or current subscriber counts are features.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from .series import midnight

CHANNEL = "war_monitor"
KEYWORDS = ["загроз", "ракет", "баліст", "шахед", "бпла", "пуск", "зліт", "циркон", "кинджал"]
FEATURES = ["posts_24h", "threat_posts_24h", "posts_6h"]
MODES = {"observed", "publication_assumed"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class Node:
    def __init__(self, tag="root", attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def text(self):
        return "".join(c.text() if isinstance(c, Node) else c for c in self.children)

    def has_class(self, name):
        return name in self.attrs.get("class", "").split()


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag == "br":
            node.children.append("\n")
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def parse_page(raw: str, observed_at: str):
    parser = PageParser()
    parser.feed(raw)
    records = []
    for node in parser.root.walk():
        identity = node.attrs.get("data-post", "")
        if not re.fullmatch(r"war_monitor/\d+", identity):
            continue
        descendants = list(node.walk())
        dates = [n.attrs["datetime"] for n in descendants if n.tag == "time" and "datetime" in n.attrs]
        if len(dates) != 1:
            raise ValueError(f"Expected one publication timestamp for {identity}, got {dates}")
        texts = list(dict.fromkeys(n.text() for n in descendants if n.has_class("tgme_widget_message_text") and n.has_class("js-message_text")))
        if len(texts) > 1:
            raise ValueError(f"Ambiguous text body {identity}")
        text = texts[0].strip() if texts else ""
        edited = any(n.has_class("tgme_widget_message_meta") and "edited" in n.text().lower() for n in descendants)
        # Keep a derived, redistributable record; full copyrighted text stays in local raw capture.
        records.append({"message_id": int(identity.split("/")[1]), "published_at": dates[0],
                        "observed_at": observed_at, "edited_marker": edited, "has_text": bool(text),
                        "threat_keyword": any(k in text.casefold() for k in KEYWORDS),
                        "technical_pause": "технічн" in text.casefold(),
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "url": "https://t.me/" + identity})
    return records


def capture(destination: Path, since: str, max_pages=700):
    """Sequential pagination with raw hashes. Resuming never fabricates first-seen times."""
    destination.mkdir(parents=True, exist_ok=True)
    raw_dir = destination / "raw"
    raw_dir.mkdir(exist_ok=True)
    manifest_path = destination / "captures.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    before, rows = None, []
    cutoff = pd.Timestamp(since, tz="UTC")
    complete = False
    for page in range(max_pages):
        key = "latest" if before is None else str(before)
        # An explicit invocation re-fetches latest, while historical pages are immutable/cacheable.
        cached = next((m for m in manifest if m["key"] == key), None) if before else None
        if cached:
            body = gzip.decompress((raw_dir / cached["file"]).read_bytes())
            meta = cached
            if hashlib.sha256(body).hexdigest() != meta["sha256"]:
                raise ValueError("Corrupt Telegram raw capture")
        else:
            url = f"https://t.me/s/{CHANNEL}" + (f"?before={before}" if before else "")
            for attempt in range(3):
                try:
                    with urlopen(Request(url, headers={"User-Agent": "airraid-research/0.1"}), timeout=35) as response:
                        body = response.read()
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
            observed = datetime.now(timezone.utc).isoformat()
            name = f"{key}_{len(manifest):04d}.html.gz"
            (raw_dir / name).write_bytes(gzip.compress(body, mtime=0))
            meta = {"key": key, "url": url, "observed_at": observed, "file": name,
                    "sha256": hashlib.sha256(body).hexdigest()}
            manifest.append(meta)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            time.sleep(.15)
        parsed = parse_page(body.decode("utf-8"), meta["observed_at"])
        if not parsed:
            raise ValueError("Telegram returned no parsed messages; no empty page is treated as zero activity")
        rows.extend(parsed)
        oldest = min(r["message_id"] for r in parsed)
        earliest = min(pd.Timestamp(r["published_at"]) for r in parsed)
        if page % 20 == 0:
            print(f"Telegram page {page + 1}, oldest {earliest}, id {oldest}", flush=True)
        if earliest <= cutoff:
            complete = True
            break
        if before is not None and oldest >= before:
            raise ValueError("Telegram pagination stopped advancing")
        before = oldest
    # Include all previously observed versions for future strict as-of use, not just this traversal.
    all_rows = []
    for meta in manifest:
        body = gzip.decompress((raw_dir / meta["file"]).read_bytes())
        if hashlib.sha256(body).hexdigest() != meta["sha256"]:
            raise ValueError("Capture hash mismatch")
        all_rows.extend(parse_page(body.decode("utf-8"), meta["observed_at"]))
    versions = pd.DataFrame(all_rows).sort_values("observed_at").drop_duplicates(["message_id", "text_sha256", "edited_marker"])
    versions.to_csv(destination / "message_versions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    if not complete:
        raise ValueError("Page budget exhausted before requested date; archive is incomplete")
    audit_capture(destination, since)
    names = ["message_versions.csv.gz", "captures.json", "daily_source_activity.csv", "source_audit.json"]
    (destination / "checksums.json").write_text(json.dumps({n: hashlib.sha256((destination / n).read_bytes()).hexdigest() for n in names}, indent=2))
    return versions


def load_messages(path: Path):
    d = pd.read_csv(path)
    for c in ["published_at", "observed_at"]:
        d[c] = pd.to_datetime(d[c], utc=True)
    if d.empty or d[["message_id", "text_sha256"]].isna().any().any() or (d.published_at > d.observed_at).any():
        raise ValueError("Invalid Telegram message ledger")
    for c in ["edited_marker", "has_text", "threat_keyword", "technical_pause"]:
        if not d[c].isin([True, False]).all():
            raise ValueError(f"Invalid boolean {c}")
    return d


def audit_capture(destination: Path, since: str):
    d = load_messages(destination / "message_versions.csv.gz")
    one = d.sort_values("observed_at").drop_duplicates("message_id", keep="last")
    manifest = json.loads((destination / "captures.json").read_text())
    # Request ranges must overlap or touch: ID gaps still may be deletions, not proof of zero posts.
    ranges = []
    for meta in manifest:
        path = destination / "raw" / meta["file"]
        body = gzip.decompress(path.read_bytes())
        if hashlib.sha256(body).hexdigest() != meta["sha256"]:
            raise ValueError("Raw hash mismatch")
        r = parse_page(body.decode(), meta["observed_at"])
        ranges.append((min(x["message_id"] for x in r), max(x["message_id"] for x in r), meta["key"]))
    requested = sorted((lo, hi) for lo, hi, key in ranges)
    for (lo, hi), (next_lo, next_hi) in zip(requested, requested[1:]):
        # A removed/nonexistent ID is allowed, but every next request must start at the previous minimum.
        if next_lo > hi + 1:
            bridge = any(key != "latest" and int(key) >= hi and int(key) <= next_lo + 1 for _, _, key in ranges)
            if not bridge:
                raise ValueError("Unrequested pagination gap")
    daily = one.assign(date=one.published_at.dt.tz_convert("Europe/Kyiv").dt.date).groupby("date").agg(
        posts=("message_id", "size"), edited=("edited_marker", "sum"), text_posts=("has_text", "sum"),
        technical_pause=("technical_pause", "sum")).reset_index()
    daily.to_csv(destination / "daily_source_activity.csv", index=False)
    audit = {"channel": CHANNEL, "requested_since_utc": since, "pages": len(manifest),
             "schema_version": 1, "threat_substrings": KEYWORDS,
             "message_versions": len(d), "unique_messages": len(one), "first_published": str(one.published_at.min()),
             "last_published": str(one.published_at.max()), "first_observed": str(d.observed_at.min()),
             "last_observed": str(d.observed_at.max()), "edited_marker_messages": int(one.edited_marker.sum()),
             "no_text_messages": int((~one.has_text).sum()), "technical_pause_messages": int(one.technical_pause.sum()),
             "missing_ids_in_span": int(one.message_id.max() - one.message_id.min() + 1 - len(one)),
             "raw_sha256_verified": True, "pagination_traversed": True,
             "historical_versions_proven": False, "complete_channel_history_proven": False,
             "warning": "ID gaps may be deletions; no edited marker does not prove no edit (Telegram edit_hide)."}
    (destination / "source_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def available_messages(messages: pd.DataFrame, origin, mode="observed"):
    if mode not in MODES:
        raise ValueError("Unknown Telegram evidence mode")
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError("Timezone-aware origin required")
    d = messages.loc[messages.published_at < origin].copy()
    if mode == "observed":
        d = d.loc[d.observed_at < origin]
    # Choose version before filtering: an edited latest version must not revive an older version.
    d = d.sort_values("observed_at").drop_duplicates("message_id", keep="last")
    if mode == "publication_assumed":
        d = d.loc[~d.edited_marker]
    return d


def feature_row(messages, day, mode="observed"):
    origin = midnight(pd.Timestamp(day).date())
    d = available_messages(messages, origin, mode)
    recent = d.loc[(d.published_at >= origin - pd.Timedelta(hours=24)) & d.has_text]
    counts = [len(recent), int(recent.threat_keyword.sum()), int((recent.published_at >= origin - pd.Timedelta(hours=6)).sum())]
    return {"date": str(pd.Timestamp(day).date()), "origin": origin.isoformat(),
            **dict(zip(FEATURES, counts)), "max_publication_used": str(recent.published_at.max()) if len(recent) else "",
            "max_observation_used": str(recent.observed_at.max()) if len(recent) else "", "mode": mode}


def feature_table(messages, dates, mode="observed"):
    return pd.DataFrame([feature_row(messages, d, mode) for d in dates])
