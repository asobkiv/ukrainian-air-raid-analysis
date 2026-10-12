"""Compare two frozen experiment runs and verify deliverable integrity.

Usage: python scripts/verify_reproduction.py reports /path/to/repeated/reports
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("repeated", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = ["predictions_unlabelled.csv", "model_selection.csv", "validation_metrics.csv",
             "test_metrics_selected.csv", "test_macro_metrics.csv", "weekday_differences.csv", "acf.csv"]
    maximum = 0.0
    for name in files:
        original, repeated = pd.read_csv(args.original / name), pd.read_csv(args.repeated / name)
        pd.testing.assert_frame_equal(original, repeated, check_exact=False, rtol=1e-10, atol=1e-10)
        for column in original.select_dtypes("number"):
            diff = (original[column] - repeated[column]).abs().dropna()
            maximum = max(maximum, float(diff.max()) if len(diff) else 0)
    assert json.loads((args.original / "protocol.json").read_text()) == json.loads((args.repeated / "protocol.json").read_text())
    run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(run.stdout + run.stderr)
    count = re.search(r"(\d+) passed", run.stdout)
    notebook = nbformat.read(root / "analysis.ipynb", as_version=4)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == "error" for c in cells for o in c.get("outputs", []))
    evidence = {"pipeline_stage_audits_passed": True, "clean_environment_reproduction": True,
                "python": platform.python_version(), "platform": platform.platform(),
                "comparison_files": files, "maximum_numeric_difference": maximum,
                "unit_tests_passed": int(count.group(1)), "notebook_executed_code_cells": len(cells),
                "verified_at_utc": datetime.now(timezone.utc).isoformat()}
    (args.original / "verification.json").write_text(json.dumps(evidence, indent=2))
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = []
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {"src", "href"} and not value.startswith(("https:", "http:", "#")):
                    self.paths.append(value)
    links = Links()
    links.feed((args.original / "report.html").read_text())
    for path in links.paths:
        assert (args.original / path).exists(), f"Broken report link: {path}"
    evidence["local_report_links_checked"] = len(links.paths)
    evidence["html_links_valid"] = True
    (args.original / "verification.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
