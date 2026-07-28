#!/usr/bin/env python3
"""Audit public result payloads and documentation using the standard library."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "current_best_s7"
EXPECTED = {
    "top1": 0.9063232323232323,
    "macro_f1": 0.8891966645824468,
    "weighted_f1": 0.90653926148745,
    "num_samples": 49500,
    "num_classes": 27,
    "water_concrete_slight_f1": 0.7569310122501612,
}


def fail(message: str) -> None:
    raise AssertionError(message)


def audit_metrics() -> None:
    payload = json.loads((RESULTS / "metrics_summary.json").read_text(encoding="utf-8-sig"))
    for key, expected in EXPECTED.items():
        actual = payload.get(key)
        if isinstance(expected, float):
            if actual is None or not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12):
                fail(f"metrics_summary.json: {key}={actual!r}, expected {expected!r}")
        elif actual != expected:
            fail(f"metrics_summary.json: {key}={actual!r}, expected {expected!r}")


def audit_per_class() -> None:
    with (RESULTS / "per_class_metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 27:
        fail(f"per_class_metrics.csv has {len(rows)} classes, expected 27")
    if len({row["class"] for row in rows}) != 27:
        fail("per_class_metrics.csv contains duplicate class labels")
    support = sum(float(row["support"]) for row in rows)
    if not math.isclose(support, 49500.0, rel_tol=0, abs_tol=1e-9):
        fail(f"per-class support sums to {support}, expected 49500")
    for row in rows:
        for key in ("precision", "recall", "f1"):
            value = float(row[key])
            if not 0.0 <= value <= 1.0:
                fail(f"{row['class']} {key} is outside [0, 1]: {value}")


def audit_confusion() -> None:
    with (RESULTS / "confusion_matrix.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    labels = rows[0][1:]
    if len(labels) != 27 or len(rows[1:]) != 27:
        fail("confusion_matrix.csv is not 27×27")
    if [row[0] for row in rows[1:]] != labels:
        fail("confusion-matrix row/column labels differ")
    matrix = [[int(float(cell)) for cell in row[1:]] for row in rows[1:]]
    if any(len(row) != 27 for row in matrix):
        fail("confusion_matrix.csv has an invalid row width")
    if sum(sum(row) for row in matrix) != 49500:
        fail("confusion-matrix total is not 49,500")


def audit_local_markdown_links() -> None:
    markdown_files = [
        ROOT / "README.md",
        ROOT / "README_zh-CN.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md",
        ROOT / "docs" / "engineering_case_study.md",
        ROOT / "docs" / "research_status.md",
        ROOT / "docs" / "results_current_best.md",
    ]
    patterns = [
        re.compile(r"!?\[[^\]]*\]\(([^)]+)\)"),
        re.compile(r'''(?:src|href)=["']([^"']+)["']'''),
    ]
    for markdown in markdown_files:
        content = markdown.read_text(encoding="utf-8")
        raw_targets = [target for pattern in patterns for target in pattern.findall(content)]
        for raw_target in raw_targets:
            target = raw_target.strip().split(" ", 1)[0].strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            resolved = (markdown.parent / target).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                fail(f"{markdown.name}: link escapes repository: {target}")
            if not resolved.exists():
                fail(f"{markdown.name}: missing local link target: {target}")


def audit_assets() -> None:
    expected = {
        "hero.svg",
        "architecture.svg",
        "results-overview.svg",
        "per-class-f1.svg",
        "confusion-matrix.svg",
    }
    present = {path.name for path in (ROOT / "assets").glob("*.svg")}
    missing = expected - present
    if missing:
        fail(f"missing public assets: {sorted(missing)}")
    for name in expected:
        content = (ROOT / "assets" / name).read_text(encoding="utf-8")
        if not content.startswith("<svg") or "</svg>" not in content:
            fail(f"invalid SVG envelope: {name}")
        try:
            ET.fromstring(content)
        except ET.ParseError as exc:
            fail(f"malformed SVG {name}: {exc}")


def audit_local_secret_files() -> None:
    forbidden = [
        ROOT / ".env",
        ROOT / ".env.paper-search",
        ROOT / "configs" / "data" / "local_paths.yaml",
    ]
    existing = [path.relative_to(ROOT).as_posix() for path in forbidden if path.exists()]
    if existing:
        fail(f"machine-local files are present in the release worktree: {existing}")


def main() -> int:
    checks = [
        audit_metrics,
        audit_per_class,
        audit_confusion,
        audit_local_markdown_links,
        audit_assets,
        audit_local_secret_files,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("PUBLIC RELEASE AUDIT: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"PUBLIC RELEASE AUDIT: FAIL — {exc}", file=sys.stderr)
        raise SystemExit(1)
