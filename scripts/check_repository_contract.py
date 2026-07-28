from __future__ import annotations

import re
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
    "README_zh-CN.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "pyproject.toml",
    "assets/road-affordance-lab-banner.svg",
    "assets/c3-farnet-overview.svg",
    "assets/verified-results.svg",
    "docs/algorithm.md",
    "docs/algorithm_zh.md",
    "docs/data_and_reproduction.md",
    "docs/data_and_reproduction_zh-CN.md",
    "docs/results_current_best.md",
    "docs/results_current_best_zh-CN.md",
    "results/current_best_s7/metrics_summary.json",
    "configs/c3_farnet/current_best_s7_public.yaml",
)

PUBLIC_MARKDOWN = (
    ROOT / "README.md",
    ROOT / "README_zh-CN.md",
    ROOT / "docs" / "data_and_reproduction_zh-CN.md",
    ROOT / "docs" / "results_current_best_zh-CN.md",
)

MACHINE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\(?:Users|Documents|Desktop|Anaconda|路面感知)\\", re.IGNORECASE),
    re.compile(r"/(?:home|Users)/[^/\s]+/"),
)

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def check_required_files(errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required public artifact: {relative}")


def check_language_navigation(errors: list[str]) -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README_zh-CN.md").read_text(encoding="utf-8")
    if "README_zh-CN.md" not in english:
        errors.append("README.md does not link to README_zh-CN.md")
    if "README.md" not in chinese:
        errors.append("README_zh-CN.md does not link back to README.md")


def check_local_markdown_links(errors: list[str]) -> None:
    for markdown in PUBLIC_MARKDOWN:
        text = markdown.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            target = target.strip().split("#", maxsplit=1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            path = (markdown.parent / target).resolve()
            try:
                path.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(f"link escapes repository: {markdown.relative_to(ROOT)} -> {target}")
                continue
            if not path.exists():
                errors.append(f"broken local link: {markdown.relative_to(ROOT)} -> {target}")


def check_machine_paths(errors: list[str]) -> None:
    for path in PUBLIC_MARKDOWN:
        text = path.read_text(encoding="utf-8")
        for pattern in MACHINE_PATH_PATTERNS:
            if pattern.search(text):
                errors.append(f"machine-specific absolute path in {path.relative_to(ROOT)}")


def check_svg_assets(errors: list[str]) -> None:
    for relative in (
        "assets/road-affordance-lab-banner.svg",
        "assets/c3-farnet-overview.svg",
        "assets/verified-results.svg",
    ):
        try:
            ET.parse(ROOT / relative)
        except ET.ParseError as exc:
            errors.append(f"invalid SVG XML in {relative}: {exc}")


def check_headline_evidence(errors: list[str]) -> None:
    payload = json.loads(
        (ROOT / "results/current_best_s7/metrics_summary.json").read_text(encoding="utf-8-sig")
    )
    expected = {
        f"{100.0 * float(payload['top1']):.3f}%",
        f"{100.0 * float(payload['macro_f1']):.3f}%",
        f"{100.0 * float(payload['water_concrete_slight_f1']):.3f}%",
        f"{int(payload['num_samples']):,}",
    }
    for readme in (ROOT / "README.md", ROOT / "README_zh-CN.md"):
        text = readme.read_text(encoding="utf-8")
        for value in expected:
            if value not in text:
                errors.append(f"{readme.name} headline is missing evidence value {value}")


def main() -> int:
    errors: list[str] = []
    check_required_files(errors)
    check_language_navigation(errors)
    check_local_markdown_links(errors)
    check_machine_paths(errors)
    check_svg_assets(errors)
    check_headline_evidence(errors)
    if errors:
        print("Repository contract check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Repository contract check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
