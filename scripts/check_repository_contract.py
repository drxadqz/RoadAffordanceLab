from __future__ import annotations

import json
import re
import sys
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
    "assets/drel-qrfme-overview.svg",
    "assets/c3-farnet-overview.svg",
    "docs/drel_qrfme_full_release.md",
    "docs/drel_qrfme_full_release_zh-CN.md",
    "docs/data_and_reproduction.md",
    "docs/data_and_reproduction_zh-CN.md",
    "docs/results_current_best.md",
    "docs/results_current_best_zh-CN.md",
    "docs/drel_algorithm.md",
    "docs/drel_algorithm_zh-CN.md",
    "results/drel_qrfme_epoch097/metrics_summary.json",
    "results/drel_qrfme_epoch097/training_metrics_epochs001-100.csv",
    "results/drel_qrfme_epoch097/test_metrics_direct_resize224.json",
    "results/drel_qrfme_epoch097/test_metrics_rspnet_center_crop.json",
    "results/drel_qrfme_epoch097/per_class_metrics_direct_resize224.csv",
    "results/drel_qrfme_epoch097/confusion_matrix_direct_resize224.csv",
    "results/current_best_s7/metrics_summary.json",
    "results/drel_d350_validation/metrics_summary.json",
    "configs/drel_qrfme/base_drel_qrfme_model.yaml",
    "configs/drel_qrfme/rscd_full_train_seed097.yaml",
    "src/drel_qrfme/models/drel_qrfme_model.py",
    "tests/test_drel_qrfme_full.py",
    "checkpoints/drel_qrfme_epoch097/best_checkpoint.pth",
    "checkpoints/drel_qrfme_epoch097/CHECKPOINT.sha256",
)

PUBLIC_MARKDOWN = tuple(
    ROOT / relative
    for relative in (
        "README.md",
        "README_zh-CN.md",
        "docs/drel_qrfme_full_release.md",
        "docs/drel_qrfme_full_release_zh-CN.md",
        "docs/data_and_reproduction.md",
        "docs/data_and_reproduction_zh-CN.md",
        "docs/results_current_best.md",
        "docs/results_current_best_zh-CN.md",
        "docs/drel_algorithm.md",
        "docs/drel_algorithm_zh-CN.md",
        "docs/drel_validation_evidence.md",
        "docs/drel_validation_evidence_zh-CN.md",
    )
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
        "assets/drel-qrfme-overview.svg",
        "assets/c3-farnet-overview.svg",
        "assets/verified-results.svg",
    ):
        try:
            ET.parse(ROOT / relative)
        except ET.ParseError as exc:
            errors.append(f"invalid SVG XML in {relative}: {exc}")


def check_headline_evidence(errors: list[str]) -> None:
    payload = json.loads(
        (ROOT / "results/drel_qrfme_epoch097/metrics_summary.json").read_text(encoding="utf-8")
    )
    result = payload["test_direct_resize224"]
    expected = {
        f"{100.0 * float(result['top1']):.3f}%",
        f"{100.0 * float(result['macro_f1']):.3f}%",
        f"{100.0 * float(result['bottom5_mean_f1']):.3f}%",
        f"{100.0 * float(result['weakest_class_f1']):.3f}%",
        f"{int(payload['splits']['test_samples']):,}",
        f"{int(payload['parameters']):,}",
        f"Epoch {int(payload['checkpoint_epoch'])}",
    }
    for readme in (ROOT / "README.md", ROOT / "README_zh-CN.md"):
        text = readme.read_text(encoding="utf-8")
        for value in expected:
            if value not in text:
                errors.append(f"{readme.name} headline is missing evidence value {value}")

    if payload.get("pretrained") is not False:
        errors.append("DREL-QRFME full release must preserve pretrained=false")
    if payload.get("checkpoint_epoch") != payload.get("best_validation", {}).get("epoch"):
        errors.append("released checkpoint epoch must match the validation-selected epoch")
    if payload.get("ensemble") is not False or payload.get("test_time_augmentation") is not False:
        errors.append("headline evidence must remain single-model without TTA")
    if not str(payload.get("claim_boundary", "")).strip():
        errors.append("DREL-QRFME evidence must include a non-empty claim boundary")


def check_checkpoint_record(errors: list[str]) -> None:
    payload = json.loads(
        (ROOT / "results/drel_qrfme_epoch097/metrics_summary.json").read_text(encoding="utf-8")
    )
    record = (ROOT / "checkpoints/drel_qrfme_epoch097/CHECKPOINT.sha256").read_text(
        encoding="utf-8"
    )
    if str(payload["checkpoint_sha256"]) not in record:
        errors.append("checkpoint SHA-256 record does not match metrics_summary.json")


def check_legacy_validation_boundary(errors: list[str]) -> None:
    payload = json.loads(
        (ROOT / "results/drel_d350_validation/metrics_summary.json").read_text(encoding="utf-8")
    )
    if payload.get("test_data_accessed") is not False:
        errors.append("legacy DREL validation evidence must preserve the test firewall")
    if payload.get("formal_test_claim") is not False:
        errors.append("legacy DREL D350 validation must not be a formal-test claim")


def main() -> int:
    errors: list[str] = []
    check_required_files(errors)
    check_language_navigation(errors)
    check_local_markdown_links(errors)
    check_machine_paths(errors)
    check_svg_assets(errors)
    check_headline_evidence(errors)
    check_checkpoint_record(errors)
    check_legacy_validation_boundary(errors)
    if errors:
        print("Repository contract check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Repository contract check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
