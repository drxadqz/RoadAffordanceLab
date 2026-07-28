from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "current_best_s7"
LINEAGE_DIR = ROOT / "results" / "s7_lineage"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_metrics() -> None:
    metrics = json.loads((RESULT_DIR / "metrics_summary.json").read_text(encoding="utf-8-sig"))
    required = {
        "top1",
        "macro_f1",
        "weighted_f1",
        "num_samples",
        "num_classes",
        "friction_acc",
        "material_acc",
        "roughness_acc",
        "water_concrete_slight_f1",
    }
    missing = required.difference(metrics)
    require(not missing, f"metrics_summary.json missing fields: {sorted(missing)}")
    require(int(metrics["num_samples"]) == 49_500, "unexpected test sample count")
    require(int(metrics["num_classes"]) == 27, "unexpected class count")
    for key in required.difference({"num_samples", "num_classes"}):
        require(0.0 <= float(metrics[key]) <= 1.0, f"metric outside [0, 1]: {key}")

    with (RESULT_DIR / "per_class_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        per_class = list(csv.DictReader(handle))
    require(len(per_class) == 27, "per_class_metrics.csv must contain 27 rows")
    names = [row["class"] for row in per_class]
    require(len(set(names)) == 27, "class names are not unique")
    supports = [float(row["support"]) for row in per_class]
    require(round(sum(supports)) == 49_500, "per-class support does not sum to 49,500")
    macro = sum(float(row["f1"]) for row in per_class) / len(per_class)
    weighted = sum(float(row["f1"]) * support for row, support in zip(per_class, supports)) / sum(supports)
    require(abs(macro - float(metrics["macro_f1"])) < 1e-12, "Macro-F1 mismatch")
    require(abs(weighted - float(metrics["weighted_f1"])) < 1e-12, "weighted-F1 mismatch")
    weakest = min(per_class, key=lambda row: float(row["f1"]))
    require(weakest["class"] == "water_concrete_slight", "unexpected weakest class")
    require(
        abs(float(weakest["f1"]) - float(metrics["water_concrete_slight_f1"])) < 1e-12,
        "weakest-class F1 mismatch",
    )

    with (RESULT_DIR / "confusion_matrix.csv").open(encoding="utf-8-sig", newline="") as handle:
        matrix_rows = list(csv.reader(handle))
    require(len(matrix_rows) == 28, "confusion matrix must have one header and 27 rows")
    header = matrix_rows[0][1:]
    require(header == names, "confusion matrix class order differs from per-class metrics")
    require(all(len(row) == 28 for row in matrix_rows[1:]), "confusion matrix row width mismatch")
    values = [[int(value) for value in row[1:]] for row in matrix_rows[1:]]
    total = sum(sum(row) for row in values)
    diagonal = sum(values[i][i] for i in range(27))
    require(total == 49_500, "confusion matrix does not sum to 49,500")
    require(abs(diagonal / total - float(metrics["top1"])) < 1e-12, "Top-1 mismatch")

    with (RESULT_DIR / "hard_pair_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        hard_pairs = list(csv.DictReader(handle))
    require(len(hard_pairs) >= 20, "hard-pair evidence is unexpectedly small")
    require({row["axis"] for row in hard_pairs} >= {"friction", "material", "roughness"}, "factor axes missing")


def _relative_links(markdown: Path) -> set[Path]:
    text = markdown.read_text(encoding="utf-8")
    targets: set[str] = set()
    targets.update(re.findall(r"\[[^\]]*\]\(([^)]+)\)", text))
    targets.update(re.findall(r'(?:src|href)="([^"]+)"', text))
    resolved: set[Path] = set()
    for raw in targets:
        target = raw.strip().split()[0].strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = unquote(target.split("#", 1)[0])
        if target:
            resolved.add((markdown.parent / target).resolve())
    return resolved


def verify_links() -> None:
    markdown_files = [
        ROOT / "README.md",
        ROOT / "README_zh-CN.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md",
        ROOT / "docs" / "engineering.md",
        ROOT / "docs" / "model_card.md",
    ]
    missing: list[str] = []
    for markdown in markdown_files:
        require(markdown.exists(), f"missing documentation file: {markdown.relative_to(ROOT)}")
        for target in _relative_links(markdown):
            if not target.exists():
                missing.append(f"{markdown.relative_to(ROOT)} -> {target}")
    require(not missing, "broken local documentation links:\n" + "\n".join(missing))


def _read_lfs_pointer(path: Path) -> tuple[str, int] | None:
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    if not content.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    oid_match = re.search(r"^oid sha256:([0-9a-f]{64})$", content, re.MULTILINE)
    size_match = re.search(r"^size (\d+)$", content, re.MULTILINE)
    require(oid_match is not None and size_match is not None, f"malformed LFS pointer: {path}")
    return oid_match.group(1), int(size_match.group(1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoints(*, require_materialized: bool) -> None:
    manifest_path = LINEAGE_DIR / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(len(manifest) >= 4, "checkpoint manifest is incomplete")
    for item in manifest:
        path = ROOT / item["path"]
        require(path.exists(), f"missing checkpoint or LFS pointer: {item['path']}")
        pointer = _read_lfs_pointer(path)
        if pointer is not None:
            require(not require_materialized, f"checkpoint is still an LFS pointer; run git lfs pull: {item['path']}")
            oid, size = pointer
            require(oid == item["sha256"], f"LFS OID mismatch: {item['path']}")
            require(size == int(item["bytes"]), f"LFS size mismatch: {item['path']}")
            continue
        require(path.stat().st_size == int(item["bytes"]), f"checkpoint byte-size mismatch: {item['path']}")
        if require_materialized:
            require(_sha256(path) == item["sha256"], f"checkpoint SHA256 mismatch: {item['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the public RoadAffordanceLab release contract.")
    parser.add_argument(
        "--check-checkpoints",
        action="store_true",
        help="Require materialized Git-LFS checkpoint binaries and verify their SHA256 digests.",
    )
    args = parser.parse_args()
    verify_metrics()
    verify_links()
    verify_checkpoints(require_materialized=args.check_checkpoints)
    print("RELEASE_CONTRACT=PASS")
    print("TEST_IMAGES=49500")
    print("CLASSES=27")
    print(f"CHECKPOINT_MODE={'materialized' if args.check_checkpoints else 'pointer_or_materialized'}")


if __name__ == "__main__":
    main()
