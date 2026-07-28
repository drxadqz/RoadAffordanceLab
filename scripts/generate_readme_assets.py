from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "current_best_s7"
ASSET_DIR = ROOT / "docs" / "assets"


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 24,
    weight: int = 400,
    fill: str = "#1F2937",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Inter, Segoe UI, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def _svg_document(width: int, height: int, body: list[str], *, title: str, description: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"  <title id=\"title\">{html.escape(title)}</title>",
            f"  <desc id=\"desc\">{html.escape(description)}</desc>",
            *[f"  {line}" for line in body],
            "</svg>",
            "",
        ]
    )


def render_hero() -> str:
    body = [
        "<defs>",
        '  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '    <stop offset="0" stop-color="#071A3D"/>',
        '    <stop offset="0.58" stop-color="#155EEF"/>',
        '    <stop offset="1" stop-color="#0891B2"/>',
        "  </linearGradient>",
        '  <linearGradient id="road" x1="0" y1="0" x2="1" y2="0">',
        '    <stop offset="0" stop-color="#FFFFFF" stop-opacity="0.06"/>',
        '    <stop offset="0.5" stop-color="#FFFFFF" stop-opacity="0.28"/>',
        '    <stop offset="1" stop-color="#FFFFFF" stop-opacity="0.06"/>',
        "  </linearGradient>",
        '  <filter id="glow"><feGaussianBlur stdDeviation="9"/></filter>',
        "</defs>",
        '<rect width="1200" height="360" rx="28" fill="url(#bg)"/>',
        '<circle cx="1040" cy="55" r="135" fill="#FFFFFF" opacity="0.06"/>',
        '<circle cx="1070" cy="330" r="175" fill="#F97316" opacity="0.10"/>',
        '<path d="M670 360 L815 76 L995 76 L1160 360 Z" fill="url(#road)"/>',
        '<path d="M907 100 L919 100 L940 175 L928 175 Z" fill="#FFFFFF" opacity="0.70"/>',
        '<path d="M948 205 L968 205 L1002 326 L978 326 Z" fill="#FFFFFF" opacity="0.70"/>',
        '<g fill="none" stroke="#7DD3FC" stroke-width="2" opacity="0.62">',
        '  <path d="M760 86 C815 130 800 188 860 228 C900 255 908 300 936 344"/>',
        '  <path d="M825 92 C872 130 870 175 914 198"/>',
        "</g>",
        '<g fill="#FFFFFF">',
        '  <circle cx="760" cy="86" r="7"/><circle cx="860" cy="228" r="7"/><circle cx="936" cy="344" r="7"/>',
        '  <circle cx="825" cy="92" r="7"/><circle cx="914" cy="198" r="7"/>',
        "</g>",
        _text(72, 96, "RoadAffordanceLab", size=48, weight=500, fill="#FFFFFF"),
        _text(72, 143, "Factor-aware road-surface intelligence", size=27, weight=400, fill="#D7E8FF"),
        _text(72, 201, "From physical evidence to auditable decisions", size=22, weight=400, fill="#FFFFFF"),
        '<g>',
        '  <rect x="72" y="242" width="184" height="42" rx="21" fill="#FFFFFF" opacity="0.14"/>',
        '  <rect x="270" y="242" width="170" height="42" rx="21" fill="#FFFFFF" opacity="0.14"/>',
        '  <rect x="454" y="242" width="202" height="42" rx="21" fill="#FFFFFF" opacity="0.14"/>',
        _text(164, 270, "Physics-guided", size=17, weight=500, fill="#FFFFFF", anchor="middle"),
        _text(355, 270, "Factor-coupled", size=17, weight=500, fill="#FFFFFF", anchor="middle"),
        _text(555, 270, "Reproducible", size=17, weight=500, fill="#FFFFFF", anchor="middle"),
        "</g>",
        _text(72, 326, "PyTorch  •  CUDA  •  RSCD  •  Evidence-first evaluation", size=18, fill="#D7E8FF"),
    ]
    return _svg_document(
        1200,
        360,
        body,
        title="RoadAffordanceLab",
        description="A blue research-engineering banner for factor-aware, physics-guided road-surface intelligence.",
    )


def _arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#94A3B8") -> list[str]:
    return [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3"/>',
        f'<path d="M{x2} {y2} l-12 -7 v14 z" fill="{color}"/>',
    ]


def _node(x: int, y: int, w: int, h: int, title: str, subtitle: str, color: str) -> list[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="#FFFFFF" stroke="{color}" stroke-width="3"/>',
        _text(x + w / 2, y + 40, title, size=22, weight=500, fill=color, anchor="middle"),
        _text(x + w / 2, y + 70, subtitle, size=15, fill="#475569", anchor="middle"),
    ]


def render_architecture() -> str:
    body = [
        '<rect width="1200" height="520" rx="24" fill="#F8FAFC"/>',
        _text(54, 58, "C3-FaRNet · coupled evidence pipeline", size=30, weight=500, fill="#0F172A"),
        _text(54, 88, "One image, three factor views, one auditable 27-class decision", size=17, fill="#64748B"),
    ]
    # Connections are drawn before nodes so arrows never cross labels.
    body += _arrow(190, 250, 276, 170)
    body += _arrow(190, 266, 276, 350)
    body += _arrow(480, 170, 570, 250)
    body += _arrow(480, 350, 570, 270)
    body += _arrow(760, 260, 825, 260)
    body += _arrow(995, 260, 1030, 260)
    body += _node(40, 210, 150, 100, "RGB image", "192×192", "#155EEF")
    body += _node(276, 115, 204, 110, "Visual carrier", "ConvNeXt context", "#155EEF")
    body += _node(276, 295, 204, 110, "Physics evidence", "wetness · texture", "#0891B2")
    body += _node(570, 200, 190, 120, "Factor coupling", "condition × material", "#7C3AED")
    body += _node(825, 205, 170, 110, "Hard-pair head", "boundary residuals", "#F97316")
    body += _node(1030, 210, 135, 100, "27 classes", "single model", "#155EEF")
    chips = [
        (565, 370, 164, "condition / friction", "#DBEAFE", "#155EEF"),
        (744, 370, 130, "material", "#CFFAFE", "#0E7490"),
        (889, 370, 144, "roughness", "#FFEDD5", "#C2410C"),
    ]
    for x, y, w, label, bg, fg in chips:
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="42" rx="21" fill="{bg}"/>')
        body.append(_text(x + w / 2, y + 27, label, size=16, weight=500, fill=fg, anchor="middle"))
    body.append(_text(600, 472, "The physics branch conditions difficult decisions; it does not claim to measure physical friction directly.", size=16, fill="#64748B", anchor="middle"))
    return _svg_document(
        1200,
        520,
        body,
        title="C3-FaRNet architecture",
        description="RGB input is processed by visual and physics evidence branches, combined through factor coupling and a hard-pair head, and emitted as 27 classes.",
    )


def render_metrics(metrics: dict[str, float]) -> str:
    values = [
        ("Top-1", metrics["top1"], "#155EEF"),
        ("Macro-F1", metrics["macro_f1"], "#0891B2"),
        ("Condition", metrics["friction_acc"], "#7C3AED"),
        ("Material", metrics["material_acc"], "#0F766E"),
        ("Roughness", metrics["roughness_acc"], "#F97316"),
        ("Weakest F1", metrics["water_concrete_slight_f1"], "#DC2626"),
    ]
    body = [
        '<rect width="1200" height="530" rx="24" fill="#F8FAFC"/>',
        _text(54, 58, "Verified full-test evidence", size=30, weight=500, fill="#0F172A"),
        _text(54, 88, "Single model · 49,500 images · values generated from metrics_summary.json", size=17, fill="#64748B"),
    ]
    for i, (label, value, color) in enumerate(values):
        col = i % 2
        row = i // 2
        x = 54 + col * 570
        y = 132 + row * 120
        body.append(_text(x, y + 26, label, size=19, weight=500, fill="#334155"))
        body.append(_text(x + 516, y + 26, f"{value * 100:.2f}%", size=21, weight=500, fill=color, anchor="end"))
        body.append(f'<rect x="{x}" y="{y + 47}" width="516" height="22" rx="11" fill="#E2E8F0"/>')
        body.append(f'<rect x="{x}" y="{y + 47}" width="{516 * value:.2f}" height="22" rx="11" fill="{color}"/>')
    body.append(_text(54, 495, "Not a SOTA claim · historical 192×192 letterbox protocol · see model card for limits", size=16, fill="#64748B"))
    return _svg_document(
        1200,
        530,
        body,
        title="Verified C3-FaRNet S7 metrics",
        description="Horizontal bars show Top-1, Macro-F1, condition, material, roughness and weakest-class F1 metrics.",
    )


def render_per_class(rows: list[dict[str, str]]) -> str:
    ordered = sorted(rows, key=lambda row: float(row["f1"]))
    height = 132 + len(ordered) * 34 + 72
    chart_x = 338
    chart_w = 760
    body = [
        f'<rect width="1200" height="{height}" rx="24" fill="#F8FAFC"/>',
        _text(54, 58, "Per-class F1", size=30, weight=500, fill="#0F172A"),
        _text(54, 88, "All 27 classes · sorted by F1 · difficult water/wet concrete classes highlighted", size=17, fill="#64748B"),
        f'<line x1="{chart_x}" y1="112" x2="{chart_x}" y2="{height - 52}" stroke="#CBD5E1"/>',
        f'<line x1="{chart_x + chart_w / 2}" y1="112" x2="{chart_x + chart_w / 2}" y2="{height - 52}" stroke="#E2E8F0"/>',
        f'<line x1="{chart_x + chart_w}" y1="112" x2="{chart_x + chart_w}" y2="{height - 52}" stroke="#CBD5E1"/>',
    ]
    focus = {
        "water_concrete_slight",
        "water_concrete_severe",
        "wet_concrete_slight",
        "wet_concrete_severe",
    }
    for i, row in enumerate(ordered):
        label = row["class"]
        value = float(row["f1"])
        y = 128 + i * 34
        color = "#F97316" if label in focus else "#155EEF"
        body.append(_text(chart_x - 18, y + 16, label, size=15, weight=500 if label in focus else 400, fill="#334155", anchor="end"))
        body.append(f'<rect x="{chart_x}" y="{y}" width="{chart_w}" height="21" rx="7" fill="#E2E8F0"/>')
        body.append(f'<rect x="{chart_x}" y="{y}" width="{chart_w * value:.2f}" height="21" rx="7" fill="{color}"/>')
        body.append(_text(chart_x + chart_w + 16, y + 16, f"{value * 100:.1f}", size=15, weight=500, fill=color))
    body.append(_text(chart_x, height - 25, "0", size=14, fill="#64748B", anchor="middle"))
    body.append(_text(chart_x + chart_w / 2, height - 25, "50", size=14, fill="#64748B", anchor="middle"))
    body.append(_text(chart_x + chart_w, height - 25, "100%", size=14, fill="#64748B", anchor="middle"))
    return _svg_document(
        1200,
        height,
        body,
        title="Per-class F1 on the released RSCD test record",
        description="All 27 class F1 scores are sorted from lowest to highest; difficult water and wet concrete classes are orange.",
    )


def build_assets() -> dict[Path, str]:
    metrics = json.loads((RESULT_DIR / "metrics_summary.json").read_text(encoding="utf-8-sig"))
    with (RESULT_DIR / "per_class_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        ASSET_DIR / "road-affordance-lab-hero.svg": render_hero(),
        ASSET_DIR / "c3-farnet-architecture.svg": render_architecture(),
        ASSET_DIR / "verified-metrics.svg": render_metrics(metrics),
        ASSET_DIR / "per-class-f1.svg": render_per_class(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README SVG assets from committed result evidence.")
    parser.add_argument("--check", action="store_true", help="Fail if committed assets are stale.")
    args = parser.parse_args()

    assets = build_assets()
    stale: list[Path] = []
    for path, content in assets.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(path.relative_to(ROOT))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")

    if stale:
        rendered = ", ".join(str(path) for path in stale)
        raise SystemExit(f"README assets are missing or stale: {rendered}")
    action = "verified" if args.check else "generated"
    print(f"README_ASSETS_{action.upper()}={len(assets)}")


if __name__ == "__main__":
    main()
