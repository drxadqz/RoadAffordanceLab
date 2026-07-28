#!/usr/bin/env python3
"""Build deterministic, dependency-free SVG assets for the public README.

Every displayed metric is read from the checked-in result payloads.  The
script deliberately uses only the Python standard library so the public
release can rebuild its visual evidence without installing the training stack.
"""

from __future__ import annotations

import csv
import json
import math
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "current_best_s7"
ASSETS = ROOT / "assets"

NAVY = "#07152D"
NAVY_2 = "#0D2245"
BLUE = "#3B82F6"
CYAN = "#22D3EE"
ORANGE = "#F97316"
GREEN = "#22C55E"
INK = "#172033"
MUTED = "#64748B"
LIGHT = "#F7FAFC"
LINE = "#DCE6F2"
WHITE = "#FFFFFF"


def write_svg(name: str, width: int, height: int, body: str, *, dark: bool = False) -> None:
    background = NAVY if dark else WHITE
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <title>{escape(name.replace('-', ' ').title())}</title>
  <rect width="{width}" height="{height}" fill="{background}"/>
  {body}
</svg>
'''
    (ASSETS / f"{name}.svg").write_text(svg, encoding="utf-8", newline="\n")


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 24,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
    opacity: float = 1.0,
    family: str = "Inter,Segoe UI,Arial,sans-serif",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}" opacity="{opacity:.3f}">{escape(value)}</text>'
    )


def round_rect(x: float, y: float, w: float, h: float, fill: str, *, radius: int = 18,
               stroke: str = "none", stroke_width: float = 1.0, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" '
        f'opacity="{opacity:.3f}"/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = LINE,
         width: float = 2, dash: str | None = None, opacity: float = 1.0) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity:.3f}"{dash_attr}/>'
    )


def polyline(points: list[tuple[float, float]], *, stroke: str, width: float = 3,
             fill: str = "none") -> str:
    payload = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{payload}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def load_metrics() -> dict[str, object]:
    return json.loads((RESULTS / "metrics_summary.json").read_text(encoding="utf-8-sig"))


def load_per_class() -> list[dict[str, str]]:
    with (RESULTS / "per_class_metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_confusion() -> tuple[list[str], list[list[int]]]:
    with (RESULTS / "confusion_matrix.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    labels = rows[0][1:]
    matrix = [[int(float(cell)) for cell in row[1:]] for row in rows[1:]]
    if [row[0] for row in rows[1:]] != labels:
        raise ValueError("Confusion-matrix row and column labels differ")
    return labels, matrix


def build_hero() -> None:
    body: list[str] = [
        '''<defs>
  <linearGradient id="hero-bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#07152D"/><stop offset="1" stop-color="#123A67"/>
  </linearGradient>
  <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#22D3EE"/><stop offset="1" stop-color="#3B82F6"/>
  </linearGradient>
  <filter id="glow"><feGaussianBlur stdDeviation="18"/></filter>
</defs>
<rect width="1400" height="500" fill="url(#hero-bg)"/>
<circle cx="1220" cy="90" r="155" fill="#22D3EE" opacity="0.11" filter="url(#glow)"/>
<circle cx="1090" cy="430" r="190" fill="#3B82F6" opacity="0.15" filter="url(#glow)"/>''',
        round_rect(70, 60, 202, 38, "#173B65", radius=19, stroke="#2D5A88"),
        text(171, 86, "RESEARCH × SYSTEMS", size=15, weight=700, fill=CYAN, anchor="middle"),
        text(70, 166, "C3-FaRNet", size=66, weight=760, fill=WHITE),
        text(70, 214, "Factor-aware road surface intelligence", size=28, weight=500, fill="#CDE6FF"),
        text(70, 263, "Physics-guided representation learning for coupled road states", size=20, fill="#9FB8D5"),
        text(70, 298, "with auditable PyTorch/CUDA evaluation.", size=20, fill="#9FB8D5"),
    ]

    chips = [("27 classes", 70, 354), ("958,941 train frames", 222, 354), ("49,500 full-test images", 478, 354)]
    for label, x, y in chips:
        w = 132 if label == "27 classes" else (236 if "958" in label else 246)
        body.append(round_rect(x, y, w, 48, "#102B4D", radius=14, stroke="#2B527C"))
        body.append(text(x + w / 2, y + 31, label, size=16, weight=650, fill="#D9ECFF", anchor="middle"))

    # Right-hand evidence-to-decision motif.
    body.extend([
        text(1020, 74, "COUPLED EVIDENCE", size=16, weight=750, fill="#8BB7E4", anchor="middle"),
        round_rect(835, 105, 230, 68, "#11395C", radius=16, stroke="#2C6B8F"),
        round_rect(835, 195, 230, 68, "#11395C", radius=16, stroke="#2C6B8F"),
        round_rect(835, 285, 230, 68, "#11395C", radius=16, stroke="#2C6B8F"),
        text(950, 133, "Condition", size=18, weight=700, fill=WHITE, anchor="middle"),
        text(950, 157, "dry · wet · water · snow · ice", size=13, fill="#9FC6E8", anchor="middle"),
        text(950, 223, "Material", size=18, weight=700, fill=WHITE, anchor="middle"),
        text(950, 247, "asphalt · concrete · mud · gravel", size=13, fill="#9FC6E8", anchor="middle"),
        text(950, 313, "Roughness", size=18, weight=700, fill=WHITE, anchor="middle"),
        text(950, 337, "smooth · slight · severe", size=13, fill="#9FC6E8", anchor="middle"),
        line(1065, 139, 1120, 229, stroke=CYAN, width=3),
        line(1065, 229, 1120, 229, stroke=CYAN, width=3),
        line(1065, 319, 1120, 229, stroke=CYAN, width=3),
        round_rect(1120, 169, 208, 120, "#153F70", radius=24, stroke=BLUE, stroke_width=2),
        text(1224, 213, "Coupled", size=23, weight=750, fill=WHITE, anchor="middle"),
        text(1224, 244, "factor reasoning", size=23, weight=750, fill=WHITE, anchor="middle"),
        text(1224, 271, "→ 27 road states", size=15, weight=600, fill=CYAN, anchor="middle"),
        round_rect(835, 408, 493, 42, "#0B2847", radius=13, stroke="#265782"),
        text(1081, 435, "Representation · Calibration · Reproducibility", size=16, weight=650, fill="#C6E4FF", anchor="middle"),
    ])
    write_svg("hero", 1400, 500, "\n  ".join(body), dark=True)


def build_architecture() -> None:
    body: list[str] = [
        text(70, 64, "C3-FaRNet: evidence is structured before the final decision", size=31, weight=760),
        text(70, 96, "Verified S7 implementation · arrows show data flow, dashed arrows show gated corrections", size=16, fill=MUTED),
    ]
    # Connection lines are emitted before nodes so they remain behind text.
    body.extend([
        line(226, 350, 295, 350, stroke=BLUE, width=4),
        line(585, 350, 650, 350, stroke=BLUE, width=4),
        line(935, 350, 1000, 350, stroke=BLUE, width=4),
        line(1265, 350, 1330, 350, stroke=BLUE, width=4),
        line(440, 238, 440, 210, stroke=CYAN, width=3),
        line(440, 490, 440, 518, stroke=CYAN, width=3),
        line(770, 235, 770, 205, stroke=ORANGE, width=3),
        line(770, 495, 770, 525, stroke=ORANGE, width=3),
        line(1128, 235, 1128, 205, stroke="#8B5CF6", width=3, dash="8 7"),
        line(1128, 495, 1128, 525, stroke="#8B5CF6", width=3, dash="8 7"),
    ])
    # Main path.
    body.extend([
        round_rect(60, 284, 166, 132, LIGHT, radius=22, stroke=LINE, stroke_width=2),
        text(143, 322, "Road image", size=21, weight=750, anchor="middle"),
        text(143, 351, "RGB · 192 × 192", size=16, fill=MUTED, anchor="middle"),
        text(143, 379, "historical protocol", size=14, fill=MUTED, anchor="middle"),
        round_rect(295, 238, 290, 224, "#EFF6FF", radius=24, stroke="#9CC2F8", stroke_width=2),
        text(440, 282, "Visual carrier", size=23, weight=750, anchor="middle"),
        text(440, 313, "ConvNeXt-Tiny feature hierarchy", size=16, fill=MUTED, anchor="middle"),
        text(440, 342, "+ gated tensor-coupling stem", size=16, fill=BLUE, weight=650, anchor="middle"),
        text(440, 382, "shape · texture · spatial context", size=15, fill=MUTED, anchor="middle"),
        text(440, 410, "early/mid feature conditioning", size=15, fill=MUTED, anchor="middle"),
        round_rect(650, 235, 285, 230, "#ECFEFF", radius=24, stroke="#80DDEA", stroke_width=2),
        text(792, 277, "Factor evidence", size=23, weight=750, anchor="middle"),
        text(792, 313, "condition", size=17, weight=700, fill="#087F8C", anchor="middle"),
        text(792, 342, "material", size=17, weight=700, fill="#087F8C", anchor="middle"),
        text(792, 371, "roughness", size=17, weight=700, fill="#087F8C", anchor="middle"),
        text(792, 415, "single + pair + triple factors", size=15, fill=MUTED, anchor="middle"),
        round_rect(1000, 235, 265, 230, "#FFF7ED", radius=24, stroke="#FDBA74", stroke_width=2),
        text(1132, 277, "Coupled decision", size=23, weight=750, anchor="middle"),
        text(1132, 316, "factor tensor score", size=16, weight=650, fill="#B45309", anchor="middle"),
        text(1132, 347, "hard-pair error gate", size=16, weight=650, fill="#B45309", anchor="middle"),
        text(1132, 378, "calibrated correction", size=16, weight=650, fill="#B45309", anchor="middle"),
        text(1132, 419, "boundary-aware, not global rewrite", size=14, fill=MUTED, anchor="middle"),
        round_rect(1330, 284, 130, 132, "#EAF3FF", radius=22, stroke=BLUE, stroke_width=2),
        text(1395, 327, "27-class", size=21, weight=760, fill=BLUE, anchor="middle"),
        text(1395, 357, "posterior", size=21, weight=760, fill=BLUE, anchor="middle"),
        text(1395, 388, "single crop", size=14, fill=MUTED, anchor="middle"),
    ])
    # Supporting evidence/training cards.
    body.extend([
        round_rect(295, 128, 290, 82, "#F0FDFA", radius=18, stroke="#99F6E4"),
        text(440, 160, "Physics evidence", size=18, weight=750, fill="#0F766E", anchor="middle"),
        text(440, 187, "wetness · glare · texture loss", size=14, fill=MUTED, anchor="middle"),
        round_rect(295, 518, 290, 92, "#F0FDFA", radius=18, stroke="#99F6E4"),
        text(440, 550, "Local + semantic fields", size=18, weight=750, fill="#0F766E", anchor="middle"),
        text(440, 578, "regional evidence and cue selection", size=14, fill=MUTED, anchor="middle"),
        round_rect(650, 125, 285, 80, "#FFF7ED", radius=18, stroke="#FED7AA"),
        text(792, 157, "Structured label graph", size=18, weight=750, fill="#B45309", anchor="middle"),
        text(792, 183, "legal states + hard boundaries", size=14, fill=MUTED, anchor="middle"),
        round_rect(650, 525, 285, 92, "#FFF7ED", radius=18, stroke="#FED7AA"),
        text(792, 557, "Selective optimization", size=18, weight=750, fill="#B45309", anchor="middle"),
        text(792, 585, "1.09M trainable / 32.49M total", size=14, fill=MUTED, anchor="middle"),
        round_rect(1000, 123, 265, 82, "#F5F3FF", radius=18, stroke="#C4B5FD"),
        text(1132, 155, "Anchor protection", size=18, weight=750, fill="#6D28D9", anchor="middle"),
        text(1132, 182, "consistency + no-flip constraints", size=14, fill=MUTED, anchor="middle"),
        round_rect(1000, 525, 265, 92, "#F5F3FF", radius=18, stroke="#C4B5FD"),
        text(1132, 557, "Reliable-source router", size=18, weight=750, fill="#6D28D9", anchor="middle"),
        text(1132, 585, "strict top-k and margin gating", size=14, fill=MUTED, anchor="middle"),
        round_rect(60, 664, 1400, 64, "#F8FAFC", radius=18, stroke=LINE),
        text(760, 704, "Research boundary: this is visual road-state / friction-affordance estimation, not direct tire–road friction measurement.", size=16, weight=600, fill=MUTED, anchor="middle"),
    ])
    write_svg("architecture", 1520, 770, "\n  ".join(body))


def build_results(metrics: dict[str, object]) -> None:
    cards = [
        ("Top-1", float(metrics["top1"]), BLUE),
        ("Macro-F1", float(metrics["macro_f1"]), CYAN),
        ("Weighted F1", float(metrics["weighted_f1"]), GREEN),
        ("Weakest-class F1", float(metrics["water_concrete_slight_f1"]), ORANGE),
    ]
    body: list[str] = [
        '''<defs><linearGradient id="res-bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#07152D"/><stop offset="1" stop-color="#153D67"/></linearGradient></defs><rect width="1400" height="620" fill="url(#res-bg)"/>''',
        text(64, 68, "Verified full-test evidence", size=32, weight=760, fill=WHITE),
        text(64, 101, "Self-contained C3-FaRNet S7 checkpoint · 49,500 images · 27 classes", size=17, fill="#AFC8E1"),
    ]
    for idx, (label, value, color) in enumerate(cards):
        x = 64 + idx * 327
        body.append(round_rect(x, 142, 292, 160, "#0E2A4A", radius=22, stroke="#2B527C"))
        body.append(text(x + 24, 180, label, size=17, weight=650, fill="#AAC7E3"))
        body.append(text(x + 24, 241, f"{value * 100:.3f}%", size=40, weight=780, fill=WHITE))
        body.append(round_rect(x + 24, 264, 244, 8, "#183D63", radius=4))
        body.append(round_rect(x + 24, 264, 244 * value, 8, color, radius=4))

    factor_cards = [
        ("Condition", float(metrics["friction_acc"]), "dry / wet / water / snow / ice", BLUE),
        ("Material", float(metrics["material_acc"]), "asphalt / concrete / mud / gravel", CYAN),
        ("Roughness", float(metrics["roughness_acc"]), "smooth / slight / severe", ORANGE),
    ]
    body.append(text(64, 359, "Factor-level diagnosis", size=23, weight=720, fill=WHITE))
    for idx, (label, value, desc, color) in enumerate(factor_cards):
        x = 64 + idx * 435
        body.append(round_rect(x, 388, 400, 132, "#102C4C", radius=18, stroke="#2B527C"))
        body.append(text(x + 22, 425, label, size=18, weight=700, fill=WHITE))
        body.append(text(x + 378, 425, f"{value * 100:.2f}%", size=21, weight=750, fill=color, anchor="end"))
        body.append(text(x + 22, 458, desc, size=13, fill="#AFC8E1"))
        body.append(round_rect(x + 22, 484, 356, 8, "#183D63", radius=4))
        body.append(round_rect(x + 22, 484, 356 * value, 8, color, radius=4))

    body.extend([
        line(64, 558, 1336, 558, stroke="#2D5074", width=1),
        text(64, 592, "Protocol", size=15, weight=700, fill="#9DB9D5"),
        text(145, 592, "single model · single crop · historical 192×192 letterbox protocol", size=15, fill="#D2E5F6"),
        text(1336, 592, "No cross-protocol SOTA claim", size=15, weight=650, fill="#FBBF84", anchor="end"),
    ])
    write_svg("results-overview", 1400, 620, "\n  ".join(body), dark=True)


def build_per_class(rows: list[dict[str, str]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["f1"]))
    width, height = 1440, 1150
    x0, x1 = 370, 1360
    y0, row_h = 140, 34
    body: list[str] = [
        text(60, 58, "Per-class F1: the aggregate score does not hide the weak boundary classes", size=29, weight=760),
        text(60, 91, "Sorted ascending · values come from results/current_best_s7/per_class_metrics.csv", size=15, fill=MUTED),
    ]
    for pct in (0.6, 0.7, 0.8, 0.9, 1.0):
        x = x0 + (pct - 0.6) / 0.4 * (x1 - x0)
        body.append(line(x, y0 - 20, x, y0 + row_h * len(ordered), stroke=LINE, width=1))
        body.append(text(x, y0 - 30, f"{pct * 100:.0f}%", size=13, fill=MUTED, anchor="middle"))

    for idx, row in enumerate(ordered):
        y = y0 + idx * row_h
        score = float(row["f1"])
        score_clip = max(0.6, min(1.0, score))
        bar_w = (score_clip - 0.6) / 0.4 * (x1 - x0)
        if idx < 4:
            color = ORANGE
        elif score >= 0.95:
            color = GREEN
        else:
            color = BLUE
        if idx % 2 == 0:
            body.append(round_rect(48, y - 21, 1334, 30, "#F8FAFC", radius=5))
        body.append(text(x0 - 18, y, row["class"], size=13, weight=570, anchor="end"))
        body.append(round_rect(x0, y - 17, max(2, bar_w), 19, color, radius=5, opacity=0.88))
        body.append(text(min(x0 + bar_w + 10, 1390), y, f"{score * 100:.2f}%", size=13, weight=650, fill=color))

    body.extend([
        round_rect(60, 1080, 1320, 44, "#FFF7ED", radius=12, stroke="#FED7AA"),
        text(720, 1108, "Measured bottleneck: water/wet concrete slight–severe boundaries; current research targets this failure mode.", size=15, weight=650, fill="#9A3412", anchor="middle"),
    ])
    write_svg("per-class-f1", width, height, "\n  ".join(body))


def heat_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    # Interpolate from pale blue to deep blue, with stronger nonlinear contrast.
    value = math.sqrt(value)
    lo = (239, 246, 255)
    hi = (24, 92, 171)
    rgb = tuple(round(lo[i] + (hi[i] - lo[i]) * value) for i in range(3))
    return "#" + "".join(f"{part:02X}" for part in rgb)


def build_confusion(labels: list[str], matrix: list[list[int]]) -> None:
    width, height = 1700, 1610
    x0, y0, cell = 435, 180, 40
    body: list[str] = [
        text(60, 58, "Normalized 27-class confusion matrix", size=30, weight=760),
        text(60, 92, "Rows are ground truth; columns are predictions · diagonal concentration shows class separability", size=15, fill=MUTED),
        text(260, y0 + 27 * cell / 2, "GROUND TRUTH", size=15, weight=750, fill=MUTED, anchor="middle"),
        text(x0 + 27 * cell / 2, 145, "PREDICTION", size=15, weight=750, fill=MUTED, anchor="middle"),
    ]
    for idx, label in enumerate(labels):
        cx = x0 + idx * cell + cell / 2
        cy = y0 + idx * cell + cell / 2
        body.append(
            f'<text x="{cx:.1f}" y="{y0 - 12:.1f}" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="10" fill="{MUTED}" transform="rotate(-58 {cx:.1f} {y0 - 12:.1f})">{escape(label)}</text>'
        )
        body.append(text(x0 - 14, cy + 4, label, size=10, fill=MUTED, anchor="end"))

    for r_idx, row in enumerate(matrix):
        total = sum(row) or 1
        for c_idx, count in enumerate(row):
            value = count / total
            x = x0 + c_idx * cell
            y = y0 + r_idx * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" fill="{heat_color(value)}"/>')
            if r_idx == c_idx:
                body.append(text(x + cell / 2, y + 25, f"{value * 100:.0f}", size=9, weight=700,
                                 fill=WHITE if value > 0.70 else INK, anchor="middle"))

    # Compact legend and interpretation.
    legend_y = 1320
    body.append(text(435, legend_y, "Row-normalized recall", size=14, weight=700, fill=MUTED))
    for idx in range(101):
        body.append(f'<rect x="{435 + idx * 5}" y="{legend_y + 18}" width="5" height="18" fill="{heat_color(idx / 100)}"/>')
    body.append(text(435, legend_y + 57, "0%", size=12, fill=MUTED))
    body.append(text(940, legend_y + 57, "100%", size=12, fill=MUTED, anchor="end"))
    body.append(round_rect(1040, 1292, 540, 128, "#F8FAFC", radius=18, stroke=LINE))
    body.append(text(1072, 1330, "Reading the plot", size=18, weight=750))
    body.append(text(1072, 1362, "• Strong diagonal: correct class separation", size=14, fill=MUTED))
    body.append(text(1072, 1390, "• Off-diagonal blocks: coupled boundary errors", size=14, fill=MUTED))
    write_svg("confusion-matrix", width, height, "\n  ".join(body))


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    per_class = load_per_class()
    labels, matrix = load_confusion()
    build_hero()
    build_architecture()
    build_results(metrics)
    build_per_class(per_class)
    build_confusion(labels, matrix)
    print(f"Built 5 SVG assets in {ASSETS}")


if __name__ == "__main__":
    main()
