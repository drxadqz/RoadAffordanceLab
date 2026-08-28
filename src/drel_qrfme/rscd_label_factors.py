"""RSCD-27 固定类别表与因素分解。

普通路面类可写成 friction × material × roughness，例如
``wet_asphalt_slight``；snow/ice 没有可定义的 material/roughness。这里固定
类别顺序而不从 CSV 动态推断，确保另一台电脑的 logit 第 c 维仍对应同一类。
修改 ``RSCD_CLASS_LABELS`` 顺序会让旧 checkpoint 分类头语义错位，禁止修改。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


IGNORE_INDEX = -100

FRICTION_LABELS = ("dry", "wet", "water", "fresh_snow", "melted_snow", "ice")
MATERIAL_LABELS = ("none", "asphalt", "concrete", "mud", "gravel")
ROUGHNESS_LABELS = ("none", "smooth", "slight", "severe")
FACTOR_AXES = ("friction", "material", "roughness")
FACTOR_LABELS = {
    "friction": FRICTION_LABELS,
    "material": MATERIAL_LABELS,
    "roughness": ROUGHNESS_LABELS,
}

# The public RSCD classifier is a closed 27-way task.  Keep this order aligned
# with the released result files and historical checkpoints instead of deriving
# a class map from a train/validation/test union (which can silently turn a
# malformed manifest row into a 28th class).
RSCD_CLASS_LABELS = (
    "dry_asphalt_severe",
    "dry_asphalt_slight",
    "dry_asphalt_smooth",
    "dry_concrete_severe",
    "dry_concrete_slight",
    "dry_concrete_smooth",
    "dry_gravel",
    "dry_mud",
    "fresh_snow",
    "ice",
    "melted_snow",
    "water_asphalt_severe",
    "water_asphalt_slight",
    "water_asphalt_smooth",
    "water_concrete_severe",
    "water_concrete_slight",
    "water_concrete_smooth",
    "water_gravel",
    "water_mud",
    "wet_asphalt_severe",
    "wet_asphalt_slight",
    "wet_asphalt_smooth",
    "wet_concrete_severe",
    "wet_concrete_slight",
    "wet_concrete_smooth",
    "wet_gravel",
    "wet_mud",
)

_CLASS_LABEL_ALIASES = {
    "dry_dirt_mud": "dry_mud",
    "wet_dirt_mud": "wet_mud",
    "water_dirt_mud": "water_mud",
}
_RSCD_CLASS_LABEL_SET = frozenset(RSCD_CLASS_LABELS)


@dataclass(frozen=True)
class FactorTriple:
    """一个类别在摩擦状态、材质、粗糙度三个轴上的整数索引。"""
    friction: int
    material: int
    roughness: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (int(self.friction), int(self.material), int(self.roughness))


@dataclass(frozen=True)
class HardPair:
    left: int
    right: int
    axis: str
    boundary: str


@dataclass(frozen=True)
class RSCDFactorSpec:
    """模型使用的类别—因素查找表、有效掩码及困难类别对集合。"""
    class_to_idx: dict[str, int]
    class_to_factor: torch.Tensor
    valid_class_mask: torch.Tensor
    valid_factor_mask: torch.Tensor
    valid_tensor_mask: torch.Tensor
    class_index_grid: torch.Tensor
    hard_pairs: tuple[HardPair, ...]

    @property
    def num_classes(self) -> int:
        return int(self.class_to_factor.shape[0])


def canonical_class_label(value: Any) -> str:
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _CLASS_LABEL_ALIASES.get(label, label)


def resolve_rscd_class_label(value: Any, image_path: Any | None = None) -> str:
    """Resolve malformed flat-split labels from an unambiguous class suffix.

    Some official flat-split filenames contain a transition prefix before the
    actual class suffix, for example ``...wet-dry-asphalt-smooth.jpg``. A
    generated manifest can incorrectly reduce such a row to the invalid label
    ``wet``. Valid RSCD-27 manifest labels remain authoritative; filename
    recovery is attempted only when the supplied label is outside the closed
    set, and only for a unique token-aligned suffix.
    """

    label = canonical_class_label(value)
    if label in _RSCD_CLASS_LABEL_SET or image_path is None:
        return label

    stem = canonical_class_label(Path(str(image_path)).stem)
    suffix_to_label = {
        **{name: name for name in RSCD_CLASS_LABELS},
        **{
            canonical_class_label(alias): canonical_class_label(target)
            for alias, target in _CLASS_LABEL_ALIASES.items()
        },
    }
    matches = [
        (suffix, target)
        for suffix, target in suffix_to_label.items()
        if stem == suffix or stem.endswith(f"_{suffix}")
    ]
    if not matches:
        return label
    matches.sort(key=lambda item: len(item[0]), reverse=True)
    longest_length = len(matches[0][0])
    longest_targets = {
        target for suffix, target in matches if len(suffix) == longest_length
    }
    if len(longest_targets) != 1:
        return label
    return longest_targets.pop()


def fixed_rscd_class_map() -> dict[str, int]:
    """Return the checkpoint-compatible closed-set RSCD-27 class map."""

    return {name: idx for idx, name in enumerate(RSCD_CLASS_LABELS)}


def factor_index(axis: str, value: str | None) -> int:
    labels = FACTOR_LABELS[axis]
    if value is None:
        value = "none"
    return labels.index(value) if value in labels else IGNORE_INDEX


def parse_rscd_label(name: str) -> FactorTriple:
    """Parse one RSCD class name into friction/material/roughness factors."""

    label = canonical_class_label(name)
    if label in {"fresh_snow", "melted_snow", "ice"}:
        return FactorTriple(
            factor_index("friction", label),
            factor_index("material", "none"),
            factor_index("roughness", "none"),
        )
    parts = label.split("_")
    friction = parts[0] if len(parts) >= 1 else None
    material = parts[1] if len(parts) >= 2 else "none"
    roughness = parts[2] if len(parts) >= 3 else "none"
    return FactorTriple(
        factor_index("friction", friction),
        factor_index("material", material),
        factor_index("roughness", roughness),
    )


def factor_name(axis: str, idx: int) -> str:
    if idx < 0:
        return "invalid"
    return FACTOR_LABELS[axis][int(idx)]


def boundary_name(axis: str, left: FactorTriple, right: FactorTriple) -> str:
    li = left.as_tuple()[FACTOR_AXES.index(axis)]
    ri = right.as_tuple()[FACTOR_AXES.index(axis)]
    a = factor_name(axis, li)
    b = factor_name(axis, ri)
    if axis == "friction" and {a, b} == {"wet", "water"}:
        return "wet_water"
    if axis == "friction" and {a, b} == {"dry", "wet"}:
        return "dry_wet"
    if axis == "friction" and {a, b} == {"fresh_snow", "melted_snow"}:
        return "snow_phase"
    if axis == "friction" and "ice" in {a, b}:
        return "snow_ice"
    if axis == "material" and {a, b} == {"asphalt", "concrete"}:
        return "asphalt_concrete"
    if axis == "material" and {a, b} == {"mud", "gravel"}:
        return "mud_gravel"
    if axis == "roughness" and {a, b} <= {"smooth", "slight", "severe"}:
        return "roughness"
    return f"{axis}_{a}_vs_{b}"


def build_rscd_factor_spec(class_to_idx: dict[str, int]) -> RSCDFactorSpec:
    """Build class-factor maps, valid combination mask, and hard-pair graph."""

    idx_to_class = {idx: canonical_class_label(name) for name, idx in class_to_idx.items()}
    num_classes = len(idx_to_class)
    class_to_factor = torch.full((num_classes, 3), IGNORE_INDEX, dtype=torch.long)
    valid_factor_mask = torch.zeros((num_classes, 3), dtype=torch.bool)
    class_index_grid = torch.full(
        (len(FRICTION_LABELS), len(MATERIAL_LABELS), len(ROUGHNESS_LABELS)),
        -1,
        dtype=torch.long,
    )
    triples: dict[int, FactorTriple] = {}
    for idx in range(num_classes):
        triple = parse_rscd_label(idx_to_class[idx])
        triples[idx] = triple
        values = torch.tensor(triple.as_tuple(), dtype=torch.long)
        class_to_factor[idx] = values
        valid_factor_mask[idx] = values.ge(0)
        if bool(values.ge(0).all()):
            class_index_grid[triple.friction, triple.material, triple.roughness] = int(idx)
    valid_tensor_mask = class_index_grid.ge(0)

    hard_pairs: list[HardPair] = []
    for i in range(num_classes):
        a = triples[i]
        av = a.as_tuple()
        if min(av) < 0:
            continue
        for j in range(i + 1, num_classes):
            b = triples[j]
            bv = b.as_tuple()
            if min(bv) < 0:
                continue
            diff_axes = [axis for axis, x, y in zip(FACTOR_AXES, av, bv, strict=True) if int(x) != int(y)]
            if len(diff_axes) != 1:
                continue
            axis = diff_axes[0]
            hard_pairs.append(HardPair(i, j, axis, boundary_name(axis, a, b)))

    return RSCDFactorSpec(
        class_to_idx={canonical_class_label(k): int(v) for k, v in class_to_idx.items()},
        class_to_factor=class_to_factor,
        valid_class_mask=class_to_factor.ge(0).all(dim=1),
        valid_factor_mask=valid_factor_mask,
        valid_tensor_mask=valid_tensor_mask,
        class_index_grid=class_index_grid,
        hard_pairs=tuple(hard_pairs),
    )


def class_factor_targets(labels: torch.Tensor, spec: RSCDFactorSpec, device: torch.device) -> dict[str, torch.Tensor]:
    factors = spec.class_to_factor.to(device=device).index_select(0, labels)
    return {
        "friction": factors[:, 0],
        "material": factors[:, 1],
        "roughness": factors[:, 2],
    }


def sanity_summary(class_to_idx: dict[str, int]) -> str:
    spec = build_rscd_factor_spec(class_to_idx)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    lines = ["RSCD factor parsing sanity:"]
    for idx in range(spec.num_classes):
        f, m, r = spec.class_to_factor[idx].tolist()
        lines.append(
            f"{idx:02d} {idx_to_class[idx]} -> "
            f"({factor_name('friction', f)}, {factor_name('material', m)}, {factor_name('roughness', r)})"
        )
    by_axis: dict[str, int] = {axis: 0 for axis in FACTOR_AXES}
    by_boundary: dict[str, int] = {}
    for pair in spec.hard_pairs:
        by_axis[pair.axis] += 1
        by_boundary[pair.boundary] = by_boundary.get(pair.boundary, 0) + 1
    lines.append(f"hard_pairs={len(spec.hard_pairs)} by_axis={by_axis} by_boundary={by_boundary}")
    return "\n".join(lines)
