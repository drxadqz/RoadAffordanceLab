"""Protected Quotient Spatial Organizer for ARCQ Q3.

PQSO preserves all ten Q3 response-group identities and all four centred
orientation-composition coordinates.  It does not pool, sort, or claim a
physical frequency ordering for the group axis.  The product-difference
candidate and additive control have exactly the same trainable tensors and
initialization; only their fixed local relation operator differs.
"""

from __future__ import annotations

import math
from typing import Any, Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_DEFAULT_Q3_GROUPS: Final[int] = 10
_DEFAULT_ORIENTATIONS: Final[int] = 4


def _positive_int(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def trainable_state_signature(
    module: nn.Module,
) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    return tuple(
        (name, tuple(int(size) for size in parameter.shape), parameter.numel())
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )


class ProtectedQuotientSpatialSource(nn.Module):
    """Validate and flatten BxGxOxHxW centred composition without information loss."""

    def __init__(
        self,
        *,
        ordered_groups: int = _DEFAULT_Q3_GROUPS,
        composition_parts: int = _DEFAULT_ORIENTATIONS,
        centered_tolerance: float = 1.0e-4,
        validate_centered_input: bool = True,
    ) -> None:
        super().__init__()
        self.ordered_groups = _positive_int(ordered_groups, name="ordered_groups")
        self.composition_parts = _positive_int(
            composition_parts,
            name="composition_parts",
        )
        centered_tolerance = float(centered_tolerance)
        if not math.isfinite(centered_tolerance) or centered_tolerance <= 0.0:
            raise ValueError("centered_tolerance must be finite and positive")
        self.centered_tolerance = centered_tolerance
        self.validate_centered_input = bool(validate_centered_input)
        self.output_channels = self.ordered_groups * self.composition_parts

    def forward(self, centered_composition: Tensor) -> Tensor:
        if centered_composition.ndim != 5:
            raise ValueError(
                "centered composition must have shape BxGxOxHxW, got "
                f"{tuple(centered_composition.shape)}"
            )
        if int(centered_composition.shape[1]) != self.ordered_groups:
            raise ValueError(
                "ordered group width mismatch: expected "
                f"{self.ordered_groups}, got {int(centered_composition.shape[1])}"
            )
        if int(centered_composition.shape[2]) != self.composition_parts:
            raise ValueError(
                "composition width mismatch: expected "
                f"{self.composition_parts}, got {int(centered_composition.shape[2])}"
            )
        if not centered_composition.is_floating_point():
            raise TypeError("centered composition must be floating point")
        if self.validate_centered_input:
            if not torch.isfinite(centered_composition.float()).all().item():
                raise ValueError("centered composition must contain only finite values")
            centered_error = (
                centered_composition.float().sum(dim=2).abs().amax().item()
            )
            if centered_error > self.centered_tolerance:
                raise ValueError(
                    "PQSO requires centered composition c=p-1/O with sum_o c_o=0; "
                    f"observed maximum absolute sum {centered_error:.6g}"
                )
        return centered_composition.flatten(1, 2).contiguous()


def zero_anchored_contraction(feature: Tensor) -> Tensor:
    """Contract, but never amplify, weak quotient evidence."""

    if not feature.is_floating_point():
        raise TypeError("feature must be floating point")
    return feature * torch.rsqrt(1.0 + feature.square().mean(dim=1, keepdim=True))


def _detached_per_sample_rms(feature: Tensor) -> Tensor:
    with torch.no_grad():
        dims = tuple(range(1, feature.ndim))
        return feature.detach().square().mean(dim=dims, dtype=torch.float32).sqrt()


class ChannelwiseConvexNeighbourhood(nn.Module):
    """Learnable non-negative, unit-mass depthwise neighbourhood averaging."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channels = _positive_int(channels, name="channels")
        self.kernel_size = _positive_int(kernel_size, name="kernel_size")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        radius = self.kernel_size // 2
        coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        sigma = max(float(self.kernel_size) / 3.0, 1.0)
        gaussian = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
        gaussian = gaussian / gaussian.sum()
        self.kernel_logits = nn.Parameter(
            gaussian.log().reshape(1, 1, self.kernel_size, self.kernel_size).repeat(
                self.channels, 1, 1, 1
            )
        )

    def normalized_kernel(self) -> Tensor:
        return torch.softmax(self.kernel_logits.flatten(1), dim=1).reshape_as(
            self.kernel_logits
        )

    def forward(self, feature: Tensor) -> Tensor:
        if feature.ndim != 4 or int(feature.shape[1]) != self.channels:
            raise ValueError(
                f"feature must have shape Bx{self.channels}xHxW, got "
                f"{tuple(feature.shape)}"
            )
        radius = self.kernel_size // 2
        padded = F.pad(feature, (radius, radius, radius, radius), mode="replicate")
        return F.conv2d(
            padded,
            self.normalized_kernel().to(dtype=feature.dtype),
            groups=self.channels,
        )


class SharedQuotientNeighbourhoodOrganizer(nn.Module):
    """Exact trainable graph shared by additive and product PQSO arms."""

    VALID_MODES: Final[frozenset[str]] = frozenset(
        {"product_difference", "additive_control"}
    )

    def __init__(
        self,
        *,
        source_channels: int,
        hidden_channels: int,
        out_channels: int,
        relation_mode: str,
        residual_max: float = 0.25,
    ) -> None:
        super().__init__()
        relation_mode = str(relation_mode).strip().lower()
        if relation_mode not in self.VALID_MODES:
            raise ValueError(
                f"relation_mode must be one of {sorted(self.VALID_MODES)}, "
                f"got {relation_mode!r}"
            )
        self.relation_mode = relation_mode
        self.source_channels = _positive_int(source_channels, name="source_channels")
        self.hidden_channels = _positive_int(hidden_channels, name="hidden_channels")
        self.out_channels = _positive_int(out_channels, name="out_channels")
        residual_max = float(residual_max)
        if not math.isfinite(residual_max) or not 0.0 < residual_max <= 1.0:
            raise ValueError("residual_max must be finite and lie in (0, 1]")
        self.residual_max = residual_max
        self.register_buffer(
            "_relation_mode_code",
            torch.tensor(
                1 if relation_mode == "product_difference" else 0,
                dtype=torch.int64,
            ),
        )

        self.center = nn.Conv2d(
            self.source_channels, self.hidden_channels, kernel_size=1, bias=False
        )
        self.neighbour3 = ChannelwiseConvexNeighbourhood(
            self.hidden_channels,
            kernel_size=3,
        )
        self.neighbour5 = ChannelwiseConvexNeighbourhood(
            self.hidden_channels,
            kernel_size=5,
        )
        self.relation_mix = nn.Conv2d(
            5 * self.hidden_channels,
            self.hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.local_refine = ChannelwiseConvexNeighbourhood(
            self.hidden_channels,
            kernel_size=3,
        )
        self.writer = nn.Conv2d(
            self.hidden_channels, self.out_channels, kernel_size=1, bias=False
        )
        nn.init.zeros_(self.writer.weight)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        state_name = prefix + "_relation_mode_code"
        supplied = state_dict.get(state_name)
        if supplied is not None:
            valid = (
                torch.is_tensor(supplied)
                and supplied.dtype == torch.int64
                and supplied.shape == torch.Size([])
                and torch.equal(supplied.detach().cpu(), self._relation_mode_code.cpu())
            )
            if not valid:
                error_msgs.append(
                    f'Invalid PQSO relation mode buffer "{state_name}": '
                    "a checkpoint from the other causal arm cannot be loaded."
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def contract_source(self, source: Tensor) -> Tensor:
        if source.ndim != 4 or int(source.shape[1]) != self.source_channels:
            raise ValueError(
                f"source must have shape Bx{self.source_channels}xHxW, got "
                f"{tuple(source.shape)}"
            )
        return zero_anchored_contraction(source)

    def relation_parts(
        self,
        centre: Tensor,
        neighbour3: Tensor,
        neighbour5: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if centre.shape != neighbour3.shape or centre.shape != neighbour5.shape:
            raise ValueError("centre and neighbourhood tensors must have equal shapes")
        difference3 = 0.5 * (centre - neighbour3)
        difference5 = 0.5 * (centre - neighbour5)
        if self.relation_mode == "product_difference":
            return (
                centre,
                centre * neighbour3,
                difference3,
                centre * neighbour5,
                difference5,
            )
        return (
            centre,
            0.5 * (centre + neighbour3),
            difference3,
            0.5 * (centre + neighbour5),
            difference5,
        )

    def relation_map(self, source: Tensor) -> Tensor:
        centre = self.center(self.contract_source(source))
        neighbour3 = self.neighbour3(centre)
        neighbour5 = self.neighbour5(centre)
        relation = F.gelu(
            self.relation_mix(
                torch.cat(
                    self.relation_parts(centre, neighbour3, neighbour5),
                    dim=1,
                )
            )
        )
        return F.gelu(self.local_refine(relation))

    def forward(self, source: Tensor) -> Tensor:
        return self.residual_max * torch.tanh(self.writer(self.relation_map(source)))

    def forward_with_summaries(
        self,
        source: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        relation = self.relation_map(source)
        writer_pre_tanh = self.writer(relation)
        residual = self.residual_max * torch.tanh(writer_pre_tanh)
        with torch.no_grad():
            saturation_threshold = math.atanh(0.95)
            summaries = {
                "relation_rms": _detached_per_sample_rms(relation),
                "writer_pre_tanh_rms": _detached_per_sample_rms(writer_pre_tanh),
                "saturation_fraction": (
                    writer_pre_tanh.detach().abs() >= saturation_threshold
                ).float().mean(dim=(1, 2, 3)),
            }
        return residual, summaries


class _ProtectedQuotientSpatialBase(nn.Module):
    def __init__(
        self,
        *,
        relation_mode: str,
        ordered_groups: int = _DEFAULT_Q3_GROUPS,
        composition_parts: int = _DEFAULT_ORIENTATIONS,
        hidden_channels: int = 32,
        out_channels: int = 192,
        residual_max: float = 0.25,
        centered_tolerance: float = 1.0e-4,
        validate_centered_input: bool = True,
    ) -> None:
        super().__init__()
        self.relation_mode = relation_mode
        self.source = ProtectedQuotientSpatialSource(
            ordered_groups=ordered_groups,
            composition_parts=composition_parts,
            centered_tolerance=centered_tolerance,
            validate_centered_input=validate_centered_input,
        )
        self.organizer = SharedQuotientNeighbourhoodOrganizer(
            source_channels=self.source.output_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            relation_mode=relation_mode,
            residual_max=residual_max,
        )

    @property
    def output_channels(self) -> int:
        return self.organizer.out_channels

    def source_map(self, centered_composition: Tensor) -> Tensor:
        return self.source(centered_composition)

    def forward(self, centered_composition: Tensor) -> Tensor:
        return self.organizer(self.source_map(centered_composition))

    def forward_with_summaries(
        self,
        centered_composition: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        source = self.source_map(centered_composition)
        residual, summaries = self.organizer.forward_with_summaries(source)
        summaries = dict(summaries)
        summaries["source_rms"] = _detached_per_sample_rms(source)
        summaries["source_abs_max"] = source.detach().abs().amax(
            dim=(1, 2, 3)
        ).float()
        summaries["residual_rms"] = _detached_per_sample_rms(residual)
        return residual, summaries


class ProtectedQuotientSpatialOrganizer(_ProtectedQuotientSpatialBase):
    """PQSO candidate with explicit local products and differences."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(relation_mode="product_difference", **kwargs)


class AdditiveProtectedQuotientControl(_ProtectedQuotientSpatialBase):
    """Exact-match additive control for PQSO."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(relation_mode="additive_control", **kwargs)


__all__ = [
    "AdditiveProtectedQuotientControl",
    "ChannelwiseConvexNeighbourhood",
    "ProtectedQuotientSpatialOrganizer",
    "ProtectedQuotientSpatialSource",
    "SharedQuotientNeighbourhoodOrganizer",
    "trainable_state_signature",
    "zero_anchored_contraction",
]
