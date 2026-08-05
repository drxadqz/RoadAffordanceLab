"""Production DREL-E B component-full road-surface classifier.

This module is the release-sized implementation of the best *validated D350
production candidate*.  It deliberately contains only the retained algorithm:

* a semantic carrier operating at /4, /8, /16, and /32 resolution;
* a one-octave-finer, one-way directional/radial evidence ledger;
* constrained matched-response filters;
* moment-preserving transitions;
* regional mean/deviation organization; and
* a bounded write from the ledger into the semantic carrier.

The exploratory ablation switches live outside this public production module.
Keeping them out makes accidental deployment of a failed study arm impossible.
The attribute and state-dict names remain compatible with the research
``DRELComponentStudyBackbone(study_mode="full")`` checkpoint schema.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    "DRELBackbone",
    "DRELClassifier",
    "DropPath",
    "HomogeneousCarrierBlock",
    "MatchedResponseConv2d",
    "MomentPreservingTransition",
    "RadialCompositionQuotient",
    "RegionalContrastBlock",
    "build_drel_component_full",
]


def _as_four(values: tuple[int, ...], name: str) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 4 or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain four positive integers, got {result}")
    return result  # type: ignore[return-value]


def _reflect_pad(x: torch.Tensor, padding: int) -> torch.Tensor:
    padding = int(padding)
    if padding <= 0:
        return x
    mode = "reflect" if min(x.shape[-2:]) > padding else "replicate"
    return F.pad(x, (padding, padding, padding, padding), mode=mode)


def _reflect_average(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
) -> torch.Tensor:
    radius = int(kernel_size) // 2
    return F.avg_pool2d(
        _reflect_pad(x, radius),
        kernel_size=int(kernel_size),
        stride=int(stride),
    )


def _pool_one_octave(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Pool the ledger by exactly one octave and fail closed on shape drift."""

    pooled = _reflect_average(x, 3, stride=2)
    target = tuple(int(value) for value in target_hw)
    if tuple(pooled.shape[-2:]) != target:
        raise RuntimeError(
            "DREL ledger/semantic octave contract drift: "
            f"pooled={tuple(pooled.shape[-2:])}, target={target}"
        )
    return pooled


class DropPath(nn.Module):
    """Per-sample stochastic depth without a timm dependency."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= float(probability) < 1.0:
            raise ValueError("drop-path probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class _NormalizedConv2d(nn.Module):
    """Bias-free normalized convolution for the homogeneous ledger carrier."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        kernel_size = int(kernel_size)
        groups = int(groups)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = int(stride)
        self.groups = groups
        self.eps = float(eps)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="leaky_relu")

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        super()._apply(fn, recurse=recurse)
        # The theorem-bearing ledger parameters remain FP32 under model.half().
        self.weight.data = self.weight.data.float()
        if self.weight.grad is not None:
            self.weight.grad.data = self.weight.grad.data.float()
        return self

    def projected_weight(self) -> torch.Tensor:
        flat = self.weight.float().flatten(1)
        norm = flat.square().sum(dim=1, keepdim=True).sqrt().clamp_min(self.eps)
        return (flat / norm).reshape(self.weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = _reflect_pad(x.float(), self.kernel_size // 2)
            return F.conv2d(
                work,
                self.projected_weight(),
                bias=None,
                stride=self.stride,
                groups=self.groups,
            )


class MatchedResponseConv2d(nn.Module):
    """Constrained even/odd directional matched-response filter bank.

    Each group learns one canonical even/odd residual around a fixed anchor.
    Rotating that canonical pair creates four directions that share one radial
    basis.  Every forward pass projects the filters to zero spatial DC, unit
    norm, and even/odd orthogonality.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        orientations: int = 4,
        eps: float = 1.0e-8,
        max_residual_norm: float = 0.25,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        kernel_size = int(kernel_size)
        orientations = int(orientations)
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and greater than one")
        if orientations <= 0:
            raise ValueError("orientations must be positive")
        response_width = 2 * orientations
        if out_channels % response_width != 0:
            raise ValueError("out_channels must be divisible by 2*orientations")
        if int(stride) <= 0:
            raise ValueError("stride must be positive")
        if not 0.0 <= float(max_residual_norm) <= 0.25:
            raise ValueError("max_residual_norm must be in [0, 0.25]")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = int(stride)
        self.orientations = orientations
        self.num_groups = out_channels // response_width
        self.eps = float(eps)
        self.max_residual_norm = float(max_residual_norm)
        self.canonical_weight = nn.Parameter(
            torch.zeros(self.num_groups, 2, in_channels, kernel_size, kernel_size)
        )
        self.register_buffer(
            "_rotation_grids",
            self._make_rotation_grids(kernel_size, orientations),
            persistent=False,
        )
        self.register_buffer(
            "_anchor_weight",
            self._make_anchor_weight(
                kernel_size,
                in_channels,
                orientations,
                self.num_groups,
            ),
        )

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        super()._apply(fn, recurse=recurse)
        self.canonical_weight.data = self.canonical_weight.data.float()
        if self.canonical_weight.grad is not None:
            self.canonical_weight.grad.data = self.canonical_weight.grad.data.float()
        self._anchor_weight = self._anchor_weight.float()
        self._rotation_grids = self._rotation_grids.float()
        return self

    @staticmethod
    def _make_rotation_grids(kernel_size: int, orientations: int) -> torch.Tensor:
        matrices = []
        for orientation in range(int(orientations)):
            theta = math.pi * float(orientation) / float(orientations)
            cosine = math.cos(theta)
            sine = math.sin(theta)
            matrices.append(
                torch.tensor(
                    [[cosine, -sine, 0.0], [sine, cosine, 0.0]],
                    dtype=torch.float32,
                )
            )
        affine = torch.stack(matrices, dim=0)
        return F.affine_grid(
            affine,
            size=(int(orientations), 1, int(kernel_size), int(kernel_size)),
            align_corners=True,
        )

    @staticmethod
    def _make_anchor_weight(
        kernel_size: int,
        in_channels: int,
        orientations: int,
        num_groups: int,
    ) -> torch.Tensor:
        size = int(kernel_size)
        coord = torch.linspace(-1.0, 1.0, size)
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        envelope = torch.exp(-(xx.square() + yy.square()) / 0.42)
        grids = MatchedResponseConv2d._make_rotation_grids(size, orientations)
        groups = []
        for group_index in range(int(num_groups)):
            radial_fraction = (float(group_index) + 0.5) / float(num_groups)
            frequency = math.pi * (0.85 + 0.70 * radial_fraction)
            even = envelope * torch.cos(frequency * xx)
            odd = envelope * torch.sin(frequency * xx)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                104729
                + 1009 * size
                + 9176 * int(in_channels)
                + 37 * int(group_index)
            )
            channel_gains = torch.randn(int(in_channels), generator=generator)
            channel_gains = channel_gains / channel_gains.norm().clamp_min(1.0e-8)
            base = (
                torch.stack([even, odd], dim=0)[:, None]
                * channel_gains[None, :, None, None]
            ).reshape(2 * int(in_channels), 1, size, size)
            orientations_for_group = []
            for orientation in range(int(orientations)):
                grid = grids[orientation : orientation + 1].expand(
                    base.shape[0], -1, -1, -1
                )
                rotated = F.grid_sample(
                    base,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).reshape(2, int(in_channels), size, size)
                rotated = rotated - rotated.mean(dim=(-2, -1), keepdim=True)
                anchor_even = rotated[0].flatten()
                anchor_even = anchor_even / anchor_even.norm()
                anchor_odd = rotated[1].flatten()
                anchor_odd = anchor_odd - torch.dot(anchor_odd, anchor_even) * anchor_even
                anchor_odd = anchor_odd / anchor_odd.norm()
                orientations_for_group.append(
                    torch.stack([anchor_even, anchor_odd], dim=0).reshape(
                        2, int(in_channels), size, size
                    )
                )
            groups.append(torch.stack(orientations_for_group, dim=0))
        return torch.stack(groups, dim=0)

    def _rotated_canonical(self) -> torch.Tensor:
        base = self.canonical_weight.float().reshape(
            self.num_groups * 2 * self.in_channels,
            1,
            self.kernel_size,
            self.kernel_size,
        )
        grids = self._rotation_grids.to(device=base.device, dtype=base.dtype)
        rotated = []
        for orientation in range(self.orientations):
            grid = grids[orientation : orientation + 1].expand(
                base.shape[0], -1, -1, -1
            )
            current = F.grid_sample(
                base,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            rotated.append(
                current.reshape(
                    self.num_groups,
                    2,
                    self.in_channels,
                    self.kernel_size,
                    self.kernel_size,
                )
            )
        return torch.stack(rotated, dim=1)

    def projected_weight(self) -> torch.Tensor:
        residual = self._rotated_canonical()
        residual = residual - residual.mean(dim=(-2, -1), keepdim=True)
        residual = residual.flatten(3)
        residual_norm = residual.square().sum(dim=-1, keepdim=True).add(
            self.eps**2
        ).sqrt()
        residual = self.max_residual_norm * residual / (1.0 + residual_norm)
        anchor = self._anchor_weight.to(
            device=residual.device,
            dtype=residual.dtype,
        ).flatten(3)

        even = anchor[:, :, 0] + residual[:, :, 0]
        even = even / even.square().sum(dim=-1, keepdim=True).sqrt()
        odd = anchor[:, :, 1] + residual[:, :, 1]
        odd = odd - (odd * even).sum(dim=-1, keepdim=True) * even
        odd = odd / odd.square().sum(dim=-1, keepdim=True).sqrt()
        return torch.stack([even, odd], dim=2).reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

    @torch.no_grad()
    def zero_dc_error(self) -> torch.Tensor:
        return self.projected_weight().sum(dim=(-2, -1)).abs().amax()

    @torch.no_grad()
    def unit_norm_error(self) -> torch.Tensor:
        weight = self.projected_weight().flatten(1)
        return (weight.norm(dim=1) - 1.0).abs().amax()

    @torch.no_grad()
    def pair_orthogonality_error(self) -> torch.Tensor:
        weight = self.projected_weight().reshape(
            self.num_groups, self.orientations, 2, -1
        )
        return (weight[:, :, 0] * weight[:, :, 1]).sum(dim=-1).abs().amax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"MatchedResponseConv2d expected Bx{self.in_channels}xHxW, "
                f"got {tuple(x.shape)}"
            )
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = _reflect_pad(x.float(), self.kernel_size // 2)
            return F.conv2d(
                work,
                self.projected_weight(),
                bias=None,
                stride=self.stride,
            )


class RadialCompositionQuotient(nn.Module):
    """Split matched responses into direction, energy, and reliability states."""

    def __init__(
        self,
        channels: int,
        *,
        group_width: int = 8,
        orientations: int = 4,
        smoothing: float = 0.05,
        numerical_eps: float = 1.0e-12,
        observability_floor: float = 1.0e-8,
        fully_observable_energy: float = 1.0e-4,
    ) -> None:
        super().__init__()
        channels = int(channels)
        group_width = int(group_width)
        orientations = int(orientations)
        if group_width != 2 * orientations:
            raise ValueError("group_width must equal 2*orientations")
        if channels % group_width != 0:
            raise ValueError("channels must be divisible by group_width")
        if orientations < 2:
            raise ValueError("orientations must be at least two")
        if not math.isfinite(float(smoothing)) or float(smoothing) < 0.0:
            raise ValueError("smoothing must be finite and non-negative")
        if not math.isfinite(float(numerical_eps)) or float(numerical_eps) <= 0.0:
            raise ValueError("numerical_eps must be finite and positive")
        if not math.isfinite(float(observability_floor)) or not (
            float(observability_floor) > float(numerical_eps)
        ):
            raise ValueError("observability_floor must exceed numerical_eps")
        if not math.isfinite(float(fully_observable_energy)) or not (
            float(fully_observable_energy) > float(observability_floor)
        ):
            raise ValueError("fully_observable_energy must exceed observability_floor")

        self.channels = channels
        self.group_width = group_width
        self.orientations = orientations
        self.num_groups = channels // group_width
        self.composition_coordinate_mode = "centered_simplex"
        self.register_buffer(
            "_simplex_smoothing", torch.tensor(float(smoothing), dtype=torch.float64)
        )
        self.register_buffer(
            "_numerical_eps", torch.tensor(float(numerical_eps), dtype=torch.float64)
        )
        self.register_buffer(
            "_observability_floor",
            torch.tensor(float(observability_floor), dtype=torch.float64),
        )
        self.register_buffer(
            "_fully_observable_energy",
            torch.tensor(float(fully_observable_energy), dtype=torch.float64),
        )
        self.reliability_logit = nn.Parameter(torch.full((self.num_groups,), -4.0))

    @property
    def composition_channels(self) -> int:
        return self.num_groups * self.orientations

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        super()._apply(fn, recurse=recurse)
        self._simplex_smoothing = self._simplex_smoothing.double()
        self._numerical_eps = self._numerical_eps.double()
        self._observability_floor = self._observability_floor.double()
        self._fully_observable_energy = self._fully_observable_energy.double()
        return self

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        names = {
            "smoothing": prefix + "_simplex_smoothing",
            "numerical_eps": prefix + "_numerical_eps",
            "observability_floor": prefix + "_observability_floor",
            "fully_observable_energy": prefix + "_fully_observable_energy",
        }
        current = {
            "smoothing": float(self._simplex_smoothing.item()),
            "numerical_eps": float(self._numerical_eps.item()),
            "observability_floor": float(self._observability_floor.item()),
            "fully_observable_energy": float(self._fully_observable_energy.item()),
        }
        values = dict(current)
        for semantic_name, state_name in names.items():
            supplied = state_dict.get(state_name)
            if supplied is None:
                continue
            if (
                not torch.is_tensor(supplied)
                or supplied.shape != torch.Size([])
                or supplied.dtype != torch.float64
            ):
                error_msgs.append(
                    f'Invalid DREL numerical buffer "{state_name}": '
                    "expected a scalar float64 tensor."
                )
                continue
            values[semantic_name] = float(supplied.item())
        valid = (
            math.isfinite(values["smoothing"])
            and values["smoothing"] >= 0.0
            and math.isfinite(values["numerical_eps"])
            and values["numerical_eps"] > 0.0
            and math.isfinite(values["observability_floor"])
            and values["observability_floor"] > values["numerical_eps"]
            and math.isfinite(values["fully_observable_energy"])
            and values["fully_observable_energy"] > values["observability_floor"]
        )
        if not valid:
            error_msgs.append(
                f'Invalid DREL numerical domain at "{prefix}": {values}.'
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

    def directional_energy(self, response: torch.Tensor) -> torch.Tensor:
        if response.ndim != 4 or response.shape[1] != self.channels:
            raise ValueError(
                f"RadialCompositionQuotient expected Bx{self.channels}xHxW, "
                f"got {tuple(response.shape)}"
            )
        work = (
            response.float()
            if response.dtype in {torch.float16, torch.bfloat16}
            else response
        )
        batch, _, height, width = work.shape
        grouped = work.reshape(
            batch,
            self.num_groups,
            self.orientations,
            2,
            height,
            width,
        )
        return grouped.square().sum(dim=3)

    def forward_from_directional_energy(
        self,
        energy: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        if energy.ndim != 5:
            raise ValueError("directional energy must be BxGxOxHxW")
        batch, groups, orientations, height, width = energy.shape
        if groups != self.num_groups or orientations != self.orientations:
            raise ValueError("directional-energy geometry mismatch")
        if energy.dtype in {torch.float16, torch.bfloat16}:
            energy = energy.float()

        total = energy.sum(dim=2)
        observability_floor = self._observability_floor.to(total)
        fully_observable_energy = self._fully_observable_energy.to(total)
        transition = (
            (total - observability_floor)
            / (fully_observable_energy - observability_floor)
        ).clamp(0.0, 1.0)
        observability = transition.square() * (3.0 - 2.0 * transition)
        composition_total = total.clamp_min(observability_floor)
        eta = self._simplex_smoothing.to(total)
        numerator = energy + (eta / float(self.orientations)) * total.unsqueeze(2)
        denominator = (1.0 + eta) * composition_total.unsqueeze(2)
        uniform = torch.full_like(energy, 1.0 / float(self.orientations))
        centered = observability.unsqueeze(2) * (numerator / denominator - uniform)
        probabilities = uniform + centered
        log_energy = 0.5 * torch.log(total.clamp_min(self._numerical_eps.to(total)))
        nu = 1.0e-6 + (1.0 - 1.0e-6) * torch.sigmoid(self.reliability_logit)
        reliability = total / (total + nu[None, :, None, None])

        result = {
            "composition": centered.reshape(
                batch, self.composition_channels, height, width
            ),
            "log_energy": log_energy,
            "reliability": reliability,
            "observability": observability,
            "probabilities": probabilities,
            "total_energy": total,
        }
        if output_dtype in {torch.float16, torch.bfloat16}:
            result = {name: value.to(dtype=output_dtype) for name, value in result.items()}
        return result

    def forward(self, response: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_from_directional_energy(
            self.directional_energy(response), output_dtype=response.dtype
        )


class HomogeneousCarrierBlock(nn.Module):
    """Positive-homogeneous residual refinement for the isolated ledger."""

    def __init__(self, channels: int, *, residual_scale: float = 1.0e-3) -> None:
        super().__init__()
        channels = int(channels)
        self.depthwise = _NormalizedConv2d(
            channels, channels, 3, groups=channels
        )
        self.activation = nn.PReLU(channels, init=0.20)
        self.pointwise = _NormalizedConv2d(channels, channels, 1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        super()._apply(fn, recurse=recurse)
        self.activation.weight.data = self.activation.weight.data.float()
        if self.activation.weight.grad is not None:
            self.activation.weight.grad.data = self.activation.weight.grad.data.float()
        self.residual_scale.data = self.residual_scale.data.float()
        if self.residual_scale.grad is not None:
            self.residual_scale.grad.data = self.residual_scale.grad.data.float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x.float()
            activated = F.prelu(
                self.depthwise(work), self.activation.weight.float()
            )
            return work + self.residual_scale.float() * self.pointwise(activated)


class MomentPreservingTransition(nn.Module):
    """Downsample while explicitly carrying local mean and dispersion."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            2 * int(in_channels), int(out_channels), 1, bias=False
        )
        self.norm = nn.BatchNorm2d(int(out_channels))
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x.float()
            mean = _reflect_average(work, 3, stride=2)
            second = _reflect_average(work.square(), 3, stride=2)
            deviation = (second - mean.square()).clamp_min(0.0).add(1.0e-6).sqrt()
            moments = torch.cat([mean, deviation], dim=1)
        return self.activation(self.norm(self.projection(moments)))


class RegionalContrastBlock(nn.Module):
    """Organize low-frequency regional context and local deviations."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 2,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        hidden = int(expansion) * channels
        self.pre_norm = nn.BatchNorm2d(channels)
        self.context = nn.Conv2d(
            channels,
            channels,
            5,
            padding=2,
            groups=channels,
            bias=False,
            padding_mode="replicate",
        )
        self.deviation = nn.Conv2d(
            channels,
            channels,
            3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
            padding_mode="replicate",
        )
        self.mix_norm = nn.BatchNorm2d(channels)
        self.expand = nn.Conv2d(channels, 2 * hidden, 1, bias=False)
        self.expand_norm = nn.BatchNorm2d(2 * hidden)
        self.compress = nn.Conv2d(hidden, channels, 1, bias=False)
        self.compress_norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU6(inplace=True)
        self.gate_activation = nn.ReLU6(inplace=False)
        self.drop_path = DropPath(float(drop_path))

    def reset_residual_strength(self) -> None:
        nn.init.constant_(self.compress_norm.weight, 0.10)
        nn.init.zeros_(self.compress_norm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre_norm(x)
        local_mean = _reflect_average(z, 3, stride=1)
        local_deviation = z - local_mean
        organized = self.activation(
            self.mix_norm(
                self.context(local_mean) + self.deviation(local_deviation)
            )
        )
        content, gate = self.expand_norm(self.expand(organized)).chunk(2, dim=1)
        bounded_product = self.gate_activation(content) * torch.sigmoid(gate)
        update = self.compress_norm(self.compress(bounded_product))
        return x + self.drop_path(update)


class DRELBackbone(nn.Module):
    """Frozen DREL-E B ``component_full`` production backbone.

    The defaults are the exact D350 production-candidate architecture.  The
    public class intentionally offers no ``ledger_mode`` or ablation switch.
    """

    def __init__(
        self,
        out_dim: int = 320,
        *,
        semantic_channels: tuple[int, ...] = (48, 96, 192, 320),
        semantic_depths: tuple[int, ...] = (1, 2, 5, 2),
        ledger_channels: tuple[int, ...] = (16, 24, 32, 48),
        response_kernel_sizes: tuple[int, ...] = (7, 5, 3, 3),
        group_width: int = 8,
        orientations: int = 4,
        simplex_smoothing: float = 0.05,
        drop_path_rate: float = 0.10,
        semantic_expansion: int = 2,
        evidence_write_bound: float = 0.25,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("DREL is trained from scratch and has no pretrained weights")
        self.out_dim = int(out_dim)
        self.semantic_channels = _as_four(semantic_channels, "semantic_channels")
        self.semantic_depths = _as_four(semantic_depths, "semantic_depths")
        self.ledger_channels = _as_four(ledger_channels, "ledger_channels")
        self.response_kernel_sizes = _as_four(
            response_kernel_sizes, "response_kernel_sizes"
        )
        self.group_width = int(group_width)
        self.orientations = int(orientations)
        self.ledger_mode = "bounded_energy_control"
        if self.group_width != 2 * self.orientations:
            raise ValueError("group_width must equal 2*orientations")
        if any(channels % self.group_width for channels in self.ledger_channels):
            raise ValueError("every ledger width must be divisible by group_width")
        if any(kernel <= 1 or kernel % 2 == 0 for kernel in self.response_kernel_sizes):
            raise ValueError("response kernels must be odd and greater than one")
        if not 0.0 <= float(drop_path_rate) < 1.0:
            raise ValueError("drop_path_rate must lie in [0, 1)")
        self.evidence_write_bound = float(evidence_write_bound)
        if not math.isfinite(self.evidence_write_bound) or not (
            0.0 < self.evidence_write_bound <= 0.5
        ):
            raise ValueError("evidence_write_bound must lie in (0, 0.5]")

        # These buffers make checkpoint semantics fail closed.
        self.register_buffer("_drel_ledger_mode_code", torch.tensor(1, dtype=torch.int64))
        self.register_buffer("_drel_schema_version", torch.tensor(1, dtype=torch.int64))

        c0 = self.semantic_channels[0]
        self.semantic_stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU6(inplace=True),
            nn.Conv2d(c0, c0, 3, stride=2, padding=1, groups=c0, bias=False),
            nn.Conv2d(c0, c0, 1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU6(inplace=True),
        )
        self.semantic_transitions = nn.ModuleList(
            [
                MomentPreservingTransition(
                    self.semantic_channels[index - 1], self.semantic_channels[index]
                )
                for index in range(1, 4)
            ]
        )

        rates = torch.linspace(
            0.0, float(drop_path_rate), sum(self.semantic_depths)
        ).tolist()
        cursor = 0
        semantic_stages: list[nn.Module] = []
        for channels, depth in zip(
            self.semantic_channels, self.semantic_depths, strict=True
        ):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    RegionalContrastBlock(
                        channels,
                        expansion=int(semantic_expansion),
                        drop_path=float(rates[cursor]),
                    )
                )
                cursor += 1
            semantic_stages.append(nn.Sequential(*blocks))
        self.semantic_stages = nn.ModuleList(semantic_stages)

        response_banks: list[nn.Module] = []
        quotients: list[nn.Module] = []
        ledger_refiners: list[nn.Module] = []
        composition_writes: list[nn.Module] = []
        radial_writes: list[nn.Module] = []
        previous = 3
        for stage_index, (ledger_width, semantic_width, kernel) in enumerate(
            zip(
                self.ledger_channels,
                self.semantic_channels,
                self.response_kernel_sizes,
                strict=True,
            )
        ):
            response_banks.append(
                MatchedResponseConv2d(
                    previous,
                    ledger_width,
                    kernel,
                    stride=2,
                    orientations=self.orientations,
                )
            )
            quotient = RadialCompositionQuotient(
                ledger_width,
                group_width=self.group_width,
                orientations=self.orientations,
                smoothing=float(simplex_smoothing),
            )
            quotients.append(quotient)
            ledger_refiners.append(
                nn.Identity()
                if stage_index == 3
                else HomogeneousCarrierBlock(ledger_width, residual_scale=1.0e-3)
            )
            composition_writes.append(
                nn.Conv2d(
                    quotient.composition_channels, semantic_width, 1, bias=False
                )
            )
            radial_writes.append(
                nn.Conv2d(2 * quotient.num_groups, semantic_width, 1, bias=False)
            )
            previous = ledger_width
        self.response_banks = nn.ModuleList(response_banks)
        self.quotients = nn.ModuleList(quotients)
        self.ledger_refiners = nn.ModuleList(ledger_refiners)
        self.composition_writes = nn.ModuleList(composition_writes)
        self.radial_writes = nn.ModuleList(radial_writes)

        self.final_norm = nn.BatchNorm2d(self.semantic_channels[-1])
        self.output_projection: nn.Module = (
            nn.Identity()
            if self.out_dim == self.semantic_channels[-1]
            else nn.Linear(self.semantic_channels[-1], self.out_dim)
        )
        self.last_feature_map: torch.Tensor | None = None
        self.stage_feature_maps: dict[str, torch.Tensor] = {}
        self.last_aux: dict[str, torch.Tensor] = {}

        self._initialize_semantic_path()
        for writer in (*self.composition_writes, *self.radial_writes):
            nn.init.zeros_(writer.weight)

        # Match the research component_full checkpoint schema exactly.
        self.register_buffer(
            "_drel_component_study_code", torch.tensor(0, dtype=torch.int64)
        )
        self.register_buffer(
            "_drel_component_study_schema", torch.tensor(1, dtype=torch.int64)
        )

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _initialize_semantic_path(self) -> None:
        self.semantic_stem.apply(self._initialize_module)
        self.semantic_transitions.apply(self._initialize_module)
        self.semantic_stages.apply(self._initialize_module)
        self.composition_writes.apply(self._initialize_module)
        self.radial_writes.apply(self._initialize_module)
        self.final_norm.apply(self._initialize_module)
        self.output_projection.apply(self._initialize_module)
        for stage in self.semantic_stages:
            for block in stage:
                if isinstance(block, RegionalContrastBlock):
                    block.reset_residual_strength()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for suffix, expected in (
            ("_drel_ledger_mode_code", self._drel_ledger_mode_code),
            ("_drel_schema_version", self._drel_schema_version),
            ("_drel_component_study_code", self._drel_component_study_code),
            ("_drel_component_study_schema", self._drel_component_study_schema),
        ):
            key = prefix + suffix
            supplied = state_dict.get(key)
            if supplied is None:
                continue
            valid = (
                torch.is_tensor(supplied)
                and supplied.shape == expected.shape
                and supplied.dtype == expected.dtype
                and torch.equal(supplied.detach().cpu(), expected.detach().cpu())
            )
            if not valid:
                error_msgs.append(
                    f"DREL checkpoint semantic buffer {suffix!r} does not match "
                    "the frozen component_full model"
                )
                # Preserve the target identity even when the load raises.
                state_dict[key] = expected.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _bounded_energy_coordinate(
        self,
        quotient: RadialCompositionQuotient,
        response: torch.Tensor,
    ) -> torch.Tensor:
        energy = quotient.directional_energy(response)
        bounded = energy / (1.0 + energy)
        centered = bounded - bounded.mean(dim=2, keepdim=True)
        return centered.reshape(
            response.shape[0],
            quotient.composition_channels,
            response.shape[2],
            response.shape[3],
        )

    def _write_evidence(
        self,
        *,
        stage_index: int,
        response: torch.Tensor,
        decomposition: dict[str, torch.Tensor],
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direction = self._bounded_energy_coordinate(
            self.quotients[stage_index], response
        )
        radial = torch.cat(
            [
                torch.tanh(decomposition["log_energy"] / 4.0),
                2.0 * decomposition["reliability"] - 1.0,
            ],
            dim=1,
        )
        direction = _pool_one_octave(direction, target_hw)
        radial = _pool_one_octave(radial, target_hw)
        raw_write = (
            self.composition_writes[stage_index](direction)
            + self.radial_writes[stage_index](radial)
        )
        return (
            self.evidence_write_bound * torch.tanh(raw_write),
            decomposition["reliability"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or int(x.shape[1]) != 3:
            raise ValueError(
                f"DREL expects Bx3xHxW, got {getattr(x, 'shape', None)}"
            )
        semantic = self.semantic_stem(x)
        ledger = x
        stage_maps: dict[str, torch.Tensor] = {}
        write_rms: list[torch.Tensor] = []
        reliability_means: list[torch.Tensor] = []
        stage_names = ("early", "mid", "late", "final")

        for stage_index in range(4):
            if stage_index > 0:
                semantic = self.semantic_transitions[stage_index - 1](semantic)
            response = self.response_banks[stage_index](ledger)
            decomposition = self.quotients[stage_index](response)
            ledger = self.ledger_refiners[stage_index](response)
            write, reliability = self._write_evidence(
                stage_index=stage_index,
                response=response,
                decomposition=decomposition,
                target_hw=(int(semantic.shape[-2]), int(semantic.shape[-1])),
            )
            semantic = self.semantic_stages[stage_index](
                semantic + write.to(dtype=semantic.dtype)
            )
            stage_maps[stage_names[stage_index]] = semantic
            write_rms.append(
                write.detach().float().square().mean(dim=(1, 2, 3)).sqrt()
            )
            reliability_means.append(
                reliability.detach().float().mean(dim=(1, 2, 3))
            )

        feature_map = self.final_norm(semantic)
        pooled = feature_map.mean(dim=(2, 3))
        embedding = self.output_projection(pooled)
        self.stage_feature_maps = stage_maps
        self.last_feature_map = feature_map
        self.last_aux = {
            "drel_write_rms": torch.stack(write_rms, dim=1),
            "drel_mean_reliability": torch.stack(reliability_means, dim=1),
            "drel_semantic_embedding": pooled.detach(),
        }
        return embedding


class DRELClassifier(nn.Module):
    """DREL backbone plus the 27-way linear head used in the D350 study."""

    def __init__(
        self,
        num_classes: int = 27,
        *,
        embedding_dim: int = 320,
        head_init_seed: int | None = None,
        **backbone_kwargs: Any,
    ) -> None:
        super().__init__()
        if int(num_classes) <= 1:
            raise ValueError("num_classes must be greater than one")
        self.backbone = DRELBackbone(
            out_dim=int(embedding_dim), **backbone_kwargs
        )
        self.head = nn.Linear(int(embedding_dim), int(num_classes))
        if head_init_seed is not None:
            generator = torch.Generator(device=self.head.weight.device)
            generator.manual_seed(int(head_init_seed))
            nn.init.kaiming_uniform_(
                self.head.weight, a=math.sqrt(5), generator=generator
            )
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.head.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.head.bias, -bound, bound, generator=generator)

    def forward(
        self,
        image: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        embedding = self.backbone(image)
        logits = self.head(embedding)
        if not return_aux:
            return logits
        return {
            "logits": logits,
            "embedding": embedding,
            "drel": self.backbone.last_aux,
            "stage_feature_maps": self.backbone.stage_feature_maps,
        }


def build_drel_component_full(
    num_classes: int = 27,
    *,
    head_init_seed: int | None = None,
) -> DRELClassifier:
    """Build the exact frozen DREL-E B component-full default model."""

    return DRELClassifier(
        num_classes=num_classes,
        embedding_dim=320,
        head_init_seed=head_init_seed,
        semantic_channels=(48, 96, 192, 320),
        semantic_depths=(1, 2, 5, 2),
        ledger_channels=(16, 24, 32, 48),
        response_kernel_sizes=(7, 5, 3, 3),
        group_width=8,
        orientations=4,
        simplex_smoothing=0.05,
        drop_path_rate=0.10,
        semantic_expansion=2,
        evidence_write_bound=0.25,
        pretrained=False,
    )
