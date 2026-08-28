from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


__all__ = [
    "ADDITIVE_CONTROL",
    "PRODUCT_CONSENSUS",
    "QuotientOrthogonalDetailConsensusLift",
    "balanced_block_lift",
    "balanced_block_project",
    "balanced_block_restrict",
    "balanced_partition_sizes",
]


ADDITIVE_CONTROL = "additive_control"
PRODUCT_CONSENSUS = "product_consensus"
_MODE_TO_CODE = {
    ADDITIVE_CONTROL: 1,
    PRODUCT_CONSENSUS: 2,
}


def balanced_partition_sizes(length: int, target: int) -> tuple[int, ...]:
    """Return the exact non-overlapping cells induced by floor(t*N/T)."""

    length = int(length)
    target = int(target)
    if length <= 0:
        raise ValueError("length must be positive")
    if target <= 0 or target > length:
        raise ValueError("target must satisfy 1 <= target <= length")
    boundaries = tuple((index * length) // target for index in range(target + 1))
    sizes = tuple(
        boundaries[index + 1] - boundaries[index] for index in range(target)
    )
    if any(size <= 0 for size in sizes) or sum(sizes) != length:
        raise RuntimeError("balanced partition did not cover the axis exactly")
    return sizes


def balanced_block_restrict(
    x: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    accumulation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Average exact non-overlapping balanced blocks.

    This is deliberately not ``adaptive_avg_pool2d``.  The paired lift below
    uses the very same cells, which makes ``lift(restrict(x))`` an orthogonal
    projector even when the input shape is not divisible by the target shape.
    """

    if x.ndim < 2:
        raise ValueError("x must have at least two spatial dimensions")
    height, width = int(x.shape[-2]), int(x.shape[-1])
    height_sizes = balanced_partition_sizes(height, int(target_height))
    width_sizes = balanced_partition_sizes(width, int(target_width))
    # Separable restriction matrices implement the exact declared cells with
    # two small batched matrix products.  Unlike an integral-image subtraction,
    # this form does not magnify cancellation at late spatial positions and
    # maps a block-constant FP32 tensor back to the same representable value for
    # the 1/2-sized cells used by QODC's half-resolution partition.
    def _matrix(sizes: tuple[int, ...], length: int) -> torch.Tensor:
        repeats = torch.tensor(sizes, device=x.device, dtype=torch.long)
        region_ids = torch.repeat_interleave(
            torch.arange(len(sizes), device=x.device),
            repeats,
            output_size=length,
        )
        matrix = F.one_hot(region_ids, num_classes=len(sizes)).to(
            dtype=accumulation_dtype
        ).transpose(0, 1)
        return matrix / repeats.to(dtype=accumulation_dtype).view(-1, 1)

    height_matrix = _matrix(height_sizes, height)
    width_matrix = _matrix(width_sizes, width)
    x_accumulated = x.to(dtype=accumulation_dtype)
    height_pooled = torch.matmul(height_matrix, x_accumulated)
    return torch.matmul(height_pooled, width_matrix.transpose(0, 1))


def balanced_block_lift(
    y: torch.Tensor,
    *,
    output_height: int,
    output_width: int,
) -> torch.Tensor:
    """Replicate every regional value over the cells used by restriction."""

    if y.ndim < 2:
        raise ValueError("y must have at least two spatial dimensions")
    target_height, target_width = int(y.shape[-2]), int(y.shape[-1])
    height_sizes = balanced_partition_sizes(int(output_height), target_height)
    width_sizes = balanced_partition_sizes(int(output_width), target_width)
    height_repeats = torch.tensor(height_sizes, device=y.device, dtype=torch.long)
    width_repeats = torch.tensor(width_sizes, device=y.device, dtype=torch.long)
    lifted = torch.repeat_interleave(
        y,
        height_repeats,
        dim=-2,
        output_size=int(output_height),
    )
    return torch.repeat_interleave(
        lifted,
        width_repeats,
        dim=-1,
        output_size=int(output_width),
    )


def balanced_block_project(
    x: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    accumulation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Apply the exact balanced block orthogonal projector Pi = U P."""

    pooled = balanced_block_restrict(
        x,
        target_height=target_height,
        target_width=target_width,
        accumulation_dtype=accumulation_dtype,
    )
    lifted = balanced_block_lift(
        pooled,
        output_height=int(x.shape[-2]),
        output_width=int(x.shape[-1]),
    )
    return lifted.to(dtype=x.dtype)


class _ChannelLayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.weight = nn.Parameter(torch.ones(self.channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or int(x.shape[1]) != self.channels:
            raise ValueError(
                f"expected Bx{self.channels}xHxW, got {tuple(x.shape)}"
            )
        y = x.permute(0, 2, 3, 1)
        y = F.layer_norm(
            y,
            (self.channels,),
            self.weight.to(dtype=y.dtype),
            None,
            self.eps,
        )
        return y.permute(0, 3, 1, 2)


class _ChannelRMSNorm2d(nn.Module):
    """Channel RMS normalization without deleting common-channel brightness."""

    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.weight = nn.Parameter(torch.ones(self.channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or int(x.shape[1]) != self.channels:
            raise ValueError(
                f"expected Bx{self.channels}xHxW, got {tuple(x.shape)}"
            )
        denominator = torch.sqrt(
            x.float().square().mean(dim=1, keepdim=True) + self.eps**2
        )
        normalized = x / denominator.to(dtype=x.dtype)
        return (
            normalized
            * self.weight.to(dtype=normalized.dtype).view(1, -1, 1, 1)
        )


class QuotientOrthogonalDetailConsensusLift(nn.Module):
    """Projection-constrained late consensus for ARCQ's protected C/A states.

    The learned update is lifted from the balanced block-constant subspace.
    Consequently, in exact arithmetic, the original within-block complement is
    unchanged: ``(I-Pi)F' == (I-Pi)F``.  This is a non-interference guarantee,
    not a claim that the preserved complement is necessarily discriminative.
    """

    def __init__(
        self,
        *,
        mode: str,
        evidence_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
        residual_max: float = 0.10,
        detail_delta: float = 1.0e-6,
        normalization_delta: float = 1.0e-4,
    ) -> None:
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in _MODE_TO_CODE:
            raise ValueError(
                f"mode must be one of {sorted(_MODE_TO_CODE)}, got {mode!r}"
            )
        self.mode = mode
        self.evidence_channels = int(evidence_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)
        if min(self.evidence_channels, self.out_channels, self.hidden_channels) <= 0:
            raise ValueError("all channel counts must be positive")
        if not math.isfinite(float(residual_max)) or not 0.0 < residual_max <= 1.0:
            raise ValueError("residual_max must be finite and in (0, 1]")
        if detail_delta <= 0.0 or normalization_delta <= 0.0:
            raise ValueError("numerical deltas must be strictly positive")
        self.residual_max = float(
            torch.tensor(float(residual_max), dtype=torch.float32).item()
        )
        self.detail_delta = float(
            torch.tensor(float(detail_delta), dtype=torch.float32).item()
        )
        self.normalization_delta = float(
            torch.tensor(float(normalization_delta), dtype=torch.float32).item()
        )

        self.register_buffer(
            "_relation_mode_code",
            torch.tensor(_MODE_TO_CODE[mode], dtype=torch.int64),
        )
        self.register_buffer(
            "_residual_max",
            torch.tensor(float(residual_max), dtype=torch.float32),
        )
        self.register_buffer(
            "_detail_delta",
            torch.tensor(float(detail_delta), dtype=torch.float32),
        )
        self.register_buffer(
            "_normalization_delta",
            torch.tensor(float(normalization_delta), dtype=torch.float32),
        )

        channels = self.evidence_channels
        hidden = self.hidden_channels
        self.composition_mean_norm = _ChannelLayerNorm2d(channels)
        self.appearance_mean_norm = _ChannelRMSNorm2d(channels)
        self.composition_projection = nn.Conv2d(
            2 * channels,
            hidden,
            kernel_size=1,
            bias=False,
        )
        # The explicit mean and log-RMS coordinates prevent a normalizer from
        # silently deleting absolute appearance/amplitude information.
        self.appearance_projection = nn.Conv2d(
            2 * channels + 2,
            hidden,
            kernel_size=1,
            bias=False,
        )
        self.composition_relation_norm = _ChannelRMSNorm2d(hidden)
        self.appearance_relation_norm = _ChannelRMSNorm2d(hidden)
        self.relation_norm = _ChannelLayerNorm2d(hidden)
        self.relation_spatial = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=0,
            groups=hidden,
            bias=False,
        )
        self.writer = nn.Conv2d(
            hidden,
            self.out_channels,
            kernel_size=1,
            bias=False,
        )
        # All descriptor-to-writer maps are bias-free and the normalizers are
        # weight-only.  Hence the product arm is exactly zero whenever either
        # evidence descriptor is zero; it cannot manufacture unconditional
        # "consensus" through learned offsets.  Only the scalar gate is
        # zero-initialized.  Random non-zero weights still give alpha a live
        # first-step gradient on non-degenerate inputs.
        self.alpha = nn.Parameter(torch.zeros(()))
        self.last_aux: dict[str, torch.Tensor] = {}

    @staticmethod
    def target_shape(height: int, width: int) -> tuple[int, int]:
        height = int(height)
        width = int(width)
        if height <= 0 or width <= 0:
            raise ValueError("spatial dimensions must be positive")
        return (height + 1) // 2, (width + 1) // 2

    @staticmethod
    def _rms_per_sample(x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().square().mean(dim=(1, 2, 3)).sqrt()

    def _detail_ledger(
        self,
        x: torch.Tensor,
        *,
        target_height: int,
        target_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_float = x.float()
        mean = balanced_block_restrict(
            x_float,
            target_height=target_height,
            target_width=target_width,
            accumulation_dtype=torch.float32,
        )
        lifted_mean = balanced_block_lift(
            mean,
            output_height=int(x.shape[-2]),
            output_width=int(x.shape[-1]),
        )
        detail = x_float - lifted_mean
        energy = balanced_block_restrict(
            detail.square(),
            target_height=target_height,
            target_width=target_width,
            accumulation_dtype=torch.float32,
        )
        delta = self._detail_delta
        # This is algebraically sqrt(e + delta^2) - delta, written without
        # catastrophic cancellation when e is small.
        magnitude = energy / (torch.sqrt(energy + delta.square()) + delta)
        return (
            mean.to(dtype=x.dtype),
            torch.log1p(magnitude).to(dtype=x.dtype),
            detail,
        )

    @staticmethod
    def _spatial_pad(x: torch.Tensor) -> torch.Tensor:
        mode = "reflect" if int(x.shape[-2]) > 1 and int(x.shape[-1]) > 1 else "replicate"
        return F.pad(x, (1, 1, 1, 1), mode=mode)

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
        expected_buffers = {
            "_relation_mode_code": self._relation_mode_code,
            "_residual_max": self._residual_max,
            "_detail_delta": self._detail_delta,
            "_normalization_delta": self._normalization_delta,
        }
        substituted: dict[str, torch.Tensor] = {}
        for suffix, expected in expected_buffers.items():
            state_name = prefix + suffix
            incoming = state_dict.get(state_name)
            invalid = incoming is not None and not (
                torch.is_tensor(incoming)
                and incoming.dtype == expected.dtype
                and incoming.shape == expected.shape
                and torch.equal(incoming.detach().cpu(), expected.detach().cpu())
            )
            if incoming is None:
                error_msgs.append(
                    f"missing immutable QODC configuration buffer {state_name!r}"
                )
            elif invalid:
                if suffix == "_relation_mode_code":
                    error_msgs.append(
                        "QODC checkpoint belongs to the other causal relation arm; "
                        "cross-mode loading is forbidden"
                    )
                else:
                    error_msgs.append(
                        f"QODC checkpoint/config mismatch for immutable buffer "
                        f"{state_name!r}"
                    )
                substituted[state_name] = incoming
                # Do not let nn.Module mutate the target before raising.
                state_dict[state_name] = expected.detach().clone()
        try:
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        finally:
            for state_name, incoming in substituted.items():
                state_dict[state_name] = incoming

    def _validate_runtime_contract(self) -> None:
        expected_code = _MODE_TO_CODE[self.mode]
        if (
            self._relation_mode_code.dtype != torch.int64
            or self._relation_mode_code.shape != torch.Size([])
            or int(self._relation_mode_code.detach().cpu().item()) != expected_code
        ):
            raise RuntimeError("QODC relation mode buffer does not match the arm")
        expected_float_buffers = (
            (self._residual_max, self.residual_max, "residual_max"),
            (self._detail_delta, self.detail_delta, "detail_delta"),
            (
                self._normalization_delta,
                self.normalization_delta,
                "normalization_delta",
            ),
        )
        for buffer, expected, name in expected_float_buffers:
            if (
                buffer.dtype != torch.float32
                or buffer.shape != torch.Size([])
                or float(buffer.detach().cpu().item()) != expected
            ):
                raise RuntimeError(
                    f"QODC immutable {name} buffer does not match construction"
                )

    def forward(
        self,
        feature_map: torch.Tensor,
        composition_state: torch.Tensor,
        appearance_state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_runtime_contract()
        if feature_map.ndim != 4:
            raise ValueError("feature_map must be BxCxHxW")
        expected_evidence = (
            int(feature_map.shape[0]),
            self.evidence_channels,
            int(feature_map.shape[-2]),
            int(feature_map.shape[-1]),
        )
        if tuple(composition_state.shape) != expected_evidence:
            raise ValueError(
                "composition_state shape mismatch: expected "
                f"{expected_evidence}, got {tuple(composition_state.shape)}"
            )
        if tuple(appearance_state.shape) != expected_evidence:
            raise ValueError(
                "appearance_state shape mismatch: expected "
                f"{expected_evidence}, got {tuple(appearance_state.shape)}"
            )
        if int(feature_map.shape[1]) != self.out_channels:
            raise ValueError(
                f"feature_map must have {self.out_channels} channels, got "
                f"{int(feature_map.shape[1])}"
            )

        height, width = int(feature_map.shape[-2]), int(feature_map.shape[-1])
        target_height, target_width = self.target_shape(height, width)
        mean_c, detail_c, raw_detail_c = self._detail_ledger(
            composition_state,
            target_height=target_height,
            target_width=target_width,
        )
        mean_a, detail_a, raw_detail_a = self._detail_ledger(
            appearance_state,
            target_height=target_height,
            target_width=target_width,
        )

        c_descriptor = torch.cat(
            [self.composition_mean_norm(mean_c), detail_c],
            dim=1,
        )
        appearance_common_mean = mean_a.float().mean(dim=1, keepdim=True)
        appearance_mean_energy = mean_a.float().square().mean(
            dim=1,
            keepdim=True,
        )
        appearance_delta = self._detail_delta
        appearance_zero_anchored_rms = appearance_mean_energy / (
            torch.sqrt(appearance_mean_energy + appearance_delta.square())
            + appearance_delta
        )
        appearance_log_rms = torch.log1p(appearance_zero_anchored_rms)
        a_descriptor = torch.cat(
            [
                self.appearance_mean_norm(mean_a),
                detail_a,
                appearance_common_mean.to(dtype=mean_a.dtype),
                appearance_log_rms.to(dtype=mean_a.dtype),
            ],
            dim=1,
        )
        z_c = self.composition_relation_norm(
            self.composition_projection(c_descriptor)
        )
        z_a = self.appearance_relation_norm(
            self.appearance_projection(a_descriptor)
        )
        if self.mode == ADDITIVE_CONTROL:
            relation = (z_c + z_a) / math.sqrt(2.0)
        else:
            relation = z_c * z_a
        relation = self.relation_norm(relation)
        relation = self.relation_spatial(self._spatial_pad(relation))
        q = self.writer(F.gelu(relation))

        lifted_q = balanced_block_lift(
            q,
            output_height=height,
            output_width=width,
        )
        normalization_delta = self._normalization_delta
        q_denominator = torch.sqrt(
            lifted_q.float().square().mean(
                dim=(1, 2, 3),
                keepdim=True,
            )
            + normalization_delta.square()
        )
        normalized_lift = lifted_q / q_denominator.to(dtype=lifted_q.dtype)
        feature_float = feature_map.detach().float().reshape(
            int(feature_map.shape[0]),
            -1,
        )
        feature_rms = (
            torch.linalg.vector_norm(feature_float, dim=1, keepdim=True)
            / math.sqrt(int(feature_float.shape[1]))
        ).view(-1, 1, 1, 1)
        effective_scale = self._residual_max * torch.tanh(self.alpha)
        residual = (
            effective_scale.to(dtype=feature_map.dtype)
            * feature_rms.to(dtype=feature_map.dtype)
            * normalized_lift.to(dtype=feature_map.dtype)
        )
        updated = feature_map + residual

        with torch.no_grad():
            base_rms = self._rms_per_sample(feature_map)
            residual_rms = self._rms_per_sample(residual)
            self.last_aux = {
                "effective_scale": effective_scale.detach().float().expand(
                    int(feature_map.shape[0])
                ),
                "alpha_tanh_abs": torch.tanh(self.alpha.detach()).abs().float().expand(
                    int(feature_map.shape[0])
                ),
                "composition_detail_rms": raw_detail_c.detach().square().mean(
                    dim=(1, 2, 3)
                ).sqrt(),
                "appearance_detail_rms": raw_detail_a.detach().square().mean(
                    dim=(1, 2, 3)
                ).sqrt(),
                "relation_rms": self._rms_per_sample(relation),
                "writer_rms": self._rms_per_sample(q),
                "residual_rms": residual_rms,
                "residual_to_feature_rms": residual_rms / base_rms.clamp_min(1.0e-12),
            }
        return updated, self.last_aux
