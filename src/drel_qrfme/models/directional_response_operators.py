from __future__ import annotations

# =============================================================================
# 当前 DREL-QRFME 实际调用的最短阅读路径
# =============================================================================
# MatchedResponseConv2d.forward
#   RGB/上一层账本 → 多方向 × 偶/奇相位的匹配响应
# RadialCompositionQuotient.forward
#   匹配响应 → 方向能量 → composition（相对方向组成）+ log_energy
# drel_qrfme_model.py
#   根据 log_energy 得到 reliability，再把 composition/radial 送入 Writer/RFME
#
# 关键约束：滤波器逐输入通道零 DC、单位范数、偶奇相位正交。这些投影在每次
# forward 时执行，避免训练后约束漂移。涉及投影的计算强制 FP32，即使外层 BF16。
# 本文件其他大型类主要是早期 ARCQ 消融与旧 checkpoint 状态语义兼容代码。
# =============================================================================

import math
import warnings
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

"""方向匹配响应、径向能量和组成商等共享数学算子。

DREL-QRFME 当前直接使用 ``DropPath``、``MatchedResponseConv2d`` 和
``RadialCompositionQuotient``。同文件中的其余类是这些算子的经过验证的
兼容实现与状态字典语义，保留它们可避免改变既有 checkpoint 的数值行为。
"""

from drel_qrfme.models.factor_evidence_product import (
    FactorEvidenceProductResidual,
)
from drel_qrfme.models.signed_consensus import (
    ENDPOINT_ENERGY_CONTROL,
    GRID_SIGNED_CONSENSUS,
    NullspaceGraphSignedConsensus,
)
from drel_qrfme.models.protected_quotient_organizer import (
    AdditiveProtectedQuotientControl,
    ProtectedQuotientSpatialOrganizer,
)
from drel_qrfme.models.response_coherence_transition import (
    PCQT_COHERENCE_QUOTIENT,
    PCQT_DISABLED,
    PCQT_MARGINAL_ENERGY_CONTROL,
    PairedResponseCoherenceTransition,
)
from drel_qrfme.models.quotient_detail_consensus import (
    ADDITIVE_CONTROL,
    PRODUCT_CONSENSUS,
    QuotientOrthogonalDetailConsensusLift,
)
from drel_qrfme.models.terminal_octave_operator import (
    TOC_DISABLED,
    TOC_MODE_CODES,
    TOC_MODE_STRIDES,
    TOC_SCHEMA_VERSION,
    TOC_VALID_MODES,
    TerminalOctavePair,
)
from drel_qrfme.models.ordinal_regression import CoralOrdinalHead
from drel_qrfme.rscd_label_factors import FACTOR_LABELS, build_rscd_factor_spec


__all__ = [
    "ARCQLegacyStateDictMigrationWarning",
    "ARCQNumericsStateDictMigrationWarning",
    "ARCQRoadBackbone",
    "ARCQRoadSurfaceClassifier",
    "MatchedResponseConv2d",
    "PairedResponseCoherenceTransition",
    "RadialCompositionQuotient",
    "RoleIdentifiedCAQCReadout",
]


_ARCQ_STATE_VERSION = 3
_ARCQ_STATE_VERSION_KEY = "_arcq_state_version"
_ARCQ_V3_NUMERICS_BUFFER_SUFFIXES = (
    "._simplex_smoothing",
    "._numerical_eps",
    "._observability_floor",
    "._fully_observable_energy",
)

# ECQD deliberately has no trainable writer.  The persistent active-arm mode
# code below prevents a checkpoint produced with ``E(P(r))`` from being silently
# evaluated as the semantically different ``P(E(r))`` arm (and vice versa),
# while the disabled default still has the exact legacy state schema.
_ECQD_DISABLED = "disabled"
_ECQD_RESPONSE_AVERAGE_CONTROL = "response_average_control"
_ECQD_ENERGY_COMPLETE = "energy_complete"
_ECQD_MODE_CODES = {
    _ECQD_RESPONSE_AVERAGE_CONTROL: 1,
    _ECQD_ENERGY_COMPLETE: 2,
}
_ECQD_SCHEMA_VERSION = 1


# LBET changes only the terminal A3->A4 spatial transition.  The historical
# signed transition remains the schema-free default.  The two active study
# arms own identical persistent buffers so checkpoint semantics fail closed.
_FINAL_A_TRANSITION_LEARNED_SIGNED = "learned_signed"
_FINAL_A_TRANSITION_AFFINE_TANGENT = "affine_tangent_control"
_FINAL_A_TRANSITION_COVERAGE_BARYCENTRIC = "coverage_barycentric"
_FINAL_A_TRANSITION_VALID_MODES = frozenset(
    {
        _FINAL_A_TRANSITION_LEARNED_SIGNED,
        _FINAL_A_TRANSITION_AFFINE_TANGENT,
        _FINAL_A_TRANSITION_COVERAGE_BARYCENTRIC,
    }
)
_FINAL_A_TRANSITION_ACTIVE_MODE_CODES = {
    _FINAL_A_TRANSITION_AFFINE_TANGENT: 1,
    _FINAL_A_TRANSITION_COVERAGE_BARYCENTRIC: 2,
}
_LBET_SCHEMA_VERSION = 1


# ``centered_simplex`` is the original ARCQ quotient and must remain entirely
# schema-compatible with legacy checkpoints.  The three named study arms keep
# the same O-channel interface: A writes the historical centered simplex, B
# writes its fixed Helmert projection, and C writes the reconstructed Helmert
# ILR coordinate.  Keeping C in the zero-sum O-dimensional subspace (rather
# than dropping to O-1 channels) makes the experiment parameter matched.
_ILR_ARCQ_CENTERED_SIMPLEX = "centered_simplex"
_ILR_ARCQ_CENTERED_SIMPLEX_CONTROL = "centered_simplex_control"
_ILR_ARCQ_HELMERT_LINEAR_CONTROL = "helmert_linear_control"
_ILR_ARCQ_ILR_HELMERT = "ilr_helmert"
_ILR_ARCQ_ACTIVE_MODE_CODES = {
    _ILR_ARCQ_CENTERED_SIMPLEX_CONTROL: 1,
    _ILR_ARCQ_HELMERT_LINEAR_CONTROL: 2,
    _ILR_ARCQ_ILR_HELMERT: 3,
}
_ILR_ARCQ_VALID_MODES = frozenset(
    {_ILR_ARCQ_CENTERED_SIMPLEX, *_ILR_ARCQ_ACTIVE_MODE_CODES}
)
_ILR_ARCQ_SCHEMA_VERSION = 1


class ARCQLegacyStateDictMigrationWarning(UserWarning):
    """Signals the audited removal of the legacy dead terminal-H parameters."""


class ARCQNumericsStateDictMigrationWarning(UserWarning):
    """Signals weights-only migration to FP32-H/smooth observability semantics."""


def _legacy_terminal_h_shapes(module: nn.Module) -> dict[str, torch.Size]:
    """Return the exact v1-only terminal-H parameter whitelist.

    ARCQ v1 refined the response after the fourth (last) quotient had already
    consumed it.  That refinement could not reach any prediction or auxiliary
    loss. V2 and later replace it with ``Identity``. Keeping this schema explicit is
    intentional: a vaguely matching ``h_stages.3.*`` payload must never be
    mistaken for a compatible checkpoint.
    """

    channels = int(module.h_channels[-1])  # type: ignore[attr-defined]
    depth = int(module.depths[-1])  # type: ignore[attr-defined]
    shapes: dict[str, torch.Size] = {}
    for block_index in range(depth):
        root = f"h_stages.3.{block_index}"
        shapes[f"{root}.residual_scale"] = torch.Size([])
        shapes[f"{root}.depthwise.weight"] = torch.Size((channels, 1, 3, 3))
        shapes[f"{root}.activation.weight"] = torch.Size((channels,))
        shapes[f"{root}.pointwise.weight"] = torch.Size(
            (channels, channels, 1, 1)
        )
    return shapes


def _prepare_arcq_state_dict_load(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    error_msgs: list[str],
) -> dict[str, Any] | None:
    """Validate/migrate an ARCQ v1/v2 payload before strict loading.

    The migration is deliberately all-or-nothing.  It removes only the exact
    dead terminal-H whitelist and injects the v3 version marker. Missing
    active tensors, extra backbone tensors, or any shape mismatch leave the
    input untouched and add a hard loader error.
    """

    version_name = prefix + _ARCQ_STATE_VERSION_KEY
    terminal_shapes = _legacy_terminal_h_shapes(module)
    terminal_names = {prefix + name for name in terminal_shapes}
    observed_terminal = {
        name for name in state_dict if name.startswith(prefix + "h_stages.3.")
    }
    expected_version = module._arcq_state_version.detach()  # type: ignore[attr-defined]
    current_state = module.state_dict()
    numerics_local = {
        name: value
        for name, value in current_state.items()
        if name.endswith(_ARCQ_V3_NUMERICS_BUFFER_SUFFIXES)
    }

    def inject_v3_numerics() -> tuple[str, ...]:
        injected: list[str] = []
        for local_name, expected_tensor in numerics_local.items():
            full_name = prefix + local_name
            if full_name not in state_dict:
                state_dict[full_name] = expected_tensor.detach().clone()
                injected.append(local_name)
        return tuple(sorted(injected))

    if version_name in state_dict:
        supplied_version = state_dict[version_name]
        source_version = (
            int(supplied_version.item())
            if torch.is_tensor(supplied_version)
            and supplied_version.shape == torch.Size([])
            and supplied_version.dtype == torch.int64
            else None
        )
        if source_version == 2:
            if observed_terminal:
                error_msgs.append(
                    "ARCQ v2 state_dict contains forbidden legacy terminal-H keys: "
                    + ", ".join(sorted(observed_terminal))
                )
                return None
            injected_keys = inject_v3_numerics()
            state_dict[version_name] = expected_version.clone()
            audit = {
                "migration": "arcq_v2_numerics_to_v3",
                "source_version": 2,
                "target_version": _ARCQ_STATE_VERSION,
                "dropped_keys": (),
                "injected_keys": injected_keys,
            }
            warnings.warn(
                "Loaded ARCQ v2 weights into v3. V3 evaluates the theorem-bearing "
                "H path in FP32 and uses a smooth low-energy observability band; "
                "optimizer/scheduler continuation from v2 is not compatible.",
                ARCQNumericsStateDictMigrationWarning,
                stacklevel=4,
            )
            return audit
        valid_version = (
            torch.is_tensor(supplied_version)
            and supplied_version.shape == torch.Size([])
            and supplied_version.dtype == torch.int64
            and int(supplied_version.item()) == _ARCQ_STATE_VERSION
        )
        if not valid_version:
            error_msgs.append(
                f'Invalid ARCQ state version at "{version_name}": expected '
                f"scalar int64 value {_ARCQ_STATE_VERSION}."
            )
            # A failed load is allowed to copy other valid tensors, as in the
            # standard PyTorch loader, but it must not corrupt the module's own
            # architecture marker.
            state_dict[version_name] = expected_version.clone()
        if observed_terminal:
            error_msgs.append(
                "ARCQ current-version state_dict contains forbidden legacy "
                "terminal-H keys: "
                + ", ".join(sorted(observed_terminal))
            )
        return None

    # A versionless payload without terminal-H tensors is not identifiable as
    # the complete v1 format.  Let ordinary strict loading report the missing
    # version key; non-strict, intentionally partial transfer remains possible.
    if not observed_terminal:
        return None

    active_local = {
        name: value
        for name, value in current_state.items()
        if name != _ARCQ_STATE_VERSION_KEY
    }
    active_names = {prefix + name for name in active_local}
    provided_local = {name for name in state_dict if name.startswith(prefix)}
    allowed_names = active_names | terminal_names

    problems: list[str] = []
    missing_terminal = terminal_names - observed_terminal
    extra_terminal = observed_terminal - terminal_names
    injectable_names = {prefix + name for name in numerics_local}
    missing_active = active_names - provided_local - injectable_names
    extra_local = provided_local - allowed_names
    if missing_terminal:
        problems.append("missing legacy keys=" + ", ".join(sorted(missing_terminal)))
    if extra_terminal:
        problems.append("extra legacy keys=" + ", ".join(sorted(extra_terminal)))
    if missing_active:
        problems.append("missing active keys=" + ", ".join(sorted(missing_active)))
    if extra_local:
        problems.append("extra backbone keys=" + ", ".join(sorted(extra_local)))

    for local_name, expected_shape in terminal_shapes.items():
        full_name = prefix + local_name
        if full_name in state_dict:
            supplied = state_dict[full_name]
            if supplied.shape != expected_shape:
                problems.append(
                    f"shape {full_name}={tuple(supplied.shape)} "
                    f"expected={tuple(expected_shape)}"
                )
            if not torch.is_floating_point(supplied):
                problems.append(
                    f"dtype {full_name}={supplied.dtype} expected=floating-point"
                )
    for local_name, expected_tensor in active_local.items():
        full_name = prefix + local_name
        if (
            full_name in state_dict
            and state_dict[full_name].shape != expected_tensor.shape
        ):
            problems.append(
                f"shape {full_name}={tuple(state_dict[full_name].shape)} "
                f"expected={tuple(expected_tensor.shape)}"
            )

    if problems:
        error_msgs.append(
            "Invalid ARCQ v1 legacy state_dict; terminal-H migration is "
            "all-or-nothing: " + "; ".join(problems)
        )
        return None

    for name in terminal_names:
        state_dict.pop(name)
    injected_keys = inject_v3_numerics()
    state_dict[version_name] = expected_version.clone()
    audit = {
        "migration": "arcq_v1_dead_terminal_h_to_v3",
        "source_version": 1,
        "target_version": _ARCQ_STATE_VERSION,
        "dropped_keys": tuple(sorted(name[len(prefix) :] for name in terminal_names)),
        "injected_keys": injected_keys,
    }
    warnings.warn(
        "Loaded a complete ARCQ v1 state_dict by dropping only the dead "
        "terminal-H parameters and adopting v3 FP32-H/smooth-observability "
        "semantics; optimizer/scheduler continuation from that checkpoint is "
        "not compatible with ARCQ state v3.",
        ARCQLegacyStateDictMigrationWarning,
        stacklevel=4,
    )
    return audit


def _as_int_tuple(values: Sequence[int], *, name: str, length: int = 4) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != int(length):
        raise ValueError(f"{name} must contain exactly {length} values, got {result}")
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers, got {result}")
    return result


def _reflect_pad(x: torch.Tensor, padding: int) -> torch.Tensor:
    padding = int(padding)
    if padding <= 0:
        return x
    if x.shape[-2] <= padding or x.shape[-1] <= padding:
        # This is only expected in deliberately tiny unit tests. Replication
        # keeps the operation defined without introducing zero-padding edges.
        return F.pad(x, (padding, padding, padding, padding), mode="replicate")
    return F.pad(x, (padding, padding, padding, padding), mode="reflect")


def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    groups = int(groups)
    if groups <= 1:
        return x
    batch, channels, height, width = x.shape
    if channels % groups != 0:
        raise ValueError(f"cannot shuffle {channels} channels into {groups} groups")
    return (
        x.reshape(batch, groups, channels // groups, height, width)
        .transpose(1, 2)
        .contiguous()
        .reshape(batch, channels, height, width)
    )


class DropPath(nn.Module):
    """Per-sample stochastic depth without an external timm dependency."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= float(probability) < 1.0:
            raise ValueError(f"drop-path probability must be in [0, 1), got {probability}")
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x * mask / keep


class LayerNorm2d(nn.Module):
    """Layer normalization over channels at every spatial position."""

    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.channels))
        self.bias = nn.Parameter(torch.zeros(self.channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"LayerNorm2d expected Bx{self.channels}xHxW, got {tuple(x.shape)}"
            )
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class _NormalizedConv2d(nn.Module):
    """Bias-free, weight-normalized convolution used only in the H carrier.

    The projected weight is still a linear operator. Consequently, the small
    numerical epsilon used to normalize the *parameter* does not enter the H
    activation and cannot break positive input homogeneity.
    """

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
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must both be divisible by groups")
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
        # ``model.half()`` / ``model.to(dtype=...)`` must not quantize the
        # theorem-bearing H parameters before forward casts them back to FP32.
        self.weight.data = self.weight.data.float()
        if self.weight.grad is not None:
            self.weight.grad.data = self.weight.grad.data.float()
        return self

    def projected_weight(self) -> torch.Tensor:
        # H is a theorem-bearing path. Keep its parameter projection in FP32
        # even when an outer AMP context requests bf16/fp16 convolution.
        flat = self.weight.float().flatten(1)
        norm = flat.square().sum(dim=1, keepdim=True).sqrt().clamp_min(self.eps)
        return (flat / norm).reshape(self.weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = _reflect_pad(x.float(), self.kernel_size // 2)
            return F.conv2d(
                x,
                self.projected_weight(),
                bias=None,
                stride=self.stride,
                groups=self.groups,
            )


class MatchedResponseConv2d(nn.Module):
    """带严格约束的可学习方向—相位匹配滤波器组。

    输出通道顺序固定为 ``组 → 方向 → (偶相位, 奇相位)``。每组只学习一对
    canonical 残差，再旋转成多个方向，因此不同方向共享同一径向基础，而不是
    四套互不相关的卷积。每次 forward 前投影保证：空间均值为零、滤波器范数
    为一、偶奇两相位正交。这样能量和方向比例具有稳定可比较的尺度。

    The cosine/sine anchors are quadrature-like at initialization.  Learning
    preserves zero DC, unit norm and pairwise orthogonality, but it does not
    impose the Hilbert-transform relation required to call every trained pair
    a strict analytic quadrature pair.

    A group owns one learnable canonical even/odd pair. The pair is rotated to
    the requested orientations, then every input-channel slice is projected to
    zero spatial DC. A differentiable Gram-Schmidt projection makes the two
    phases orthogonal and gives every output filter unit norm.

    Output channel order is ``group -> orientation -> (even, odd)``.
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
            raise ValueError(
                f"matched response kernel_size must be an odd integer greater than one, got {kernel_size}"
            )
        if orientations <= 0:
            raise ValueError("orientations must be positive")
        response_width = 2 * orientations
        if out_channels % response_width != 0:
            raise ValueError(
                f"out_channels={out_channels} must be divisible by 2*orientations={response_width}"
            )
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

        # 可学习量不是完整滤波器，而是固定非退化 anchor 周围的小残差。
        # 将残差范数限制在 0.25 以下，防止训练把 anchor 抵消成零/共线滤波器。
        # The learnable object is a residual around a fixed, non-degenerate
        # cosine/sine phase-pair anchor. Its projected norm is strictly less
        # than 0.25, so
        # it cannot cancel the unit anchor or make the two phases collinear.
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
        radius2 = xx.square() + yy.square()
        envelope = torch.exp(-radius2 / 0.42)
        grids = MatchedResponseConv2d._make_rotation_grids(size, int(orientations))
        groups = []
        for group_index in range(int(num_groups)):
            # Groups cover nearby radial bands and deterministic channel
            # mixtures. Directions inside one group still share exactly the
            # same radial basis, as required by the quotient construction.
            radial_fraction = (float(group_index) + 0.5) / float(num_groups)
            frequency = math.pi * (0.85 + 0.70 * radial_fraction)
            even = envelope * torch.cos(frequency * xx)
            odd = envelope * torch.sin(frequency * xx)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                104729
                + 1009 * int(size)
                + 9176 * int(in_channels)
                + 37 * int(group_index)
            )
            channel_gains = torch.randn(int(in_channels), generator=generator)
            channel_gains = channel_gains / channel_gains.norm().clamp_min(1.0e-8)
            base = (
                torch.stack([even, odd], dim=0)[:, None]
                * channel_gains[None, :, None, None]
            )
            base = base.reshape(2 * int(in_channels), 1, size, size)
            orientations_for_group = []
            for orientation in range(int(orientations)):
                grid = grids[orientation : orientation + 1].expand(
                    base.shape[0],
                    -1,
                    -1,
                    -1,
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
                        2,
                        int(in_channels),
                        size,
                        size,
                    )
                )
            groups.append(torch.stack(orientations_for_group, dim=0))
        return torch.stack(groups, dim=0)

    def _rotated_canonical(self) -> torch.Tensor:
        # Rotate every group/phase/input slice with the same orientation grid;
        # this makes the four directions share one learnable radial/canonical
        # basis rather than learning four unrelated filter banks.
        base = self.canonical_weight.float().reshape(
            self.num_groups * 2 * self.in_channels,
            1,
            self.kernel_size,
            self.kernel_size,
        )
        grids = self._rotation_grids.to(device=base.device, dtype=base.dtype)
        rotated = []
        for orientation in range(self.orientations):
            grid = grids[orientation : orientation + 1].expand(base.shape[0], -1, -1, -1)
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
        """生成真正参与卷积的受约束权重；不直接使用裸参数。"""
        residual = self._rotated_canonical()
        # 第一步：每个输入通道独立去空间均值（零 DC），排除亮度常量响应。
        residual = residual - residual.mean(dim=(-2, -1), keepdim=True)
        residual = residual.flatten(3)
        residual_norm = residual.square().sum(dim=-1, keepdim=True).add(
            self.eps**2
        ).sqrt()
        residual = (
            self.max_residual_norm * residual / (1.0 + residual_norm)
        )
        anchor = self._anchor_weight.to(
            device=residual.device,
            dtype=residual.dtype,
        )
        anchor = anchor.flatten(3)

        # 第二步：anchor+残差后归一化偶相位；第三步 Gram-Schmidt 正交化奇相位。
        even = anchor[:, :, 0] + residual[:, :, 0]
        # ||residual|| < r and ||anchor|| = 1, hence this norm is
        # structurally bounded below by 1-r and needs no clamp/fallback.
        even_norm = even.square().sum(dim=-1, keepdim=True).sqrt()
        even = even / even_norm

        odd = anchor[:, :, 1] + residual[:, :, 1]
        projection = (odd * even).sum(dim=-1, keepdim=True)
        odd = odd - projection * even
        # Let ||d|| < r for the even residual. Since ||a+d|| >= 1-r,
        # the normalized even filter obeys
        # |<b, even>| <= r/(1-r) for the orthogonal odd anchor b.
        # Projecting b onto even^\perp therefore retains at least
        # sqrt(1-(r/(1-r))^2). The projected odd residual can remove less than
        # r. With the enforced r<=0.25, the remaining norm is bounded below by
        # sqrt(8/9)-1/4 > 0.69 before normalization.
        odd_norm = odd.square().sum(dim=-1, keepdim=True).sqrt()
        odd = odd / odd_norm

        pair = torch.stack([even, odd], dim=2)
        return pair.reshape(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

    @torch.no_grad()
    def zero_dc_error(self) -> torch.Tensor:
        weight = self.projected_weight()
        return weight.sum(dim=(-2, -1)).abs().amax()

    @torch.no_grad()
    def unit_norm_error(self) -> torch.Tensor:
        weight = self.projected_weight().flatten(1)
        return (weight.norm(dim=1) - 1.0).abs().amax()

    @torch.no_grad()
    def pair_orthogonality_error(self) -> torch.Tensor:
        weight = self.projected_weight().reshape(
            self.num_groups,
            self.orientations,
            2,
            -1,
        )
        return (weight[:, :, 0] * weight[:, :, 1]).sum(dim=-1).abs().amax()

    def forward(
        self,
        x: torch.Tensor,
        *,
        stride: int | None = None,
        dilation: int = 1,
    ) -> torch.Tensor:
        """对输入执行受约束匹配卷积，返回 ``[B,C_out,H',W']`` 响应账本。"""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"MatchedResponseConv2d expected Bx{self.in_channels}xHxW, got {tuple(x.shape)}"
            )
        effective_stride = self.stride if stride is None else int(stride)
        if effective_stride <= 0:
            raise ValueError(f"stride must be positive, got {effective_stride}")
        effective_dilation = int(dilation)
        if effective_dilation <= 0:
            raise ValueError(f"dilation must be positive, got {effective_dilation}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            if effective_dilation == 1:
                # Preserve ARCQ's historic default arithmetic and convolution
                # call literally.  PCQT's dilation-2 read is an optional
                # branch; its existence must not perturb default checkpoints.
                x = _reflect_pad(x.float(), self.kernel_size // 2)
                return F.conv2d(
                    x,
                    self.projected_weight(),
                    bias=None,
                    stride=effective_stride,
                )
            x = _reflect_pad(
                x.float(),
                effective_dilation * (self.kernel_size // 2),
            )
            return F.conv2d(
                x,
                self.projected_weight(),
                bias=None,
                stride=effective_stride,
                dilation=effective_dilation,
            )


class RadialCompositionQuotient(nn.Module):
    """把匹配响应分解为径向能量与相对方向组成坐标。

    对每个方向的偶/奇相位先平方求和得到相位不敏感能量。总能量描述局部响应
    是否可观测；各方向能量占比描述纹理方向组成。平滑项只用于数值稳定，最终
    输出的 ``composition`` 与 ``log_energy`` 由上层模型用于可靠性和 RFME。
    """

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
        composition_coordinate_mode: str = _ILR_ARCQ_CENTERED_SIMPLEX,
    ) -> None:
        super().__init__()
        channels = int(channels)
        group_width = int(group_width)
        orientations = int(orientations)
        if group_width != 2 * orientations:
            raise ValueError(
                "group_width must equal two matched phases times orientations; "
                f"got group_width={group_width}, orientations={orientations}"
            )
        if channels % group_width != 0:
            raise ValueError(f"channels={channels} must be divisible by group_width={group_width}")
        if orientations < 2:
            raise ValueError("orientations must be at least two")
        if not math.isfinite(float(smoothing)) or float(smoothing) < 0.0:
            raise ValueError("smoothing must be finite and non-negative")
        if not math.isfinite(float(numerical_eps)) or float(numerical_eps) <= 0.0:
            raise ValueError("numerical_eps must be finite and positive")
        if (
            not math.isfinite(float(observability_floor))
            or float(observability_floor) <= float(numerical_eps)
        ):
            raise ValueError(
                "observability_floor must be finite and greater than numerical_eps"
            )
        if (
            not math.isfinite(float(fully_observable_energy))
            or float(fully_observable_energy) <= float(observability_floor)
        ):
            raise ValueError(
                "fully_observable_energy must be finite and greater than "
                "observability_floor"
            )
        composition_coordinate_mode = str(composition_coordinate_mode).strip().lower()
        if composition_coordinate_mode not in _ILR_ARCQ_VALID_MODES:
            raise ValueError(
                "composition_coordinate_mode must be one of "
                f"{sorted(_ILR_ARCQ_VALID_MODES)}, got "
                f"{composition_coordinate_mode!r}"
            )
        # The ILR map contains log(p).  Its mathematical contract is a
        # strictly positive simplex, not an arbitrary clamp at a zero
        # component.  The standard ARCQ modes may still set smoothing=0 for
        # an ablation, but the active ILR arm must retain a positive
        # pseudocount so the claimed geometry and common-scale theorem are
        # actually the geometry that reaches the protected C stream.
        if (
            composition_coordinate_mode == _ILR_ARCQ_ILR_HELMERT
            and float(smoothing) <= 0.0
        ):
            raise ValueError(
                "ilr_helmert requires smoothing > 0 so every composition "
                "coordinate is strictly positive before log-ratio mapping"
            )
        self.channels = channels
        self.group_width = group_width
        self.orientations = orientations
        self.num_groups = channels // group_width
        self.composition_coordinate_mode = composition_coordinate_mode
        # These scalars define the quotient's numerical semantics, not merely
        # optimizer hyperparameters. Persist them so a strict checkpoint load
        # cannot silently evaluate identical weights under different v3 rules.
        self.register_buffer(
            "_simplex_smoothing",
            torch.tensor(float(smoothing), dtype=torch.float64),
        )
        self.register_buffer(
            "_numerical_eps",
            torch.tensor(float(numerical_eps), dtype=torch.float64),
        )
        self.register_buffer(
            "_observability_floor",
            torch.tensor(float(observability_floor), dtype=torch.float64),
        )
        self.register_buffer(
            "_fully_observable_energy",
            torch.tensor(float(fully_observable_energy), dtype=torch.float64),
        )
        # The legacy coordinate deliberately owns no mode or geometry buffers.
        # Active study arms serialize both the exact O-dimensional Helmert
        # basis and its projector, so a checkpoint cannot be reinterpreted as
        # another coordinate system under ``strict=False``.
        if self.composition_coordinate_mode != _ILR_ARCQ_CENTERED_SIMPLEX:
            helmert = _helmert_basis(self.orientations).to(dtype=torch.float32)
            self.register_buffer(
                "_ilr_arcq_schema_version",
                torch.tensor(_ILR_ARCQ_SCHEMA_VERSION, dtype=torch.int64),
            )
            self.register_buffer(
                "_ilr_arcq_mode_code",
                torch.tensor(
                    _ILR_ARCQ_ACTIVE_MODE_CODES[
                        self.composition_coordinate_mode
                    ],
                    dtype=torch.int64,
                ),
            )
            self.register_buffer("_ilr_arcq_helmert", helmert)
            self.register_buffer(
                "_ilr_arcq_projector",
                helmert @ helmert.transpose(0, 1),
            )
        self.reliability_logit = nn.Parameter(torch.full((self.num_groups,), -4.0))

    @property
    def smoothing(self) -> float:
        return float(self._simplex_smoothing.item())

    @property
    def numerical_eps(self) -> float:
        return float(self._numerical_eps.item())

    @property
    def observability_floor(self) -> float:
        return float(self._observability_floor.item())

    @property
    def fully_observable_energy(self) -> float:
        return float(self._fully_observable_energy.item())

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        super()._apply(fn, recurse=recurse)
        # These buffers are the serialized numerical definition. Preserve their
        # float64 values even if a caller globally casts the rest of the model.
        self._simplex_smoothing = self._simplex_smoothing.double()
        self._numerical_eps = self._numerical_eps.double()
        self._observability_floor = self._observability_floor.double()
        self._fully_observable_energy = self._fully_observable_energy.double()
        if self.composition_coordinate_mode != _ILR_ARCQ_CENTERED_SIMPLEX:
            # The ILR study geometry is intentionally FP32, including under a
            # global model half/bfloat16 cast.  Its operation is additionally
            # protected by an autocast-disabled block below.
            self._ilr_arcq_helmert = self._ilr_arcq_helmert.float()
            self._ilr_arcq_projector = self._ilr_arcq_projector.float()
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
        coordinate_names = (
            prefix + "_ilr_arcq_schema_version",
            prefix + "_ilr_arcq_mode_code",
            prefix + "_ilr_arcq_helmert",
            prefix + "_ilr_arcq_projector",
        )
        incoming_coordinate_mode = any(name in state_dict for name in coordinate_names)
        active_coordinate_mode = (
            self.composition_coordinate_mode != _ILR_ARCQ_CENTERED_SIMPLEX
        )
        if not active_coordinate_mode and incoming_coordinate_mode:
            error_msgs.append(
                "active ILR-ARCQ checkpoint cannot be loaded into a legacy "
                "centered_simplex quotient, even with strict=False"
            )
        elif active_coordinate_mode and not incoming_coordinate_mode:
            error_msgs.append(
                "legacy centered_simplex ARCQ checkpoint cannot be silently "
                "loaded into an active ILR-ARCQ coordinate mode"
            )
        elif active_coordinate_mode:
            expected_coordinate_buffers = {
                "schema": (
                    prefix + "_ilr_arcq_schema_version",
                    self._ilr_arcq_schema_version,
                ),
                "mode": (
                    prefix + "_ilr_arcq_mode_code",
                    self._ilr_arcq_mode_code,
                ),
                "Helmert basis": (
                    prefix + "_ilr_arcq_helmert",
                    self._ilr_arcq_helmert,
                ),
                "projector": (
                    prefix + "_ilr_arcq_projector",
                    self._ilr_arcq_projector,
                ),
            }
            missing_coordinate_buffers = [
                semantic_name
                for semantic_name, (state_name, _expected) in (
                    expected_coordinate_buffers.items()
                )
                if state_name not in state_dict
            ]
            if missing_coordinate_buffers:
                error_msgs.append(
                    "ILR-ARCQ checkpoint must contain schema, mode, Helmert "
                    "basis and projector buffers; missing "
                    + ", ".join(missing_coordinate_buffers)
                )
            else:
                for semantic_name, (
                    state_name,
                    expected,
                ) in expected_coordinate_buffers.items():
                    supplied = state_dict[state_name]
                    valid_type_and_shape = (
                        torch.is_tensor(supplied)
                        and supplied.dtype == expected.dtype
                        and supplied.shape == expected.shape
                    )
                    valid_value = valid_type_and_shape and torch.equal(
                        supplied.detach().cpu(),
                        expected.detach().cpu(),
                    )
                    if valid_value:
                        continue
                    if not valid_type_and_shape:
                        error_msgs.append(
                            "ILR-ARCQ checkpoint "
                            f"{semantic_name} buffer dtype/shape does not match "
                            "the instantiated quotient"
                        )
                    elif semantic_name == "mode":
                        error_msgs.append(
                            "ILR-ARCQ checkpoint mode does not match the "
                            "instantiated coordinate mode "
                            f"{self.composition_coordinate_mode!r}"
                        )
                    elif semantic_name == "schema":
                        error_msgs.append(
                            "ILR-ARCQ checkpoint schema does not match the "
                            "instantiated quotient"
                        )
                    else:
                        error_msgs.append(
                            "ILR-ARCQ checkpoint "
                            f"{semantic_name} does not match the instantiated "
                            "fixed geometry"
                        )
                    # A failed non-strict load must not mutate the target's
                    # coordinate identity before it raises its hard error.
                    state_dict[state_name] = expected.detach().clone()
        names = {
            "smoothing": prefix + "_simplex_smoothing",
            "numerical_eps": prefix + "_numerical_eps",
            "observability_floor": prefix + "_observability_floor",
            "fully_observable_energy": prefix + "_fully_observable_energy",
        }
        current_values = {
            "smoothing": float(self._simplex_smoothing.item()),
            "numerical_eps": float(self._numerical_eps.item()),
            "observability_floor": float(self._observability_floor.item()),
            "fully_observable_energy": float(
                self._fully_observable_energy.item()
            ),
        }
        values = dict(current_values)
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
                    f'Invalid ARCQ v3 numerical buffer "{state_name}": '
                    "expected a scalar float64 tensor."
                )
                continue
            values[semantic_name] = float(supplied.item())
        # ``strict=False`` permits a partial checkpoint. Validate the effective
        # quartet (checkpoint values merged with the current module values),
        # rather than validating only when all four entries happen to be
        # present. This prevents flexible transfer from silently constructing
        # an invalid mixed numerical domain.
        smoothing = values["smoothing"]
        numerical_eps = values["numerical_eps"]
        observability_floor = values["observability_floor"]
        fully_observable_energy = values["fully_observable_energy"]
        valid = (
            math.isfinite(smoothing)
            and smoothing >= 0.0
            and math.isfinite(numerical_eps)
            and numerical_eps > 0.0
            and math.isfinite(observability_floor)
            and observability_floor > numerical_eps
            and math.isfinite(fully_observable_energy)
            and fully_observable_energy > observability_floor
        )
        if not valid:
            error_msgs.append(
                f'Invalid ARCQ v3 numerical domain at "{prefix}": '
                f"smoothing={smoothing}, numerical_eps={numerical_eps}, "
                f"observability_floor={observability_floor}, "
                f"fully_observable_energy={fully_observable_energy}."
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

    @property
    def composition_channels(self) -> int:
        return self.num_groups * self.orientations

    def directional_energy(self, response: torch.Tensor) -> torch.Tensor:
        """Return phase-summed directional energy in ``B x G x O x H x W``.

        This intentionally exposes the positive quantity *before* quotient
        normalization.  It is used by the ECQD causal screen to distinguish
        ``P(r)^2`` from ``P(r^2)`` without changing ARCQ's default response
        path.  The calculation is in FP32 for half inputs, exactly as the
        original quotient path was.
        """

        if response.ndim != 4 or response.shape[1] != self.channels:
            raise ValueError(
                f"RadialCompositionQuotient expected Bx{self.channels}xHxW, got {tuple(response.shape)}"
            )
        work = response.float() if response.dtype in {torch.float16, torch.bfloat16} else response
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

    def _active_composition_coordinate(
        self,
        probabilities: torch.Tensor,
        centered: torch.Tensor,
    ) -> torch.Tensor:
        """Return one active study coordinate on ARCQ's O-channel interface.

        Let ``V`` be the fixed O-by-(O-1) orthonormal Helmert basis and
        ``P = V V^T``.  A is the literal centered-simplex control, B is the
        linear Helmert projection ``P c``, and C reconstructs the normalized
        ILR coordinate ``(1/O) V(V^T log(p))``.  B/C arithmetic is deliberately
        FP32 with autocast disabled; the returned tensor restores the existing
        quotient working dtype before the outer forward restores half outputs.
        """

        mode = self.composition_coordinate_mode
        if mode == _ILR_ARCQ_CENTERED_SIMPLEX_CONTROL:
            return centered
        if mode not in {
            _ILR_ARCQ_HELMERT_LINEAR_CONTROL,
            _ILR_ARCQ_ILR_HELMERT,
        }:
            raise RuntimeError(
                "active ILR-ARCQ coordinate requested for invalid mode "
                f"{mode!r}"
            )
        output_dtype = centered.dtype
        with torch.autocast(device_type=probabilities.device.type, enabled=False):
            helmert = self._ilr_arcq_helmert.to(
                device=probabilities.device,
                dtype=torch.float32,
            )
            if mode == _ILR_ARCQ_HELMERT_LINEAR_CONTROL:
                # P is registered alongside V as part of the checkpointed
                # semantic contract and was constructed exactly as V @ V.T.
                projector = self._ilr_arcq_projector.to(
                    device=centered.device,
                    dtype=torch.float32,
                )
                coordinate = torch.einsum(
                    "oi,bgihw->bgohw",
                    projector,
                    centered.float(),
                )
            else:
                log_probability = probabilities.float().clamp_min(1.0e-6).log()
                ilr = torch.einsum(
                    "od,bgohw->bgdhw",
                    helmert,
                    log_probability,
                )
                coordinate = torch.einsum(
                    "od,bgdhw->bgohw",
                    helmert,
                    ilr,
                ) / float(self.orientations)
                # In exact arithmetic, the projection of the uniform
                # composition is zero.  Finite FP32 matrix products can leave
                # an O(1e-8) residual, which would fabricate a protected
                # texture direction precisely where ARCQ's observability
                # contract says there is no direction evidence.  Centered is
                # exactly zero both below the observability floor and for a
                # genuinely uniform directional composition, so make that
                # mathematically exact case literal rather than relying on an
                # arbitrary threshold or the log clamp.
                zero_centered = centered.eq(0).all(dim=2, keepdim=True)
                coordinate = torch.where(
                    zero_centered,
                    torch.zeros_like(coordinate),
                    coordinate,
                )
        return coordinate.to(dtype=output_dtype)

    def forward_from_directional_energy(
        self,
        energy: torch.Tensor,
        *,
        output_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build the ordinary ARCQ quotient from pre-computed directional energy.

        ``energy`` must contain the non-negative even/odd phase-summed
        responses in ``B x G x O x H x W`` order.  This is deliberately not a
        generic learned pooling layer: it preserves the exact existing
        observability, simplex smoothing, radial and reliability semantics
        after a caller has chosen how to form the energy.
        """

        if energy.ndim != 5:
            raise ValueError(
                "RadialCompositionQuotient directional energy must be "
                f"Bx{self.num_groups}x{self.orientations}xHxW, got {tuple(energy.shape)}"
            )
        batch, groups, orientations, height, width = energy.shape
        if groups != self.num_groups or orientations != self.orientations:
            raise ValueError(
                "RadialCompositionQuotient directional-energy geometry mismatch: "
                f"expected G={self.num_groups}, O={self.orientations}; "
                f"got G={groups}, O={orientations}"
            )
        original_dtype = output_dtype
        work_energy = (
            energy.float()
            if energy.dtype in {torch.float16, torch.bfloat16}
            else energy
        )
        # Do not call ``.all()``/``.any()`` here: those checks would force a
        # host-device synchronization on every training batch.  Numerical
        # finiteness/non-negativity is asserted by the focused unit tests and
        # by the normal loss finite checks; this primitive stays fully
        # differentiable and asynchronous in the hot path.
        energy = work_energy
        total = energy.sum(dim=2)
        # No non-trivial scale-invariant direction coordinate can be both
        # continuous and informative at the zero vector. Use a three-region
        # observability model: no directional evidence below a fixed floor, a
        # C1 smoothstep transition, and the exact positive-scale quotient
        # above fully_observable_energy. This removes the old hard-threshold
        # jump and bounds the near-zero gradient seen by the protected state.
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
        quotient_centered = numerator / denominator - uniform
        centered = observability.unsqueeze(2) * quotient_centered
        probabilities = uniform + centered

        log_energy = 0.5 * torch.log(
            total.clamp_min(self._numerical_eps.to(total))
        )
        nu = 1.0e-6 + (1.0 - 1.0e-6) * torch.sigmoid(self.reliability_logit)
        reliability = total / (total + nu[None, :, None, None])

        # Keep the historic centered-simplex path literal unless an explicit
        # active study arm asks for a different fixed coordinate geometry.
        composition = centered
        if self.composition_coordinate_mode != _ILR_ARCQ_CENTERED_SIMPLEX:
            composition = self._active_composition_coordinate(
                probabilities,
                centered,
            )

        result = {
            "composition": composition.reshape(
                batch,
                self.composition_channels,
                height,
                width,
            ),
            "log_energy": log_energy,
            "reliability": reliability,
            "observability": observability,
            "probabilities": probabilities,
            "total_energy": total,
        }
        if original_dtype in {torch.float16, torch.bfloat16}:
            result = {name: value.to(dtype=original_dtype) for name, value in result.items()}
        return result

    def forward(self, response: torch.Tensor) -> dict[str, torch.Tensor]:
        if response.ndim != 4 or response.shape[1] != self.channels:
            raise ValueError(
                f"RadialCompositionQuotient expected Bx{self.channels}xHxW, got {tuple(response.shape)}"
            )
        return self.forward_from_directional_energy(
            self.directional_energy(response),
            output_dtype=response.dtype,
        )


class HomogeneousCarrierBlock(nn.Module):
    """Positive-homogeneous residual block for the isolated H stream."""

    def __init__(self, channels: int, *, residual_scale: float = 1.0e-3) -> None:
        super().__init__()
        channels = int(channels)
        self.depthwise = _NormalizedConv2d(
            channels,
            channels,
            3,
            groups=channels,
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
            x = x.float()
            activated = F.prelu(
                self.depthwise(x),
                self.activation.weight.float(),
            )
            return x + self.residual_scale.float() * self.pointwise(activated)


class CompositionBlock(nn.Module):
    """Protected C-state update; it has no interface through which A can enter."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 2,
        layer_scale: float = 1.0e-6,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        hidden = int(expansion) * channels
        pointwise_groups = math.gcd(2, channels)
        self.norm = LayerNorm2d(channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            groups=channels,
            bias=True,
        )
        self.pointwise_groups = pointwise_groups
        self.expand = nn.Conv2d(
            channels,
            hidden,
            kernel_size=1,
            groups=pointwise_groups,
            bias=True,
        )
        self.contract = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1,
            groups=pointwise_groups,
            bias=True,
        )
        self.layer_scale = nn.Parameter(torch.full((channels, 1, 1), float(layer_scale)))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.norm(x)
        update = self.depthwise(_reflect_pad(update, 2))
        update = F.gelu(self.expand(update))
        update = _channel_shuffle(update, self.pointwise_groups)
        update = self.contract(update)
        return x + self.drop_path(self.layer_scale * update)


class AppearanceAmplitudeBlock(nn.Module):
    """A-state block with parallel local and broader appearance operators."""

    def __init__(
        self,
        channels: int,
        *,
        expansion: int = 2,
        layer_scale: float = 1.0e-6,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        hidden = int(expansion) * channels
        pointwise_groups = math.gcd(2, channels)
        self.norm = LayerNorm2d(channels)
        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            groups=channels,
            bias=True,
        )
        self.broad = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            groups=channels,
            bias=True,
        )
        self.branch_logits = nn.Parameter(torch.zeros(2, channels, 1, 1))
        self.pointwise_groups = pointwise_groups
        self.expand = nn.Conv2d(
            channels,
            hidden,
            kernel_size=1,
            groups=pointwise_groups,
            bias=True,
        )
        self.contract = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1,
            groups=pointwise_groups,
            bias=True,
        )
        self.layer_scale = nn.Parameter(torch.full((channels, 1, 1), float(layer_scale)))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        branches = torch.stack(
            [
                self.local(_reflect_pad(z, 1)),
                self.broad(_reflect_pad(z, 3)),
            ],
            dim=1,
        )
        branch_weights = torch.softmax(self.branch_logits, dim=0)
        z = (branches * branch_weights.unsqueeze(0)).sum(dim=1)
        z = F.gelu(self.expand(z))
        z = _channel_shuffle(z, self.pointwise_groups)
        z = self.contract(z)
        return x + self.drop_path(self.layer_scale * z)


class _SpatialTransition(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.norm = LayerNorm2d(in_channels)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            groups=in_channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(_reflect_pad(self.norm(x), 1))
        return self.pointwise(x)


class _BarycentricSpatialTransition(nn.Module):
    """Parameter-matched terminal transition for the frozen LBET causal gate.

    Both active arms store an unconstrained 3x3 tensor in ``depthwise.weight``.
    The tensor is reparameterized at every forward pass.  At zero logits the
    affine control and the coverage-barycentric candidate have the same
    effective uniform kernel and the same first-order Jacobian with respect to
    those logits.  Only the candidate remains a positive convex combination
    away from initialization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        mode: str,
        coverage: float,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        mode = str(mode).strip().lower()
        if mode not in _FINAL_A_TRANSITION_ACTIVE_MODE_CODES:
            raise ValueError(
                "active LBET transition mode must be one of "
                f"{sorted(_FINAL_A_TRANSITION_ACTIVE_MODE_CODES)}, got {mode!r}"
            )
        coverage = float(coverage)
        if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
            raise ValueError("LBET coverage must be finite and lie strictly in (0, 1)")
        self.mode = mode
        self.coverage = coverage
        self.norm = LayerNorm2d(in_channels)
        # Keep the same Conv2d parameter topology and RNG consumption as the
        # historical transition.  Its stored weight is interpreted as logits.
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            groups=in_channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

    def initialize_matched_logits(self) -> None:
        """Set B/C to the same uniform function after the shared initializer."""

        with torch.no_grad():
            self.depthwise.weight.zero_()
            if self.depthwise.bias is not None:
                self.depthwise.bias.zero_()

    def effective_kernel(self) -> torch.Tensor:
        """Return the differentiable per-channel 3x3 effective kernel."""

        logits = self.depthwise.weight.reshape(self.depthwise.in_channels, 9)
        uniform = torch.full_like(logits, 1.0 / 9.0)
        residual_budget = 1.0 - self.coverage
        if self.mode == _FINAL_A_TRANSITION_AFFINE_TANGENT:
            centered = logits - logits.mean(dim=1, keepdim=True)
            kernel = uniform + (residual_budget / 9.0) * centered
        elif self.mode == _FINAL_A_TRANSITION_COVERAGE_BARYCENTRIC:
            kernel = (
                self.coverage * uniform
                + residual_budget * torch.softmax(logits, dim=1)
            )
        else:  # pragma: no cover - constructor and checkpoint gates forbid it.
            raise RuntimeError(f"unexpected active LBET mode {self.mode!r}")
        return kernel.reshape_as(self.depthwise.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        kernel = self.effective_kernel().to(
            device=normalized.device,
            dtype=normalized.dtype,
        )
        x = F.conv2d(
            _reflect_pad(normalized, 1),
            kernel,
            bias=(
                None
                if self.depthwise.bias is None
                else self.depthwise.bias.to(
                    device=normalized.device,
                    dtype=normalized.dtype,
                )
            ),
            stride=2,
            groups=self.depthwise.in_channels,
        )
        return self.pointwise(x)


class _RadiometricStatisticsSidecar(nn.Module):
    """Carry channel-common appearance statistics across one LN transition.

    ``_SpatialTransition`` deliberately normalizes every spatial position
    across channels.  That is useful for pattern refinement, but it also
    removes the channel-common mean and scale that the A state is supposed to
    retain as radiometric evidence.  This sidecar is an explicitly optional,
    A-only diagnostic: it records those two discarded coordinates before the
    final transition, downsamples them with a fixed local average, and writes
    them back through a very small learned projection.

    Statistics are accumulated in FP32 for mixed-precision stability.  The
    projected update is multiplied by one near-zero scalar so the candidate
    starts close to the unmodified ARCQ function without creating an A->C
    path.
    """

    def __init__(
        self,
        out_channels: int,
        *,
        initial_scale: float = 1.0e-3,
        numerical_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        out_channels = int(out_channels)
        initial_scale = float(initial_scale)
        numerical_eps = float(numerical_eps)
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if not math.isfinite(initial_scale) or not 0.0 <= initial_scale <= 1.0:
            raise ValueError("initial_scale must be finite and in [0, 1]")
        if not math.isfinite(numerical_eps) or numerical_eps <= 0.0:
            raise ValueError("numerical_eps must be finite and positive")

        self.projection = nn.Conv2d(2, out_channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))
        self.numerical_eps = numerical_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "radiometric statistics sidecar expected BxCxHxW, "
                f"got {tuple(x.shape)}"
            )

        # Keep the reduction outside low-precision autocast arithmetic.  The
        # result is cast back before the learned projection so parameter and
        # activation dtypes remain compatible under BF16/FP16 training.
        source = x.float()
        channel_mean = source.mean(dim=1, keepdim=True)
        centered = source - channel_mean
        log_rms = 0.5 * torch.log(
            centered.square().mean(dim=1, keepdim=True) + self.numerical_eps
        )
        statistics = torch.cat([channel_mean, log_rms], dim=1)
        statistics = F.avg_pool2d(
            _reflect_pad(statistics, 1),
            kernel_size=3,
            stride=2,
        )
        update = self.projection(statistics.to(dtype=x.dtype))
        return self.scale.to(dtype=update.dtype) * update


def _helmert_basis(parts: int) -> torch.Tensor:
    """Return the orthonormal ILR basis for a ``parts``-simplex.

    The columns span the zero-sum subspace, so translating probabilities into
    this basis removes the arbitrary all-ones log-probability coordinate while
    preserving Euclidean distances in Aitchison geometry.
    """

    parts = int(parts)
    if parts < 2:
        raise ValueError(f"Helmert ILR basis needs at least two parts, got {parts}")
    basis = torch.zeros(parts, parts - 1, dtype=torch.float64)
    for column in range(parts - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


class RoleIdentifiedCAQCReadout(nn.Module):
    """Read a protected S3 radial-compositional ledger without writing it back.

    The first screen intentionally implements only ``local_q3``.  It asks one
    falsifiable question: does retaining the 45x30 quotient ledger improve the
    final decision before adding any global context anchor?  Later role anchors
    are deliberately absent rather than dormant, so this module cannot hide
    unused parameters or an A/C cross-role path.

    A zero-anchored nonlinear contrast is reliability-averaged per response
    group, projected without bias, RMS-normalized and added through a bounded
    scalar.  Consequently a sample with zero reliability produces an exactly
    zero residual for arbitrary learned weights.
    """

    VALID_MODES = frozenset({"local_q3"})
    _SCHEMA_VERSION = 1
    _MODE_CODES = {"local_q3": 1}
    _LOCAL_INIT_SALT = 0x52494301

    def __init__(
        self,
        *,
        mode: str,
        response_channels: int,
        group_width: int,
        orientations: int,
        context_channels: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"RI-CAQC mode must be one of {sorted(self.VALID_MODES)}, got {mode!r}"
            )
        response_channels = int(response_channels)
        group_width = int(group_width)
        orientations = int(orientations)
        context_channels = int(context_channels)
        out_dim = int(out_dim)
        if response_channels <= 0 or group_width <= 0 or out_dim <= 0:
            raise ValueError("response_channels, group_width and out_dim must be positive")
        if response_channels % group_width != 0:
            raise ValueError(
                f"response_channels={response_channels} must be divisible by "
                f"group_width={group_width}"
            )
        if group_width != 2 * orientations:
            raise ValueError("group_width must equal two phases times orientations")
        if context_channels <= 0:
            raise ValueError("context_channels must be positive")

        self.mode = mode
        self.response_channels = response_channels
        self.group_width = group_width
        self.orientations = orientations
        self.context_channels = context_channels
        self.out_dim = out_dim
        self.num_groups = response_channels // group_width
        self.coordinate_dim = orientations
        self.local_hidden = 8
        self.descriptor_dim = self.num_groups * self.local_hidden

        # Registration order is frozen for state-schema and initialization
        # comparability across subsequent role-anchored variants.
        self.psi = nn.Sequential(
            nn.Linear(self.coordinate_dim, self.local_hidden, bias=True),
            nn.GELU(),
            nn.Linear(self.local_hidden, self.local_hidden, bias=True),
        )
        self.local_projection = nn.Linear(
            self.descriptor_dim,
            self.out_dim,
            bias=False,
        )
        self.residual_norm = nn.RMSNorm(self.out_dim, eps=1.0e-6)
        initial_scale = 1.0e-3
        residual_max = 0.25
        self.raw_residual_scale = nn.Parameter(
            torch.tensor(
                math.atanh(initial_scale / residual_max),
                dtype=torch.float32,
            )
        )

        self.register_buffer(
            "_schema_version",
            torch.tensor(self._SCHEMA_VERSION, dtype=torch.int64),
        )
        self.register_buffer(
            "_mode_code",
            torch.tensor(self._MODE_CODES[self.mode], dtype=torch.int64),
        )
        self.register_buffer("_helmert", _helmert_basis(self.orientations))
        self.register_buffer(
            "_numerical_eps",
            torch.tensor(1.0e-6, dtype=torch.float64),
        )
        self.register_buffer(
            "_composition_logit_bound",
            torch.tensor(2.0, dtype=torch.float64),
        )
        self.register_buffer(
            "_radial_origin_bound",
            torch.tensor(2.0, dtype=torch.float64),
        )
        self.register_buffer(
            "_residual_max",
            torch.tensor(residual_max, dtype=torch.float64),
        )

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        super()._apply(fn, recurse=recurse)
        # These buffers define the coordinate system and numerical contract.
        self._helmert = self._helmert.double()
        self._numerical_eps = self._numerical_eps.double()
        self._composition_logit_bound = self._composition_logit_bound.double()
        self._radial_origin_bound = self._radial_origin_bound.double()
        self._residual_max = self._residual_max.double()
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
        expected = {
            prefix + "_schema_version": self._SCHEMA_VERSION,
            prefix + "_mode_code": self._MODE_CODES[self.mode],
        }
        for name, value in expected.items():
            supplied = state_dict.get(name)
            if supplied is None:
                continue
            valid = (
                torch.is_tensor(supplied)
                and supplied.shape == torch.Size([])
                and supplied.dtype == torch.int64
                and int(supplied.item()) == int(value)
            )
            if not valid:
                error_msgs.append(
                    f'Invalid RI-CAQC schema buffer "{name}": expected scalar '
                    f"int64 value {value}."
                )
                # Keep a failed load from mutating the module's identity.
                state_dict[name] = getattr(self, name[len(prefix) :]).detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def initialize_local_parameters(self, *, seed: int) -> None:
        """Initialize this isolated subtree without advancing global RNG."""

        modulus = (1 << 63) - 1
        generator = torch.Generator(device="cpu")
        generator.manual_seed((int(seed) + self._LOCAL_INIT_SALT) % modulus)
        for module in (self.psi, self.local_projection):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.trunc_normal_(
                        child.weight,
                        std=0.02,
                        generator=generator,
                    )
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        nn.init.ones_(self.residual_norm.weight)
        with torch.no_grad():
            self.raw_residual_scale.fill_(
                math.atanh(1.0e-3 / float(self._residual_max.item()))
            )

    def local_coordinates(
        self,
        probabilities: torch.Tensor,
        log_energy: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if probabilities.ndim != 5:
            raise ValueError(
                "RI-CAQC probabilities must be BxGxOxHxW, got "
                f"{tuple(probabilities.shape)}"
            )
        batch, groups, orientations, height, width = probabilities.shape
        expected_scalar = (batch, groups, height, width)
        if groups != self.num_groups or orientations != self.orientations:
            raise ValueError(
                "RI-CAQC quotient geometry mismatch: expected "
                f"G={self.num_groups}, O={self.orientations}, got "
                f"G={groups}, O={orientations}"
            )
        if tuple(log_energy.shape) != expected_scalar:
            raise ValueError(
                f"RI-CAQC log_energy must have shape {expected_scalar}, "
                f"got {tuple(log_energy.shape)}"
            )
        if tuple(reliability.shape) != expected_scalar:
            raise ValueError(
                f"RI-CAQC reliability must have shape {expected_scalar}, "
                f"got {tuple(reliability.shape)}"
            )

        with torch.autocast(device_type=probabilities.device.type, enabled=False):
            p = probabilities.float().clamp_min(1.0e-12)
            radial = log_energy.float()
            q = reliability.float().clamp(0.0, 1.0)
            eps = self._numerical_eps.to(q)
            support = q.sum(dim=(-2, -1), keepdim=True)
            radial_mean = (q * radial).sum(
                dim=(-2, -1), keepdim=True
            ) / (support + eps)
            radial_centered = radial - radial_mean
            ilr = torch.einsum(
                "bgohw,od->bgdhw",
                p.log(),
                self._helmert.to(device=p.device, dtype=p.dtype),
            )
            coordinates = torch.cat(
                [radial_centered.unsqueeze(2), ilr],
                dim=2,
            )
            coordinates = coordinates.permute(0, 1, 3, 4, 2).contiguous()
        return coordinates, q

    def interaction_contrast(
        self,
        coordinates: torch.Tensor,
        origin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.psi(coordinates - origin) - self.psi(-origin)
        zero_origin = torch.zeros_like(origin)
        h0 = self.psi(coordinates) - self.psi(zero_origin)
        return h, h0, h - h0

    def forward(
        self,
        *,
        probabilities: torch.Tensor,
        log_energy: torch.Tensor,
        reliability: torch.Tensor,
        appearance_state: torch.Tensor,
        composition_state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Context tensors are explicit, read-only interface placeholders.  The
        # local_q3 screen must not condition on either before it proves useful.
        if appearance_state.ndim != 4 or composition_state.ndim != 4:
            raise ValueError("RI-CAQC context states must both be BxCxHxW")
        if appearance_state.shape[0] != probabilities.shape[0]:
            raise ValueError("RI-CAQC appearance batch does not match Q3")
        if composition_state.shape[0] != probabilities.shape[0]:
            raise ValueError("RI-CAQC composition batch does not match Q3")

        with torch.autocast(device_type=probabilities.device.type, enabled=False):
            coordinates, q = self.local_coordinates(
                probabilities,
                log_energy,
                reliability,
            )
            batch = coordinates.shape[0]
            origin = coordinates.new_zeros(
                batch,
                self.num_groups,
                1,
                1,
                self.coordinate_dim,
            )
            local = self.psi(coordinates - origin) - self.psi(-origin)
            weights = q[:, :, :, :, None]
            weighted = (weights * local).sum(dim=(2, 3))
            support = weights.sum(dim=(2, 3))
            descriptor_grouped = weighted / (
                support + self._numerical_eps.to(weighted)
            )
            descriptor_grouped = torch.where(
                support > 0.0,
                descriptor_grouped,
                torch.zeros_like(descriptor_grouped),
            )
            descriptor = descriptor_grouped.flatten(1)
            projected = self.local_projection(descriptor)
            normalized = self.residual_norm(projected)
            residual_scale = self._residual_max.to(projected) * torch.tanh(
                self.raw_residual_scale.float()
            )
            residual = residual_scale * normalized
            zero_radial = descriptor.new_zeros(batch, self.num_groups)
            zero_composition = descriptor.new_zeros(
                batch,
                self.num_groups,
                self.orientations - 1,
            )
            aux = {
                "descriptor": descriptor,
                "residual": residual,
                "radial_origin": zero_radial,
                "composition_origin": zero_composition,
                "mean_reliability_by_group": q.mean(dim=(-2, -1)),
                "residual_scale": residual_scale.expand(batch, 1),
            }
        return residual, aux


class _AppearanceStem(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, int(out_channels), kernel_size=3, stride=2, bias=True)
        self.norm = LayerNorm2d(int(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.conv(_reflect_pad(x, 1))))


class _AsymmetricEvidenceInjection(nn.Module):
    """Inject radial evidence and C-conditioned evidence into A only."""

    def __init__(
        self,
        composition_channels: int,
        radial_channels: int,
        appearance_channels: int,
        *,
        layer_scale: float = 1.0e-6,
    ) -> None:
        super().__init__()
        composition_channels = int(composition_channels)
        radial_channels = int(radial_channels)
        appearance_channels = int(appearance_channels)
        self.radial_projection = nn.Conv2d(
            2 * radial_channels,
            appearance_channels,
            kernel_size=1,
            bias=True,
        )
        self.composition_gate = nn.Conv2d(
            composition_channels,
            appearance_channels,
            kernel_size=1,
            bias=True,
        )
        self.composition_value = nn.Conv2d(
            composition_channels,
            appearance_channels,
            kernel_size=1,
            bias=True,
        )
        self.layer_scale = nn.Parameter(
            torch.full((appearance_channels, 1, 1), float(layer_scale))
        )

    def forward(
        self,
        appearance: torch.Tensor,
        composition: torch.Tensor,
        log_energy: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        radial = self.radial_projection(torch.cat([log_energy, reliability], dim=1))
        gate = torch.sigmoid(self.composition_gate(composition))
        conditioned = gate * self.composition_value(composition)
        return appearance + self.layer_scale * (radial + conditioned)

    def reverse_conditioned_write(self, appearance: torch.Tensor) -> torch.Tensor:
        """Parameter-shared A->C write used only by the symmetric control.

        C and A have the same width at every ARCQ stage.  Reusing the exact
        gate/value maps gives the reverse edge the same conditional operator
        as the forward C-derived term without adding any trainable capacity.
        """

        gate = torch.sigmoid(self.composition_gate(appearance))
        conditioned = gate * self.composition_value(appearance)
        return self.layer_scale * conditioned


class ARCQRoadBackbone(nn.Module):
    """Asymmetric Radial-Compositional Quotient road-surface backbone.

    ``H`` is an isolated positive-homogeneous texture carrier. At each scale a
    constrained matched phase-pair bank produces directional responses. Their
    total magnitude is retained as radial evidence while their normalized
    orientation composition forms the protected ``C`` state. The raw
    appearance/amplitude ``A`` state can read C. The default model exposes no
    A->C path; an explicitly parameter-matched symmetric ablation can enable a
    parameter-free reverse edge.
    """

    def __init__(
        self,
        out_dim: int = 768,
        *,
        h_channels: tuple[int, ...] = (32, 48, 80, 128),
        evidence_channels: tuple[int, ...] = (48, 96, 192, 256),
        depths: tuple[int, ...] = (2, 2, 6, 2),
        response_kernel_sizes: tuple[int, ...] = (7, 5, 3, 3),
        group_width: int = 8,
        orientations: int = 4,
        simplex_smoothing: float = 0.05,
        composition_coordinate_mode: str = _ILR_ARCQ_CENTERED_SIMPLEX,
        observability_floor: float = 1.0e-8,
        fully_observable_energy: float = 1.0e-4,
        drop_path_rate: float = 0.15,
        evidence_layer_scale: float = 1.0e-6,
        input_mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        input_std: tuple[float, ...] = (0.229, 0.224, 0.225),
        use_composition_evidence: bool = True,
        use_radial_evidence: bool = True,
        use_raw_appearance: bool = True,
        allow_appearance_to_composition: bool = False,
        readout_interaction_mode: str = "product",
        readout_interaction_init: str = "standard",
        use_final_radiometric_sidecar: bool = False,
        radiometric_sidecar_scale: float = 1.0e-3,
        ri_caqc_mode: str = "disabled",
        q3_spatial_organizer_mode: str = "disabled",
        qodc_lift_mode: str = "disabled",
        ecqd_downsample_mode: str = _ECQD_DISABLED,
        pcqt_transition_mode: str = PCQT_DISABLED,
        final_appearance_transition_mode: str = _FINAL_A_TRANSITION_LEARNED_SIGNED,
        final_appearance_transition_coverage: float = 0.5,
        terminal_octave_mode: str = TOC_DISABLED,
        terminal_octave_depth: int = 2,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        if self.out_dim <= 0:
            raise ValueError("out_dim must be positive")
        self.h_channels = _as_int_tuple(h_channels, name="h_channels")
        self.evidence_channels = _as_int_tuple(
            evidence_channels,
            name="evidence_channels",
        )
        self.depths = _as_int_tuple(depths, name="depths")
        self.response_kernel_sizes = _as_int_tuple(
            response_kernel_sizes,
            name="response_kernel_sizes",
        )
        self.group_width = int(group_width)
        self.orientations = int(orientations)
        self.observability_floor = float(observability_floor)
        self.fully_observable_energy = float(fully_observable_energy)
        if self.group_width != 2 * self.orientations:
            raise ValueError(
                "group_width must equal 2*orientations so every group has an even/odd pair "
                f"per orientation, got {group_width=} and {orientations=}"
            )
        if any(channels % self.group_width != 0 for channels in self.h_channels):
            raise ValueError(
                f"every h_channels entry must be divisible by group_width={self.group_width}"
            )
        if any(kernel % 2 == 0 or kernel <= 1 for kernel in self.response_kernel_sizes):
            raise ValueError("all response_kernel_sizes must be odd integers greater than one")
        if not 0.0 <= float(drop_path_rate) < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if not 0.0 <= float(evidence_layer_scale) <= 1.0:
            raise ValueError("evidence_layer_scale must be in [0, 1]")
        if len(input_mean) != 3 or len(input_std) != 3:
            raise ValueError("input_mean and input_std must each contain three RGB values")
        if any(float(value) <= 0.0 for value in input_std):
            raise ValueError("input_std entries must be positive")
        readout_interaction_mode = str(readout_interaction_mode).strip().lower()
        if readout_interaction_mode not in {"product", "additive", "none"}:
            raise ValueError(
                "readout_interaction_mode must be one of "
                "{'product', 'additive', 'none'}"
            )
        readout_interaction_init = str(readout_interaction_init).strip().lower()
        if readout_interaction_init not in {"standard", "zero"}:
            raise ValueError(
                "readout_interaction_init must be one of {'standard', 'zero'}"
            )
        composition_coordinate_mode = str(composition_coordinate_mode).strip().lower()
        if composition_coordinate_mode not in _ILR_ARCQ_VALID_MODES:
            raise ValueError(
                "composition_coordinate_mode must be one of "
                f"{sorted(_ILR_ARCQ_VALID_MODES)}, got "
                f"{composition_coordinate_mode!r}"
            )

        self.simplex_smoothing = float(simplex_smoothing)
        self.composition_coordinate_mode = composition_coordinate_mode
        self.drop_path_rate = float(drop_path_rate)
        self.evidence_layer_scale = float(evidence_layer_scale)
        self.use_composition_evidence = bool(use_composition_evidence)
        self.use_radial_evidence = bool(use_radial_evidence)
        self.use_raw_appearance = bool(use_raw_appearance)
        self.allow_appearance_to_composition = bool(
            allow_appearance_to_composition
        )
        self.readout_interaction_mode = readout_interaction_mode
        self.readout_interaction_init = readout_interaction_init
        self.use_final_radiometric_sidecar = bool(use_final_radiometric_sidecar)
        self.radiometric_sidecar_scale = float(radiometric_sidecar_scale)
        if (
            not math.isfinite(self.radiometric_sidecar_scale)
            or not 0.0 <= self.radiometric_sidecar_scale <= 1.0
        ):
            raise ValueError("radiometric_sidecar_scale must be finite and in [0, 1]")
        ri_caqc_mode = str(ri_caqc_mode).strip().lower()
        if ri_caqc_mode not in {"disabled", "local_q3"}:
            raise ValueError(
                "ri_caqc_mode must be one of {'disabled', 'local_q3'}, "
                f"got {ri_caqc_mode!r}"
            )
        self.ri_caqc_mode = ri_caqc_mode
        q3_spatial_organizer_mode = str(q3_spatial_organizer_mode).strip().lower()
        valid_q3_spatial_modes = {
            "disabled",
            "additive_control",
            "product_difference",
        }
        if q3_spatial_organizer_mode not in valid_q3_spatial_modes:
            raise ValueError(
                "q3_spatial_organizer_mode must be one of "
                f"{sorted(valid_q3_spatial_modes)}, got "
                f"{q3_spatial_organizer_mode!r}"
            )
        if q3_spatial_organizer_mode != "disabled":
            q3_groups = self.h_channels[2] // self.group_width
            if (
                q3_groups != 10
                or self.orientations != 4
                or self.evidence_channels[2] != 192
            ):
                raise ValueError(
                    "the frozen PQSO Q3 contract requires 10 ordered groups, "
                    "4 orientations, and a 192-channel C3 state"
                )
            if not self.use_composition_evidence:
                raise ValueError(
                    "Q3 spatial organization requires use_composition_evidence=True"
                )
            if self.allow_appearance_to_composition:
                raise ValueError(
                    "Q3 spatial organization requires a protected C-only path; "
                    "allow_appearance_to_composition must be False"
                )
        self.q3_spatial_organizer_mode = q3_spatial_organizer_mode
        qodc_lift_mode = str(qodc_lift_mode).strip().lower()
        valid_qodc_lift_modes = {
            "disabled",
            ADDITIVE_CONTROL,
            PRODUCT_CONSENSUS,
        }
        if qodc_lift_mode not in valid_qodc_lift_modes:
            raise ValueError(
                "qodc_lift_mode must be one of "
                f"{sorted(valid_qodc_lift_modes)}, got {qodc_lift_mode!r}"
            )
        if qodc_lift_mode != "disabled":
            if q3_spatial_organizer_mode != "disabled":
                raise ValueError(
                    "the first QODC causal gate requires "
                    "q3_spatial_organizer_mode='disabled'"
                )
            if self.ri_caqc_mode != "disabled":
                raise ValueError(
                    "the first QODC causal gate requires ri_caqc_mode='disabled'"
                )
            if self.use_final_radiometric_sidecar:
                raise ValueError(
                    "the first QODC causal gate requires the radiometric sidecar "
                    "to be disabled"
                )
            if self.allow_appearance_to_composition:
                raise ValueError(
                    "QODC requires the protected asymmetric ARCQ path; "
                    "allow_appearance_to_composition must be False"
                )
            if not (
                self.use_composition_evidence
                and self.use_radial_evidence
                and self.use_raw_appearance
            ):
                raise ValueError(
                    "the first QODC causal gate requires the complete ARCQ C/A "
                    "evidence paths"
                )
            if self.readout_interaction_mode != "product":
                raise ValueError(
                    "the first QODC causal gate requires the original product readout"
                )
            if self.readout_interaction_init != "standard":
                raise ValueError(
                    "the first QODC causal gate requires standard readout initialization"
                )
        self.qodc_lift_mode = qodc_lift_mode
        ecqd_downsample_mode = str(ecqd_downsample_mode).strip().lower()
        valid_ecqd_modes = {
            _ECQD_DISABLED,
            _ECQD_RESPONSE_AVERAGE_CONTROL,
            _ECQD_ENERGY_COMPLETE,
        }
        if ecqd_downsample_mode not in valid_ecqd_modes:
            raise ValueError(
                "ecqd_downsample_mode must be one of "
                f"{sorted(valid_ecqd_modes)}, got {ecqd_downsample_mode!r}"
            )
        if ecqd_downsample_mode != _ECQD_DISABLED:
            # This first causal screen intentionally changes the terminal
            # response/energy ordering only. Existing post-hoc organizers,
            # sidecars and reverse edges would make E∘P versus P∘E
            # uninterpretable before the unchanged quotient is formed.
            if q3_spatial_organizer_mode != "disabled":
                raise ValueError(
                    "ECQD requires q3_spatial_organizer_mode='disabled'"
                )
            if self.ri_caqc_mode != "disabled":
                raise ValueError("ECQD requires ri_caqc_mode='disabled'")
            if qodc_lift_mode != "disabled":
                raise ValueError("ECQD requires qodc_lift_mode='disabled'")
            if self.use_final_radiometric_sidecar:
                raise ValueError("ECQD requires the radiometric sidecar to be disabled")
            if self.allow_appearance_to_composition:
                raise ValueError(
                    "ECQD requires the protected asymmetric ARCQ path; "
                    "allow_appearance_to_composition must be False"
                )
            if not (
                self.use_composition_evidence
                and self.use_radial_evidence
                and self.use_raw_appearance
            ):
                raise ValueError(
                    "ECQD requires the complete ARCQ C/A evidence paths"
                )
            if self.readout_interaction_mode != "product":
                raise ValueError("ECQD requires the original product readout")
            if self.readout_interaction_init != "standard":
                raise ValueError("ECQD requires standard readout initialization")
        self.ecqd_downsample_mode = ecqd_downsample_mode
        pcqt_transition_mode = str(pcqt_transition_mode).strip().lower()
        valid_pcqt_modes = {
            PCQT_DISABLED,
            PCQT_MARGINAL_ENERGY_CONTROL,
            PCQT_COHERENCE_QUOTIENT,
        }
        if pcqt_transition_mode not in valid_pcqt_modes:
            raise ValueError(
                "pcqt_transition_mode must be one of "
                f"{sorted(valid_pcqt_modes)}, got {pcqt_transition_mode!r}"
            )
        if pcqt_transition_mode != PCQT_DISABLED:
            # PCQT is a fresh S3->S4 transition experiment.  It must not be
            # stacked on ECQD's already-rejected energy-order intervention or
            # any other late writer/sidecar; otherwise its A/B/C conclusion
            # would no longer identify the paired-response coordinate.
            if ecqd_downsample_mode != _ECQD_DISABLED:
                raise ValueError("PCQT requires ecqd_downsample_mode='disabled'")
            if q3_spatial_organizer_mode != "disabled":
                raise ValueError(
                    "PCQT requires q3_spatial_organizer_mode='disabled'"
                )
            if self.ri_caqc_mode != "disabled":
                raise ValueError("PCQT requires ri_caqc_mode='disabled'")
            if qodc_lift_mode != "disabled":
                raise ValueError("PCQT requires qodc_lift_mode='disabled'")
            if self.use_final_radiometric_sidecar:
                raise ValueError("PCQT requires the radiometric sidecar to be disabled")
            if self.allow_appearance_to_composition:
                raise ValueError(
                    "PCQT requires the protected asymmetric ARCQ path; "
                    "allow_appearance_to_composition must be False"
                )
            if not (
                self.use_composition_evidence
                and self.use_radial_evidence
                and self.use_raw_appearance
            ):
                raise ValueError("PCQT requires the complete ARCQ C/A evidence paths")
            if self.readout_interaction_mode != "product":
                raise ValueError("PCQT requires the original product readout")
            if self.readout_interaction_init != "standard":
                raise ValueError("PCQT requires standard readout initialization")
        self.pcqt_transition_mode = pcqt_transition_mode
        final_appearance_transition_mode = (
            str(final_appearance_transition_mode).strip().lower()
        )
        if final_appearance_transition_mode not in _FINAL_A_TRANSITION_VALID_MODES:
            raise ValueError(
                "final_appearance_transition_mode must be one of "
                f"{sorted(_FINAL_A_TRANSITION_VALID_MODES)}, got "
                f"{final_appearance_transition_mode!r}"
            )
        final_appearance_transition_coverage = float(
            final_appearance_transition_coverage
        )
        if (
            not math.isfinite(final_appearance_transition_coverage)
            or not 0.0 < final_appearance_transition_coverage < 1.0
        ):
            raise ValueError(
                "final_appearance_transition_coverage must be finite and lie "
                "strictly in (0, 1)"
            )
        if final_appearance_transition_mode != _FINAL_A_TRANSITION_LEARNED_SIGNED:
            if self.composition_coordinate_mode != _ILR_ARCQ_CENTERED_SIMPLEX:
                raise ValueError(
                    "active LBET modes require composition_coordinate_mode="
                    "'centered_simplex'"
                )
            if self.allow_appearance_to_composition:
                raise ValueError(
                    "active LBET modes require the protected asymmetric ARCQ path"
                )
            if self.use_final_radiometric_sidecar:
                raise ValueError("active LBET modes require the radiometric sidecar off")
            if self.ri_caqc_mode != "disabled":
                raise ValueError("active LBET modes require ri_caqc_mode='disabled'")
            if self.q3_spatial_organizer_mode != "disabled":
                raise ValueError(
                    "active LBET modes require q3_spatial_organizer_mode='disabled'"
                )
            if self.qodc_lift_mode != "disabled":
                raise ValueError("active LBET modes require qodc_lift_mode='disabled'")
            if self.ecqd_downsample_mode != _ECQD_DISABLED:
                raise ValueError(
                    "active LBET modes require ecqd_downsample_mode='disabled'"
                )
            if self.pcqt_transition_mode != PCQT_DISABLED:
                raise ValueError(
                    "active LBET modes require pcqt_transition_mode='disabled'"
                )
            if not (
                self.use_composition_evidence
                and self.use_radial_evidence
                and self.use_raw_appearance
            ):
                raise ValueError(
                    "active LBET modes require the complete ARCQ C/A evidence paths"
                )
            if self.readout_interaction_mode != "product":
                raise ValueError("active LBET modes require the original product readout")
            if self.readout_interaction_init != "standard":
                raise ValueError(
                    "active LBET modes require standard readout initialization"
                )
        self.final_appearance_transition_mode = final_appearance_transition_mode
        self.final_appearance_transition_coverage = (
            final_appearance_transition_coverage
        )
        terminal_octave_mode = str(terminal_octave_mode).strip().lower()
        if terminal_octave_mode not in TOC_VALID_MODES:
            raise ValueError(
                "terminal_octave_mode must be one of "
                f"{sorted(TOC_VALID_MODES)}, got {terminal_octave_mode!r}"
            )
        terminal_octave_depth = int(terminal_octave_depth)
        if terminal_octave_depth <= 0:
            raise ValueError("terminal_octave_depth must be a positive integer")
        if terminal_octave_mode != TOC_DISABLED:
            # TOC isolates one variable: the spatial octave of the terminal
            # consolidation.  Any other active study arm would make the result a
            # compound intervention rather than a stride contrast.
            if self.allow_appearance_to_composition:
                raise ValueError(
                    "active TOC modes require the protected asymmetric ARCQ "
                    "path; allow_appearance_to_composition must be False"
                )
            if self.use_final_radiometric_sidecar:
                raise ValueError(
                    "active TOC modes require the radiometric sidecar off"
                )
            if self.ri_caqc_mode != "disabled":
                raise ValueError("active TOC modes require ri_caqc_mode='disabled'")
            if self.q3_spatial_organizer_mode != "disabled":
                raise ValueError(
                    "active TOC modes require q3_spatial_organizer_mode='disabled'"
                )
            if self.qodc_lift_mode != "disabled":
                raise ValueError("active TOC modes require qodc_lift_mode='disabled'")
            if self.ecqd_downsample_mode != _ECQD_DISABLED:
                raise ValueError(
                    "active TOC modes require ecqd_downsample_mode='disabled'"
                )
            if self.pcqt_transition_mode != PCQT_DISABLED:
                raise ValueError(
                    "active TOC modes require pcqt_transition_mode='disabled'"
                )
            if (
                self.final_appearance_transition_mode
                != _FINAL_A_TRANSITION_LEARNED_SIGNED
            ):
                raise ValueError(
                    "active TOC modes require the historical learned-signed "
                    "final appearance transition; the falsified LBET arms must "
                    "not be stacked underneath a TOC experiment"
                )
            if self.composition_coordinate_mode != _ILR_ARCQ_CENTERED_SIMPLEX:
                raise ValueError(
                    "active TOC modes require the historical centered-simplex "
                    "composition coordinate"
                )
            if not (
                self.use_composition_evidence
                and self.use_radial_evidence
                and self.use_raw_appearance
            ):
                raise ValueError(
                    "active TOC modes require the complete ARCQ C/A evidence paths"
                )
            if self.readout_interaction_mode != "product":
                raise ValueError(
                    "active TOC modes require the original product readout"
                )
            if self.readout_interaction_init != "standard":
                raise ValueError(
                    "active TOC modes require standard readout initialization"
                )
        self.terminal_octave_mode = terminal_octave_mode
        self.terminal_octave_depth = terminal_octave_depth
        if self.composition_coordinate_mode != _ILR_ARCQ_CENTERED_SIMPLEX:
            # This is an isolated coordinate-geometry screen. A learned
            # writer, sidecar, altered terminal response ordering, or A->C
            # edge would make a result a compound intervention rather than an
            # A/B/C comparison of the quotient coordinate itself.
            if self.allow_appearance_to_composition:
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require "
                    "allow_appearance_to_composition=False"
                )
            if self.use_final_radiometric_sidecar:
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require the "
                    "radiometric sidecar to be disabled"
                )
            if self.ri_caqc_mode != "disabled":
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require "
                    "ri_caqc_mode='disabled'"
                )
            if self.q3_spatial_organizer_mode != "disabled":
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require "
                    "q3_spatial_organizer_mode='disabled'"
                )
            if self.qodc_lift_mode != "disabled":
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require "
                    "qodc_lift_mode='disabled'"
                )
            if self.ecqd_downsample_mode != _ECQD_DISABLED:
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require "
                    "ecqd_downsample_mode='disabled'"
                )
            if self.pcqt_transition_mode != PCQT_DISABLED:
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require "
                    "pcqt_transition_mode='disabled'"
                )
            if not (
                self.use_composition_evidence
                and self.use_radial_evidence
                and self.use_raw_appearance
            ):
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require the complete "
                    "ARCQ C/A evidence paths"
                )
            if self.readout_interaction_mode != "product":
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require the original "
                    "product readout"
                )
            if self.readout_interaction_init != "standard":
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require standard "
                    "readout initialization"
                )
        self.register_buffer(
            _ARCQ_STATE_VERSION_KEY,
            torch.tensor(_ARCQ_STATE_VERSION, dtype=torch.int64),
        )
        self.register_buffer(
            "input_mean",
            torch.tensor(tuple(float(value) for value in input_mean)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor(tuple(float(value) for value in input_std)).view(1, 3, 1, 1),
        )
        # Only active ECQD arms own these persistent buffers.  This preserves
        # the default ARCQ checkpoint schema exactly, yet makes B/C checkpoint
        # semantics fail closed if a caller accidentally swaps their ordering.
        if self.ecqd_downsample_mode != _ECQD_DISABLED:
            self.register_buffer(
                "_ecqd_schema_version",
                torch.tensor(_ECQD_SCHEMA_VERSION, dtype=torch.int64),
            )
            self.register_buffer(
                "_ecqd_mode_code",
                torch.tensor(
                    _ECQD_MODE_CODES[self.ecqd_downsample_mode],
                    dtype=torch.int64,
                ),
            )
            # [kernel size, stride, reflected pad] is serialized rather than
            # left as an undocumented implementation detail.
            self.register_buffer(
                "_ecqd_pool_spec",
                torch.tensor((3, 2, 1), dtype=torch.int64),
            )
        if (
            self.final_appearance_transition_mode
            != _FINAL_A_TRANSITION_LEARNED_SIGNED
        ):
            self.register_buffer(
                "_lbet_schema_version",
                torch.tensor(_LBET_SCHEMA_VERSION, dtype=torch.int64),
            )
            self.register_buffer(
                "_lbet_mode_code",
                torch.tensor(
                    _FINAL_A_TRANSITION_ACTIVE_MODE_CODES[
                        self.final_appearance_transition_mode
                    ],
                    dtype=torch.int64,
                ),
            )
            self.register_buffer(
                "_lbet_coverage",
                torch.tensor(
                    self.final_appearance_transition_coverage,
                    dtype=torch.float32,
                ),
            )
        # Only active TOC arms own these buffers, so the default ARCQ
        # checkpoint schema is untouched.  Serializing the stride alongside the
        # mode makes a same-scale checkpoint impossible to evaluate silently as
        # the terminal-octave arm, even though their weights are shape-identical.
        if self.terminal_octave_mode != TOC_DISABLED:
            self.register_buffer(
                "_toc_schema_version",
                torch.tensor(TOC_SCHEMA_VERSION, dtype=torch.int64),
            )
            self.register_buffer(
                "_toc_mode_code",
                torch.tensor(
                    TOC_MODE_CODES[self.terminal_octave_mode],
                    dtype=torch.int64,
                ),
            )
            self.register_buffer(
                "_toc_stride",
                torch.tensor(
                    TOC_MODE_STRIDES[self.terminal_octave_mode],
                    dtype=torch.int64,
                ),
            )
            self.register_buffer(
                "_toc_depth",
                torch.tensor(self.terminal_octave_depth, dtype=torch.int64),
            )

        response_banks = []
        quotients = []
        h_stages = []
        c_projections = []
        c_transitions = []
        c_stages = []
        a_transitions = []
        a_injections = []
        a_stages = []

        # The protected C path stays deterministic during training so the
        # affine-consistency objective cannot be contaminated by independent
        # stochastic-depth masks. DropPath is reserved for the sensitive A
        # path.
        total_blocks = sum(self.depths)
        drop_rates = torch.linspace(0.0, self.drop_path_rate, total_blocks).tolist()
        drop_index = 0
        previous_h = 3
        for stage_index in range(4):
            h_dim = self.h_channels[stage_index]
            evidence_dim = self.evidence_channels[stage_index]
            response_banks.append(
                MatchedResponseConv2d(
                    previous_h,
                    h_dim,
                    self.response_kernel_sizes[stage_index],
                    stride=2,
                    orientations=self.orientations,
                )
            )
            quotient = RadialCompositionQuotient(
                h_dim,
                group_width=self.group_width,
                orientations=self.orientations,
                smoothing=self.simplex_smoothing,
                observability_floor=self.observability_floor,
                fully_observable_energy=self.fully_observable_energy,
                composition_coordinate_mode=self.composition_coordinate_mode,
            )
            quotients.append(quotient)
            # The last response is consumed directly by the last quotient.
            # Refining it afterward creates parameters with no path to any
            # output, so state v2+ makes that terminal operation explicit.
            if stage_index == 3:
                h_stages.append(nn.Identity())
            else:
                h_stages.append(
                    nn.Sequential(
                        *[
                            HomogeneousCarrierBlock(h_dim, residual_scale=1.0e-3)
                            for _ in range(self.depths[stage_index])
                        ]
                    )
                )
            c_projections.append(
                nn.Conv2d(
                    quotient.composition_channels,
                    evidence_dim,
                    kernel_size=1,
                    bias=True,
                )
            )
            if stage_index > 0:
                c_transitions.append(
                    _SpatialTransition(
                        self.evidence_channels[stage_index - 1],
                        evidence_dim,
                    )
                )
                if (
                    stage_index == 3
                    and self.final_appearance_transition_mode
                    != _FINAL_A_TRANSITION_LEARNED_SIGNED
                ):
                    a_transitions.append(
                        _BarycentricSpatialTransition(
                            self.evidence_channels[stage_index - 1],
                            evidence_dim,
                            mode=self.final_appearance_transition_mode,
                            coverage=self.final_appearance_transition_coverage,
                        )
                    )
                else:
                    a_transitions.append(
                        _SpatialTransition(
                            self.evidence_channels[stage_index - 1],
                            evidence_dim,
                        )
                    )
            c_blocks = []
            for _ in range(self.depths[stage_index]):
                c_blocks.append(
                    CompositionBlock(
                        evidence_dim,
                        expansion=2,
                        layer_scale=self.evidence_layer_scale,
                        drop_path=0.0,
                    )
                )
            c_stages.append(nn.Sequential(*c_blocks))
            a_injections.append(
                _AsymmetricEvidenceInjection(
                    evidence_dim,
                    quotient.num_groups,
                    evidence_dim,
                    layer_scale=self.evidence_layer_scale,
                )
            )
            a_blocks = []
            for _ in range(self.depths[stage_index]):
                a_blocks.append(
                    AppearanceAmplitudeBlock(
                        evidence_dim,
                        expansion=2,
                        layer_scale=self.evidence_layer_scale,
                        drop_path=float(drop_rates[drop_index]),
                    )
                )
                drop_index += 1
            a_stages.append(nn.Sequential(*a_blocks))
            previous_h = h_dim

        self.response_banks = nn.ModuleList(response_banks)
        self.quotients = nn.ModuleList(quotients)
        self.h_stages = nn.ModuleList(h_stages)
        self.c_projections = nn.ModuleList(c_projections)
        self.c_transitions = nn.ModuleList(c_transitions)
        self.c_stages = nn.ModuleList(c_stages)
        self.appearance_stem = _AppearanceStem(self.evidence_channels[0])
        self.a_transitions = nn.ModuleList(a_transitions)
        self.final_radiometric_sidecar: _RadiometricStatisticsSidecar | None
        if self.use_final_radiometric_sidecar:
            self.final_radiometric_sidecar = _RadiometricStatisticsSidecar(
                self.evidence_channels[-1],
                initial_scale=self.radiometric_sidecar_scale,
            )
        else:
            self.final_radiometric_sidecar = None
        self.a_injections = nn.ModuleList(a_injections)
        self.a_stages = nn.ModuleList(a_stages)

        final_dim = self.evidence_channels[-1]
        self.readout_interaction = nn.Conv2d(final_dim, final_dim, kernel_size=1, bias=True)
        fused_dim = 3 * final_dim
        self.fusion_projection: nn.Module
        if fused_dim == self.out_dim:
            self.fusion_projection = nn.Identity()
        else:
            self.fusion_projection = nn.Conv2d(fused_dim, self.out_dim, kernel_size=1, bias=True)
        self.output_norm = nn.LayerNorm(self.out_dim)

        self.last_feature_map: torch.Tensor | None = None
        self.last_aux: dict[str, torch.Tensor] = {}
        self.last_protected_stages: tuple[torch.Tensor, ...] = ()
        self.last_load_migration: dict[str, Any] | None = None
        self._initialize_evidence_paths()
        if (
            self.final_appearance_transition_mode
            != _FINAL_A_TRANSITION_LEARNED_SIGNED
        ):
            final_transition = self.a_transitions[-1]
            if not isinstance(final_transition, _BarycentricSpatialTransition):
                raise RuntimeError("active LBET mode did not build its final transition")
            # Global initialization deliberately consumes the same RNG as the
            # historical arm first.  Zeroing afterward gives B/C identical
            # uniform initial functions without perturbing any later RNG state.
            final_transition.initialize_matched_logits()
        if self.readout_interaction_init == "zero":
            # Identity-anchored product readout.  The public forward graph and
            # parameter budget stay unchanged, but the third 256-channel slot
            # starts as exactly zero: [C4, A4, 0].  We deliberately zero only
            # after the ordinary initializer has consumed the same RNG as the
            # standard arm, so all other backbone parameters and the caller's
            # subsequent classifier-head initialization remain bit-identical.
            # At this point tanh'(0)=1, hence the readout convolution still has
            # a non-zero first-order gradient for generic C4/A4 evidence.
            with torch.no_grad():
                self.readout_interaction.weight.zero_()
                if self.readout_interaction.bias is not None:
                    self.readout_interaction.bias.zero_()
        self.terminal_octave: TerminalOctavePair | None = None
        if self.terminal_octave_mode != TOC_DISABLED:
            # Build after the ordinary evidence initializer, inside a private
            # RNG scope seeded only by the global seed.  Therefore the ARCQ
            # baseline (A), the same-scale control (B) and the terminal-octave
            # candidate (C) share bit-identical base backbone weights, and the
            # caller's later classifier-head initialization sees the same global
            # RNG state in all three arms.  ``stride`` is never read by any
            # initializer, so B and C additionally own bit-identical TOC
            # tensors: the only difference between them is the octave at which
            # the consolidation is evaluated.
            toc_local_seed = (int(torch.initial_seed()) + 1_500_450_271) % (
                2**63 - 1
            )
            terminal_dim = self.evidence_channels[-1]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(toc_local_seed)
                self.terminal_octave = TerminalOctavePair(
                    terminal_dim,
                    mode=self.terminal_octave_mode,
                    depth=self.terminal_octave_depth,
                    norm_factory=LayerNorm2d,
                    composition_block_factory=lambda channels: CompositionBlock(
                        channels,
                        expansion=2,
                        layer_scale=self.evidence_layer_scale,
                        drop_path=0.0,
                    ),
                    appearance_block_factory=(
                        lambda channels: AppearanceAmplitudeBlock(
                            channels,
                            expansion=2,
                            layer_scale=self.evidence_layer_scale,
                            drop_path=0.0,
                        )
                    ),
                )
        self.q3_spatial_organizer: (
            AdditiveProtectedQuotientControl
            | ProtectedQuotientSpatialOrganizer
            | None
        ) = None
        if self.q3_spatial_organizer_mode != "disabled":
            # Construct the paired arm after base initialization, inside an
            # RNG fork.  The disabled arm consumes no extra random values, B/C
            # receive the same explicit local seed, and the caller's later
            # classifier initialization sees the exact same RNG state.
            q3_local_seed = (int(torch.initial_seed()) + 982_451_653) % (2**63 - 1)
            organizer_class = (
                AdditiveProtectedQuotientControl
                if self.q3_spatial_organizer_mode == "additive_control"
                else ProtectedQuotientSpatialOrganizer
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(q3_local_seed)
                self.q3_spatial_organizer = organizer_class(
                    ordered_groups=10,
                    composition_parts=4,
                    hidden_channels=32,
                    out_channels=192,
                    residual_max=0.25,
                    validate_centered_input=False,
                )
        self.ri_caqc: RoleIdentifiedCAQCReadout | None = None
        if self.ri_caqc_mode != "disabled":
            # Module constructors consume random numbers even when followed by
            # explicit initialization. Forking makes the public base weights,
            # classifier-head initialization and training RNG bit-identical to
            # the disabled arm under the same seed.
            initial_seed = int(torch.initial_seed())
            with torch.random.fork_rng(devices=[]):
                self.ri_caqc = RoleIdentifiedCAQCReadout(
                    mode=self.ri_caqc_mode,
                    response_channels=self.h_channels[2],
                    group_width=self.group_width,
                    orientations=self.orientations,
                    context_channels=self.evidence_channels[-1],
                    out_dim=self.out_dim,
                )
                self.ri_caqc.initialize_local_parameters(seed=initial_seed)
        self.qodc_lift: QuotientOrthogonalDetailConsensusLift | None = None
        if self.qodc_lift_mode != "disabled":
            # The base ARCQ and the later classifier head must be identical in
            # A/B/C.  QODC therefore owns a deterministic local RNG scope and
            # is constructed only after the ordinary evidence initializer.
            qodc_local_seed = (int(torch.initial_seed()) + 1_736_912_771) % (
                2**63 - 1
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(qodc_local_seed)
                self.qodc_lift = QuotientOrthogonalDetailConsensusLift(
                    mode=self.qodc_lift_mode,
                    evidence_channels=self.evidence_channels[-1],
                    out_channels=self.out_dim,
                    hidden_channels=32,
                    residual_max=0.10,
                )
        self.pcqt_transition: PairedResponseCoherenceTransition | None = None
        if self.pcqt_transition_mode != PCQT_DISABLED:
            terminal_quotient = self.quotients[-1]
            if not isinstance(terminal_quotient, RadialCompositionQuotient):
                raise RuntimeError("PCQT requires the terminal RadialCompositionQuotient")
            # Build after the base initializer in a private RNG scope.  Thus
            # ARCQ A, PCQT-B and PCQT-C have bit-identical base weights and
            # their later classifier heads see the same global RNG state.
            pcqt_local_seed = (int(torch.initial_seed()) + 2_147_483_629) % (
                2**63 - 1
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(pcqt_local_seed)
                transition = PairedResponseCoherenceTransition(
                    mode=self.pcqt_transition_mode,
                    source_h_channels=self.h_channels[-2],
                    response_channels=self.h_channels[-1],
                    groups=terminal_quotient.num_groups,
                    orientations=self.orientations,
                    phases=2,
                    coordinate_channels=terminal_quotient.composition_channels,
                    target_channels=self.evidence_channels[-1],
                    source_stage=2,
                    target_stage=3,
                    dilation_pair=(1, 2),
                    pool_spec=(3, 2, 1),
                    support_floor=1.0e-8,
                    normalization_delta=1.0e-4,
                    residual_max=0.10,
                )
                transition.initialize_local_parameters()
            self.pcqt_transition = transition

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
        self.last_load_migration = _prepare_arcq_state_dict_load(
            self,
            state_dict,
            prefix,
            error_msgs,
        )
        # TOC's two active arms are shape-identical and key-identical; they
        # differ only in the serialized stride.  Without this check a
        # same-scale-control checkpoint would load into the terminal-octave arm
        # under strict=True and be silently evaluated as the candidate.
        toc_buffer_names = {
            "schema": (prefix + "_toc_schema_version", TOC_SCHEMA_VERSION),
            "mode": (
                prefix + "_toc_mode_code",
                TOC_MODE_CODES.get(self.terminal_octave_mode),
            ),
            "stride": (
                prefix + "_toc_stride",
                TOC_MODE_STRIDES.get(self.terminal_octave_mode),
            ),
            "depth": (prefix + "_toc_depth", self.terminal_octave_depth),
        }
        toc_prefix = prefix + "terminal_octave."
        incoming_toc = any(
            key.startswith(toc_prefix) for key in state_dict
        ) or any(name in state_dict for name, _ in toc_buffer_names.values())
        active_toc = self.terminal_octave_mode != TOC_DISABLED
        if not active_toc and incoming_toc:
            error_msgs.append(
                "active TOC checkpoint cannot be loaded into a disabled ARCQ "
                "backbone, even with strict=False"
            )
        elif active_toc and not incoming_toc:
            error_msgs.append(
                "disabled/legacy ARCQ checkpoint cannot be silently loaded into "
                "an active TOC arm"
            )
        elif active_toc:
            for semantic_name, (state_name, expected_value) in (
                toc_buffer_names.items()
            ):
                supplied = state_dict.get(state_name)
                if supplied is None:
                    error_msgs.append(
                        f"TOC checkpoint is missing its {semantic_name} buffer "
                        f'"{state_name}"'
                    )
                    continue
                valid = (
                    torch.is_tensor(supplied)
                    and supplied.shape == torch.Size([])
                    and supplied.dtype == torch.int64
                    and int(supplied.item()) == int(expected_value)
                )
                if valid:
                    continue
                error_msgs.append(
                    f"TOC checkpoint {semantic_name} does not match the "
                    f"instantiated arm terminal_octave_mode="
                    f"{self.terminal_octave_mode!r} "
                    f"(depth={self.terminal_octave_depth}); expected scalar "
                    f"int64 value {expected_value}"
                )
                # A failed load must not mutate this module's own arm identity
                # before it raises.
                state_dict[state_name] = torch.tensor(
                    int(expected_value),
                    dtype=torch.int64,
                )
        qodc_prefix = prefix + "qodc_lift."
        incoming_qodc = any(key.startswith(qodc_prefix) for key in state_dict)
        if self.qodc_lift is None and incoming_qodc:
            error_msgs.append(
                "active QODC checkpoint cannot be loaded into a disabled ARCQ "
                "backbone, even with strict=False"
            )
        if self.qodc_lift is not None and not incoming_qodc:
            error_msgs.append(
                "disabled/legacy ARCQ checkpoint cannot be silently loaded into "
                "an active QODC arm"
            )
        ecqd_schema_name = prefix + "_ecqd_schema_version"
        ecqd_mode_name = prefix + "_ecqd_mode_code"
        ecqd_pool_name = prefix + "_ecqd_pool_spec"
        ecqd_names = (ecqd_schema_name, ecqd_mode_name, ecqd_pool_name)
        incoming_ecqd = any(name in state_dict for name in ecqd_names)
        active_ecqd = self.ecqd_downsample_mode != _ECQD_DISABLED
        if not active_ecqd and incoming_ecqd:
            error_msgs.append(
                "active ECQD checkpoint cannot be loaded into a disabled ARCQ "
                "backbone, even with strict=False"
            )
        if active_ecqd and not incoming_ecqd:
            error_msgs.append(
                "disabled/legacy ARCQ checkpoint cannot be silently loaded into "
                "an active ECQD arm"
            )
        if active_ecqd and incoming_ecqd:
            expected_schema = getattr(self, "_ecqd_schema_version")
            expected_mode = getattr(self, "_ecqd_mode_code")
            expected_pool = getattr(self, "_ecqd_pool_spec")
            incoming_schema = state_dict.get(ecqd_schema_name)
            incoming_mode = state_dict.get(ecqd_mode_name)
            incoming_pool = state_dict.get(ecqd_pool_name)
            if any(value is None for value in (incoming_schema, incoming_mode, incoming_pool)):
                error_msgs.append(
                    "ECQD checkpoint must contain schema, mode and pool-spec buffers"
                )
            else:
                def _valid_exact_ecqd_buffer(
                    supplied: object,
                    expected: torch.Tensor,
                    *,
                    name: str,
                ) -> bool:
                    if not isinstance(supplied, torch.Tensor):
                        error_msgs.append(f"ECQD checkpoint {name} buffer is not a tensor")
                        return False
                    if supplied.dtype != expected.dtype or supplied.shape != expected.shape:
                        error_msgs.append(
                            f"ECQD checkpoint {name} buffer dtype/shape does not match "
                            "the instantiated backbone"
                        )
                        return False
                    return torch.equal(supplied.detach().cpu(), expected.detach().cpu())

                if not _valid_exact_ecqd_buffer(
                    incoming_schema,
                    expected_schema,
                    name="schema",
                ):
                    error_msgs.append("ECQD checkpoint schema does not match the instantiated backbone")
                if not _valid_exact_ecqd_buffer(
                    incoming_mode,
                    expected_mode,
                    name="mode",
                ):
                    error_msgs.append(
                        "ECQD checkpoint mode does not match the instantiated "
                        f"arm {self.ecqd_downsample_mode!r}"
                    )
                if not _valid_exact_ecqd_buffer(
                    incoming_pool,
                    expected_pool,
                    name="pool specification",
                ):
                    error_msgs.append(
                        "ECQD checkpoint pool specification does not match the "
                        "instantiated backbone"
                    )
        lbet_schema_name = prefix + "_lbet_schema_version"
        lbet_mode_name = prefix + "_lbet_mode_code"
        lbet_coverage_name = prefix + "_lbet_coverage"
        lbet_names = (lbet_schema_name, lbet_mode_name, lbet_coverage_name)
        incoming_lbet = any(name in state_dict for name in lbet_names)
        active_lbet = (
            self.final_appearance_transition_mode
            != _FINAL_A_TRANSITION_LEARNED_SIGNED
        )
        if not active_lbet and incoming_lbet:
            error_msgs.append(
                "active LBET checkpoint cannot be loaded into the historical "
                "signed-transition ARCQ backbone, even with strict=False"
            )
        if active_lbet and not incoming_lbet:
            error_msgs.append(
                "historical/disabled ARCQ checkpoint cannot be silently loaded "
                "into an active LBET arm"
            )
        if active_lbet and incoming_lbet:
            expected_lbet_buffers = {
                "schema": getattr(self, "_lbet_schema_version"),
                "mode": getattr(self, "_lbet_mode_code"),
                "coverage": getattr(self, "_lbet_coverage"),
            }
            supplied_lbet_buffers = {
                "schema": state_dict.get(lbet_schema_name),
                "mode": state_dict.get(lbet_mode_name),
                "coverage": state_dict.get(lbet_coverage_name),
            }
            for name, expected in expected_lbet_buffers.items():
                supplied = supplied_lbet_buffers[name]
                if not isinstance(supplied, torch.Tensor):
                    error_msgs.append(
                        f"LBET checkpoint {name} buffer is missing or is not a tensor"
                    )
                    continue
                if (
                    supplied.dtype != expected.dtype
                    or supplied.shape != expected.shape
                    or not torch.equal(
                        supplied.detach().cpu(),
                        expected.detach().cpu(),
                    )
                ):
                    error_msgs.append(
                        f"LBET checkpoint {name} does not match instantiated arm "
                        f"{self.final_appearance_transition_mode!r}"
                    )
        pcqt_prefix = prefix + "pcqt_transition."
        incoming_pcqt = any(key.startswith(pcqt_prefix) for key in state_dict)
        active_pcqt = self.pcqt_transition_mode != PCQT_DISABLED
        if not active_pcqt and incoming_pcqt:
            error_msgs.append(
                "active PCQT checkpoint cannot be loaded into a disabled ARCQ "
                "backbone, even with strict=False"
            )
        if active_pcqt and not incoming_pcqt:
            error_msgs.append(
                "disabled/legacy ARCQ checkpoint cannot be silently loaded into "
                "an active PCQT arm"
            )
        if active_pcqt and self.pcqt_transition is None:
            error_msgs.append("active PCQT mode has no instantiated PCQT transition")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _initialize_evidence_paths(self) -> None:
        # Preserve an information-carrying initialization while keeping every
        # residual update close to identity at the start of from-scratch runs.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _raw_appearance_input(self, x: torch.Tensor) -> torch.Tensor:
        # RSCD pipelines normally provide ImageNet-normalized tensors. H reads
        # x directly to preserve its exact homogeneity test; A receives the
        # reconstructed radiometric channels.
        return x * self.input_std.to(device=x.device, dtype=x.dtype) + self.input_mean.to(
            device=x.device,
            dtype=x.dtype,
        )

    @staticmethod
    def _detached_feature_rms(feature: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return feature.detach().square().mean(
                dim=(1, 2, 3),
                dtype=torch.float32,
            ).sqrt()

    def _apply_q3_spatial_organizer(
        self,
        *,
        stage_index: int,
        composition_state: torch.Tensor,
        decomposition: dict[str, torch.Tensor],
        collect_summaries: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        organizer = self.q3_spatial_organizer
        if organizer is None or int(stage_index) != 2:
            return composition_state, None

        centered_flat = decomposition["composition"]
        expected_channels = 10 * 4
        if centered_flat.ndim != 4 or int(centered_flat.shape[1]) != expected_channels:
            raise RuntimeError(
                "PQSO expected Q3 centered composition with shape "
                "Bx40xH3xW3, got "
                f"{tuple(centered_flat.shape)}"
            )
        batch, _, height, width = centered_flat.shape
        centered = centered_flat.reshape(batch, 10, 4, height, width)
        if collect_summaries:
            residual, summaries = organizer.forward_with_summaries(centered)
        else:
            residual = organizer(centered)
            summaries = None
        residual = residual.to(
            device=composition_state.device,
            dtype=composition_state.dtype,
        )
        if residual.shape != composition_state.shape:
            raise RuntimeError(
                "PQSO residual must match the C3 pre-block state exactly, got "
                f"{tuple(residual.shape)} and {tuple(composition_state.shape)}"
            )

        pre_state = composition_state
        post_state = pre_state + residual
        if summaries is not None:
            pre_rms = self._detached_feature_rms(pre_state)
            post_rms = self._detached_feature_rms(post_state)
            residual_rms = summaries["residual_rms"]
            summaries = dict(summaries)
            summaries.update(
                {
                    "c3_pre_rms": pre_rms,
                    "c3_post_rms": post_rms,
                    "c3_post_pre_rms_ratio": post_rms
                    / pre_rms.clamp_min(1.0e-12),
                    "residual_to_pre_rms_ratio": residual_rms
                    / pre_rms.clamp_min(1.0e-12),
                }
            )
        return post_state, summaries

    @staticmethod
    def _ecqd_pool(value: torch.Tensor) -> torch.Tensor:
        """Reflection-safe, constant-preserving 3x3 average decimation.

        A reflected one-pixel support followed by a 3x3, stride-two average
        maps the odd native ARCQ grids ``45x30 -> 23x15`` and the D350 screen
        grids ``36x24 -> 18x12``.  Its weights are non-negative and sum to one
        at every output location.  Therefore, for each response coordinate
        ``r``, Jensen gives ``P(r)^2 <= P(r^2)`` exactly (up to floating-point
        rounding).  The B/C arms differ only in which side of that inequality
        is supplied to the unchanged quotient.
        """

        if value.ndim != 4:
            raise ValueError(
                f"ECQD pooling expects BxCxHxW, got {tuple(value.shape)}"
            )
        if min(value.shape[-2:]) <= 1:
            raise ValueError(
                "ECQD reflection pooling requires both spatial dimensions > 1, "
                f"got {tuple(value.shape[-2:])}"
            )
        return F.avg_pool2d(
            _reflect_pad(value, 1),
            kernel_size=3,
            stride=2,
        )

    @staticmethod
    def _pcqt_pool(value: torch.Tensor) -> torch.Tensor:
        """PCQT's fixed, reflection-safe S3-to-S4 support.

        This has the same numerical stencil as ECQD's decimator but a distinct
        semantic role: it forms the common support for two scale-tied response
        energies and their phase-vector inner product.  Keeping the helper
        separate prevents a future ECQD change from silently changing the
        PCQT checkpoint definition.
        """

        if value.ndim != 4:
            raise ValueError(
                f"PCQT pooling expects BxCxHxW, got {tuple(value.shape)}"
            )
        if min(value.shape[-2:]) <= 1:
            raise ValueError(
                "PCQT reflection pooling requires both spatial dimensions > 1, "
                f"got {tuple(value.shape[-2:])}"
            )
        return F.avg_pool2d(
            _reflect_pad(value, 1),
            kernel_size=3,
            stride=2,
        )

    @staticmethod
    def _pcqt_phase_inner_product(
        quotient: RadialCompositionQuotient,
        response_first: torch.Tensor,
        response_second: torch.Tensor,
    ) -> torch.Tensor:
        r"""Return \(r_1^T r_2\) in ARCQ's exact ``B,G,O,H,W`` ordering."""

        if response_first.shape != response_second.shape:
            raise ValueError("PCQT paired responses must have identical geometry")
        if response_first.ndim != 4 or response_first.shape[1] != quotient.channels:
            raise ValueError(
                "PCQT paired response geometry does not match the terminal quotient"
            )
        first = (
            response_first.float()
            if response_first.dtype in {torch.float16, torch.bfloat16}
            else response_first
        )
        second = (
            response_second.float()
            if response_second.dtype in {torch.float16, torch.bfloat16}
            else response_second
        )
        batch, _, height, width = first.shape
        first_grouped = first.reshape(
            batch,
            quotient.num_groups,
            quotient.orientations,
            2,
            height,
            width,
        )
        second_grouped = second.reshape(
            batch,
            quotient.num_groups,
            quotient.orientations,
            2,
            height,
            width,
        )
        return (first_grouped * second_grouped).sum(dim=3)

    def _pcqt_coordinate(
        self,
        h3: torch.Tensor,
        *,
        collect_summaries: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Build PCQT B/C coordinates from the pre-decimation S3 carrier.

        The ordinary terminal S4 ``bank(h3, stride=2) -> quotient`` route is
        intentionally not touched.  This helper makes two *additional* calls
        to that same constrained bank at dilation 1 and 2, pools their matched
        energy/phase-inner-product ledgers on one fixed support, and returns
        either the marginal control or the signed coherence quotient.
        """

        transition = self.pcqt_transition
        if transition is None or self.pcqt_transition_mode == PCQT_DISABLED:
            raise RuntimeError("PCQT coordinate requested while PCQT is disabled")
        bank = self.response_banks[-1]
        quotient = self.quotients[-1]
        if not isinstance(quotient, RadialCompositionQuotient):
            raise RuntimeError("PCQT requires the terminal RadialCompositionQuotient")
        if h3.ndim != 4 or h3.shape[1] != transition.source_h_channels:
            raise ValueError(
                f"PCQT expected S3 H carrier Bx{transition.source_h_channels}xHxW, "
                f"got {tuple(h3.shape)}"
            )

        # The two calls share one projected zero-DC/orthogonal bank.  B/C both
        # execute all ledgers below; only their selected coordinate differs.
        response_first = bank(h3, stride=1, dilation=1)
        response_second = bank(h3, stride=1, dilation=2)
        energy_first_full = quotient.directional_energy(response_first)
        energy_second_full = quotient.directional_energy(response_second)
        inner_full = self._pcqt_phase_inner_product(
            quotient,
            response_first,
            response_second,
        )
        batch, groups, orientations, height, width = energy_first_full.shape
        if groups != transition.groups or orientations != transition.orientations:
            raise RuntimeError("PCQT terminal quotient geometry drifted from its schema")
        pool_shape = self._pcqt_pool(
            energy_first_full.reshape(batch, groups * orientations, height, width)
        ).shape[-2:]

        def pool_ledger(value: torch.Tensor) -> torch.Tensor:
            pooled = self._pcqt_pool(
                value.reshape(batch, groups * orientations, height, width)
            )
            return pooled.reshape(batch, groups, orientations, *pool_shape)

        energy_first = pool_ledger(energy_first_full)
        energy_second = pool_ledger(energy_second_full)
        inner = pool_ledger(inner_full)
        denominator = energy_first + energy_second
        support_floor = transition._pcqt_support_floor.to(denominator)
        safe_denominator = denominator.clamp_min(support_floor)
        # Store both B and C coordinates, even though each arm writes only one.
        # This equalizes response, pooling and arithmetic work between B/C.
        coherence_unclamped = 2.0 * inner / safe_denominator
        marginal_unclamped = (energy_first - energy_second) / safe_denominator
        coherence = coherence_unclamped.clamp(-1.0, 1.0)
        marginal = marginal_unclamped.clamp(-1.0, 1.0)
        if self.pcqt_transition_mode == PCQT_COHERENCE_QUOTIENT:
            selected = coherence
        elif self.pcqt_transition_mode == PCQT_MARGINAL_ENERGY_CONTROL:
            selected = marginal
        else:  # pragma: no cover - constructor validates the finite mode set.
            raise RuntimeError(
                f"unexpected PCQT mode {self.pcqt_transition_mode!r}"
            )
        coordinate = selected.reshape(
            batch,
            groups * orientations,
            pool_shape[0],
            pool_shape[1],
        )
        summaries: dict[str, torch.Tensor] | None = None
        if collect_summaries:
            with torch.no_grad():
                # The two non-negative identities below are the numerical
                # certificate behind |2S/(E1+E2)| <= 1.  We record detached
                # scalars only; no large response ledger is retained in aux.
                minus = denominator.detach() - 2.0 * inner.detach()
                plus = denominator.detach() + 2.0 * inner.detach()
                summaries = {
                    "energy_first_mean": energy_first.detach().mean(
                        dim=(1, 2, 3, 4), dtype=torch.float32
                    ),
                    "energy_second_mean": energy_second.detach().mean(
                        dim=(1, 2, 3, 4), dtype=torch.float32
                    ),
                    "support_fraction": (denominator.detach() > support_floor)
                    .float()
                    .mean(dim=(1, 2, 3, 4)),
                    "coherence_abs_mean": coherence.detach().abs().mean(
                        dim=(1, 2, 3, 4), dtype=torch.float32
                    ),
                    "coherence_abs_max": coherence.detach().abs().amax(
                        dim=(1, 2, 3, 4)
                    ).float(),
                    "coherence_unclamped_excess": (
                        coherence_unclamped.detach().abs() - 1.0
                    )
                    .clamp_min(0.0)
                    .amax(dim=(1, 2, 3, 4))
                    .float(),
                    "marginal_abs_mean": marginal.detach().abs().mean(
                        dim=(1, 2, 3, 4), dtype=torch.float32
                    ),
                    "minus_energy_min": minus.amin(dim=(1, 2, 3, 4)).float(),
                    "plus_energy_min": plus.amin(dim=(1, 2, 3, 4)).float(),
                }
        return coordinate, summaries

    def _apply_pcqt_transition(
        self,
        *,
        stage_index: int,
        source_h: torch.Tensor,
        composition_state: torch.Tensor,
        collect_summaries: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        if self.pcqt_transition is None:
            return composition_state, None
        if int(stage_index) != 3:
            return composition_state, None
        coordinate, coordinate_aux = self._pcqt_coordinate(
            source_h,
            collect_summaries=collect_summaries,
        )
        updated, write_aux = self.pcqt_transition(coordinate, composition_state)
        if updated.shape != composition_state.shape:
            raise RuntimeError("PCQT residual does not match the C4 base state")
        summaries: dict[str, torch.Tensor] | None = None
        if coordinate_aux is not None:
            summaries = dict(coordinate_aux)
            summaries.update(write_aux)
        return updated, summaries

    def _stage_response_and_decomposition(
        self,
        h: torch.Tensor,
        *,
        stage_index: int,
        collect_ecqd_summaries: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
        """Run one response/quotient stage, including the isolated ECQD arm.

        The default path is intentionally the same bank call followed by the
        same quotient call as historical ARCQ.  Only active ECQD arms alter the
        terminal stage.  They use a stride-one matched response then the same
        fixed pooling support for both controls:

        ``B: Q(E(P(r)))`` versus ``C: Q(P(E(r)))``, where ``E`` first sums
        the squared even/odd phase pair and the ordinary ARCQ quotient ``Q``
        remains last in both arms.

        The terminal H response has no downstream carrier consumer, so the
        pooled raw response is returned solely to keep the pre-existing H-grid
        interface and terminal spatial size unchanged.
        """

        bank = self.response_banks[int(stage_index)]
        quotient = self.quotients[int(stage_index)]
        if not isinstance(quotient, RadialCompositionQuotient):
            # This branch is retained for the project-level principle controls
            # that deliberately substitute the quotient operator.  ECQD itself
            # is intentionally unavailable for those controls.
            if self.ecqd_downsample_mode != _ECQD_DISABLED:
                raise RuntimeError("active ECQD requires RadialCompositionQuotient")
            response = bank(h)
            return response, quotient(response), None

        if not (
            self.ecqd_downsample_mode != _ECQD_DISABLED
            and int(stage_index) == len(self.response_banks) - 1
        ):
            response = bank(h)
            return response, quotient(response), None

        # The learned bank is evaluated at every pre-decimation phase.  B and C
        # both pay for this exact stride-one bank, and both form both pooled
        # quantities below.  No trainable writer or extra feature channel is
        # introduced.
        full_response = bank(h, stride=1)
        full_energy = quotient.directional_energy(full_response)
        response_pooled = self._ecqd_pool(full_response)
        batch, groups, orientations, _, _ = full_energy.shape
        energy_pooled = self._ecqd_pool(
            full_energy.reshape(
                batch,
                groups * orientations,
                full_energy.shape[-2],
                full_energy.shape[-1],
            )
        ).reshape(
            batch,
            groups,
            orientations,
            response_pooled.shape[-2],
            response_pooled.shape[-1],
        )
        response_pooled_energy = quotient.directional_energy(response_pooled)

        if self.ecqd_downsample_mode == _ECQD_RESPONSE_AVERAGE_CONTROL:
            decomposition = quotient.forward_from_directional_energy(
                response_pooled_energy,
                output_dtype=response_pooled.dtype,
            )
        elif self.ecqd_downsample_mode == _ECQD_ENERGY_COMPLETE:
            decomposition = quotient.forward_from_directional_energy(
                energy_pooled,
                output_dtype=full_response.dtype,
            )
        else:  # pragma: no cover - constructor validates the finite mode set.
            raise RuntimeError(
                f"unexpected active ECQD mode {self.ecqd_downsample_mode!r}"
            )

        summaries: dict[str, torch.Tensor] | None = None
        if collect_ecqd_summaries:
            # The Jensen gap is a deterministic activity certificate for the
            # candidate's changed ordering.  It is not a performance metric and
            # stores only O(B) detached scalars, never a feature map.
            with torch.no_grad():
                # Do not retain a redundant full S4 autograd tensor merely to
                # report the deterministic Jensen activity certificate.
                gap = energy_pooled.detach() - response_pooled_energy.detach()
                summaries = {
                    "full_energy_mean": full_energy.detach().mean(
                        dim=(1, 2, 3, 4), dtype=torch.float32
                    ),
                    "pooled_energy_mean": energy_pooled.detach().mean(
                        dim=(1, 2, 3, 4), dtype=torch.float32
                    ),
                    "response_pooled_energy_mean": response_pooled_energy.detach().mean(
                        dim=(1, 2, 3, 4), dtype=torch.float32
                    ),
                    "jensen_gap_mean": gap.detach().mean(
                        dim=(1, 2, 3, 4), dtype=torch.float32
                    ),
                    "jensen_gap_min": gap.detach().amin(
                        dim=(1, 2, 3, 4)
                    ).float(),
                    "jensen_gap_positive_fraction": (
                        gap.detach() >= -1.0e-6
                    ).float().mean(dim=(1, 2, 3, 4)),
                }
        return response_pooled, decomposition, summaries

    def forward_h(
        self,
        x: torch.Tensor,
        *,
        return_responses: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"ARCQ H carrier expected Bx3xHxW, got {tuple(x.shape)}")
        h = x
        responses: list[torch.Tensor] = []
        if (
            self.ecqd_downsample_mode == _ECQD_DISABLED
            and self.pcqt_transition_mode == PCQT_DISABLED
        ):
            # Preserve the historic carrier-only path literally for default
            # checkpoints and diagnostic tooling.
            for bank, blocks in zip(self.response_banks, self.h_stages, strict=True):
                response = bank(h)
                responses.append(response)
                h = blocks(response)
        else:
            # A diagnostic caller must not silently inspect the legacy S4
            # stride-two response when deployment uses stride-one followed by
            # the registered fixed decimation. The terminal response therefore
            # exactly matches the active ECQD carrier path.
            for stage_index, blocks in enumerate(self.h_stages):
                response, _decomposition, _summaries = self._stage_response_and_decomposition(
                    h,
                    stage_index=stage_index,
                    collect_ecqd_summaries=False,
                )
                responses.append(response)
                h = blocks(response)
        if return_responses:
            return h, tuple(responses)
        return h

    def forward_protected(
        self,
        x: torch.Tensor,
        *,
        return_stage_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        composition_state: torch.Tensor | None = None
        stage_embeddings: list[torch.Tensor] = []
        if (
            self.ecqd_downsample_mode == _ECQD_DISABLED
            and self.pcqt_transition_mode == PCQT_DISABLED
        ):
            # Preserve the historical default arithmetic/order exactly.  The
            # new terminal ECQD helper is deliberately not even called here.
            _, responses = self.forward_h(x, return_responses=True)
            for stage_index, (response, quotient, projection, blocks) in enumerate(
                zip(
                    responses,
                    self.quotients,
                    self.c_projections,
                    self.c_stages,
                    strict=True,
                )
            ):
                decomposition = quotient(response)
                injection = projection(decomposition["composition"])
                if composition_state is None:
                    composition_state = injection
                else:
                    composition_state = self.c_transitions[stage_index - 1](composition_state) + injection
                composition_state, _ = self._apply_q3_spatial_organizer(
                    stage_index=stage_index,
                    composition_state=composition_state,
                    decomposition=decomposition,
                    collect_summaries=False,
                )
                composition_state = blocks(composition_state)
                stage_embeddings.append(composition_state.mean(dim=(2, 3)))
        else:
            # Any active terminal intervention must take precisely the same
            # protected C path as deployment.  ECQD changes the terminal
            # quotient ordering; PCQT keeps it intact but adds its C4-only
            # residual immediately before the C4 blocks.
            h = x
            for stage_index, (projection, blocks) in enumerate(
                zip(self.c_projections, self.c_stages, strict=True)
            ):
                source_h = h
                response, decomposition, _ = self._stage_response_and_decomposition(
                    h,
                    stage_index=stage_index,
                    collect_ecqd_summaries=False,
                )
                h = self.h_stages[stage_index](response)
                injection = projection(decomposition["composition"])
                if composition_state is None:
                    composition_state = injection
                else:
                    composition_state = self.c_transitions[stage_index - 1](composition_state) + injection
                composition_state, _ = self._apply_q3_spatial_organizer(
                    stage_index=stage_index,
                    composition_state=composition_state,
                    decomposition=decomposition,
                    collect_summaries=False,
                )
                composition_state, _ = self._apply_pcqt_transition(
                    stage_index=stage_index,
                    source_h=source_h,
                    composition_state=composition_state,
                    collect_summaries=False,
                )
                composition_state = blocks(composition_state)
                stage_embeddings.append(composition_state.mean(dim=(2, 3)))
        if composition_state is None:
            raise RuntimeError("ARCQ protected path has no stages")
        stages = tuple(stage_embeddings)
        self.last_protected_stages = stages
        if return_stage_embeddings:
            return composition_state, stages
        return composition_state

    @staticmethod
    def _composition_entropy(probabilities: torch.Tensor) -> torch.Tensor:
        orientations = probabilities.shape[2]
        entropy = -(
            probabilities.clamp_min(1.0e-12).log() * probabilities
        ).sum(dim=2)
        entropy = entropy / math.log(float(orientations))
        return entropy.mean(dim=(1, 2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"ARCQRoadBackbone expected Bx3xHxW, got {tuple(x.shape)}")
        appearance = self.appearance_stem(self._raw_appearance_input(x))
        if not self.use_raw_appearance:
            appearance = torch.zeros_like(appearance)
        h = x
        composition_state: torch.Tensor | None = None
        stage_log_energy = []
        stage_reliability = []
        stage_observability = []
        stage_unobservable_fraction = []
        stage_transition_fraction = []
        stage_fully_observable_fraction = []
        stage_entropy = []
        stage_protected_embeddings = []
        q3_probabilities: torch.Tensor | None = None
        q3_log_energy: torch.Tensor | None = None
        q3_reliability: torch.Tensor | None = None
        q3_spatial_aux: dict[str, torch.Tensor] | None = None
        ecqd_aux: dict[str, torch.Tensor] | None = None
        pcqt_aux: dict[str, torch.Tensor] | None = None

        for stage_index in range(4):
            source_h = h
            response, decomposition, stage_ecqd_aux = self._stage_response_and_decomposition(
                h,
                stage_index=stage_index,
                collect_ecqd_summaries=True,
            )
            h = self.h_stages[stage_index](response)
            if stage_ecqd_aux is not None:
                if ecqd_aux is not None:
                    raise RuntimeError("ECQD was applied more than once")
                ecqd_aux = stage_ecqd_aux
            if self.ri_caqc is not None and stage_index == 2:
                q3_probabilities = decomposition["probabilities"]
                q3_log_energy = decomposition["log_energy"]
                q3_reliability = decomposition["reliability"]
            if self.use_composition_evidence:
                composition_injection = self.c_projections[stage_index](
                    decomposition["composition"]
                )
            else:
                composition_injection = response.new_zeros(
                    response.shape[0],
                    self.evidence_channels[stage_index],
                    response.shape[2],
                    response.shape[3],
                )

            if composition_state is None:
                composition_state = composition_injection
            else:
                previous_appearance = appearance
                composition_state = (
                    self.c_transitions[stage_index - 1](composition_state)
                    + composition_injection
                )
                appearance = self.a_transitions[stage_index - 1](
                    previous_appearance
                )
                if stage_index == 3 and self.final_radiometric_sidecar is not None:
                    appearance = (
                        appearance
                        + self.final_radiometric_sidecar(previous_appearance)
                    )

            if self.allow_appearance_to_composition:
                appearance_mean = appearance.mean(dim=1, keepdim=True)
                appearance_variance = (
                    appearance - appearance_mean
                ).square().mean(dim=1, keepdim=True)
                standardized_appearance = (
                    appearance - appearance_mean
                ) * torch.rsqrt(appearance_variance + 1.0e-6)
                composition_state = (
                    composition_state
                    + self.a_injections[
                        stage_index
                    ].reverse_conditioned_write(standardized_appearance)
                )
            composition_state, stage_q3_spatial_aux = (
                self._apply_q3_spatial_organizer(
                    stage_index=stage_index,
                    composition_state=composition_state,
                    decomposition=decomposition,
                    collect_summaries=True,
                )
            )
            if stage_q3_spatial_aux is not None:
                if q3_spatial_aux is not None:
                    raise RuntimeError("PQSO was applied more than once")
                q3_spatial_aux = stage_q3_spatial_aux
            composition_state, stage_pcqt_aux = self._apply_pcqt_transition(
                stage_index=stage_index,
                source_h=source_h,
                composition_state=composition_state,
                collect_summaries=True,
            )
            if stage_pcqt_aux is not None:
                if pcqt_aux is not None:
                    raise RuntimeError("PCQT was applied more than once")
                pcqt_aux = stage_pcqt_aux
            composition_state = self.c_stages[stage_index](composition_state)
            stage_protected_embeddings.append(
                composition_state.mean(dim=(2, 3))
            )
            appearance = self.a_injections[stage_index](
                appearance,
                composition_state,
                decomposition["log_energy"]
                if self.use_radial_evidence
                else torch.zeros_like(decomposition["log_energy"]),
                decomposition["reliability"]
                if self.use_radial_evidence
                else torch.zeros_like(decomposition["reliability"]),
            )
            appearance = self.a_stages[stage_index](appearance)

            stage_log_energy.append(
                decomposition["log_energy"].mean(dim=(1, 2, 3))
            )
            stage_reliability.append(
                decomposition["reliability"].mean(dim=(1, 2, 3))
            )
            total_energy = decomposition["total_energy"]
            quotient = self.quotients[stage_index]
            if isinstance(quotient, RadialCompositionQuotient):
                observability = decomposition["observability"]
                observability_floor = quotient._observability_floor.to(
                    total_energy
                )
                fully_observable_energy = quotient._fully_observable_energy.to(
                    total_energy
                )
                unobservable = total_energy <= observability_floor
                fully_observable = total_energy >= fully_observable_energy
                transition = ~(unobservable | fully_observable)
            else:
                # Matched principle controls intentionally replace ARCQ's RCQ
                # operator. They have a single numerical support threshold but
                # no ARCQ smooth observability band, so report a binary
                # supported/unsupported diagnostic and an empty transition
                # region rather than fabricating ARCQ quotient semantics.
                support_floor = total_energy.new_tensor(
                    float(getattr(quotient, "numerical_eps", 0.0))
                )
                unobservable = total_energy < support_floor
                transition = torch.zeros_like(unobservable)
                fully_observable = ~unobservable
                observability = fully_observable.to(dtype=total_energy.dtype)
            stage_observability.append(
                observability.mean(dim=(1, 2, 3))
            )
            stage_unobservable_fraction.append(
                unobservable.float().mean(dim=(1, 2, 3))
            )
            stage_transition_fraction.append(
                transition.float().mean(dim=(1, 2, 3))
            )
            stage_fully_observable_fraction.append(
                fully_observable.float().mean(dim=(1, 2, 3))
            )
            stage_entropy.append(
                self._composition_entropy(decomposition["probabilities"])
            )

        if composition_state is None:
            raise RuntimeError("ARCQ backbone has no stages")
        toc_aux: dict[str, torch.Tensor] | None = None
        if self.terminal_octave is not None:
            # The two streams are consolidated independently; the protected C
            # unit never reads appearance, so ARCQ's absent A->C edge survives.
            composition_state, appearance = self.terminal_octave(
                composition_state,
                appearance,
            )
            toc_aux = self.terminal_octave.summaries(composition_state, appearance)
        readout_value = torch.tanh(self.readout_interaction(appearance))
        if self.readout_interaction_mode == "product":
            interaction = composition_state * readout_value
        elif self.readout_interaction_mode == "additive":
            # Parameter- and width-matched ordinary control for the original
            # product readout.  The sqrt(2) scale keeps the variance of two
            # roughly standardized inputs comparable without introducing a
            # learned gate or another experimental variable.
            interaction = (composition_state + readout_value) / math.sqrt(2.0)
        else:
            # Deletion diagnostic.  Keep the tensor width and downstream
            # projection contract unchanged so the only semantic change is
            # whether the third readout branch contributes evidence.
            interaction = torch.zeros_like(composition_state)
        feature_map = self.fusion_projection(
            torch.cat([composition_state, appearance, interaction], dim=1)
        )
        qodc_aux: dict[str, torch.Tensor] | None = None
        if self.qodc_lift is not None:
            feature_map, qodc_aux = self.qodc_lift(
                feature_map,
                composition_state,
                appearance,
            )
        base_feature = self.output_norm(feature_map.mean(dim=(2, 3)))
        feature = base_feature
        ri_aux: dict[str, torch.Tensor] | None = None
        if self.ri_caqc is not None:
            if (
                q3_probabilities is None
                or q3_log_energy is None
                or q3_reliability is None
            ):
                raise RuntimeError("RI-CAQC did not capture the S3 quotient ledger")
            ri_residual, ri_aux = self.ri_caqc(
                probabilities=q3_probabilities,
                log_energy=q3_log_energy,
                reliability=q3_reliability,
                appearance_state=appearance,
                composition_state=composition_state,
            )
            feature = base_feature + ri_residual.to(dtype=base_feature.dtype)

        self.last_feature_map = feature_map
        self.last_aux = {
            "protected_embedding": composition_state.mean(dim=(2, 3)),
            "appearance_embedding": appearance.mean(dim=(2, 3)),
            "mean_log_energy": torch.stack(stage_log_energy, dim=1),
            "mean_reliability": torch.stack(stage_reliability, dim=1),
            "mean_observability": torch.stack(stage_observability, dim=1),
            "unobservable_fraction": torch.stack(
                stage_unobservable_fraction,
                dim=1,
            ),
            "transition_fraction": torch.stack(stage_transition_fraction, dim=1),
            "fully_observable_fraction": torch.stack(
                stage_fully_observable_fraction,
                dim=1,
            ),
            "composition_entropy": torch.stack(stage_entropy, dim=1),
        }
        if (
            self.final_appearance_transition_mode
            != _FINAL_A_TRANSITION_LEARNED_SIGNED
        ):
            final_transition = self.a_transitions[-1]
            if not isinstance(final_transition, _BarycentricSpatialTransition):
                raise RuntimeError("active LBET mode lost its final transition")
            effective_kernel = final_transition.effective_kernel().detach().float()
            flattened_kernel = effective_kernel.reshape(
                effective_kernel.shape[0],
                -1,
            )
            self.last_aux.update(
                {
                    "lbet_kernel_min": flattened_kernel.min(),
                    "lbet_kernel_max": flattened_kernel.max(),
                    "lbet_kernel_sum_error": (
                        flattened_kernel.sum(dim=1) - 1.0
                    ).abs().max(),
                    "lbet_kernel_negative_fraction": (
                        flattened_kernel < 0.0
                    ).float().mean(),
                }
            )
        if ri_aux is not None:
            self.last_aux.update(
                {
                    "ri_caqc_descriptor": ri_aux["descriptor"],
                    "ri_caqc_residual": ri_aux["residual"],
                    "ri_caqc_radial_origin": ri_aux["radial_origin"],
                    "ri_caqc_composition_origin": ri_aux[
                        "composition_origin"
                    ],
                    "ri_caqc_mean_reliability_by_group": ri_aux[
                        "mean_reliability_by_group"
                    ],
                    "ri_caqc_residual_scale": ri_aux["residual_scale"],
                }
            )
        if q3_spatial_aux is not None:
            self.last_aux.update(
                {
                    "q3_spatial_source_rms": q3_spatial_aux["source_rms"],
                    "q3_spatial_source_abs_max": q3_spatial_aux[
                        "source_abs_max"
                    ],
                    "q3_spatial_relation_rms": q3_spatial_aux[
                        "relation_rms"
                    ],
                    "q3_spatial_writer_pre_tanh_rms": q3_spatial_aux[
                        "writer_pre_tanh_rms"
                    ],
                    "q3_spatial_residual_rms": q3_spatial_aux[
                        "residual_rms"
                    ],
                    "q3_spatial_saturation_fraction": q3_spatial_aux[
                        "saturation_fraction"
                    ],
                    "q3_spatial_c3_pre_rms": q3_spatial_aux["c3_pre_rms"],
                    "q3_spatial_c3_post_rms": q3_spatial_aux["c3_post_rms"],
                    "q3_spatial_c3_post_pre_rms_ratio": q3_spatial_aux[
                        "c3_post_pre_rms_ratio"
                    ],
                    "q3_spatial_residual_to_pre_rms_ratio": q3_spatial_aux[
                        "residual_to_pre_rms_ratio"
                    ],
                }
            )
        if qodc_aux is not None:
            self.last_aux.update(
            {
                    f"qodc_{key}": value.detach()
                for key, value in qodc_aux.items()
            }
        )
        if toc_aux is not None:
            self.last_aux.update(
                {key: value.detach() for key, value in toc_aux.items()}
            )
        if ecqd_aux is not None:
            self.last_aux.update(
                {f"ecqd_{key}": value.detach() for key, value in ecqd_aux.items()}
            )
        if pcqt_aux is not None:
            self.last_aux.update(
                {f"pcqt_{key}": value.detach() for key, value in pcqt_aux.items()}
            )
        self.last_protected_stages = tuple(stage_protected_embeddings)
        return feature


class ARCQRoadSurfaceClassifier(nn.Module):
    """Lightweight RSCD classifier used to measure the ARCQ backbone itself."""

    head_type = "linear"
    backbone_class = ARCQRoadBackbone

    def __init__(
        self,
        class_to_idx: dict[str, int],
        backbone_kwargs: dict[str, Any] | None = None,
        out_dim: int = 768,
        dropout: float = 0.1,
        use_aux_heads: bool = False,
        factor_head_mode: str = "legacy_coral",
        factor_evidence_product_scale: float = 0.0,
        nullspace_graph_readout_mode: str = "disabled",
        classifier_fp32_logits: bool = False,
    ) -> None:
        super().__init__()
        self.spec = build_rscd_factor_spec(class_to_idx)
        kwargs = dict(backbone_kwargs or {})
        if "out_dim" in kwargs:
            configured_out_dim = int(kwargs.pop("out_dim"))
            if configured_out_dim != int(out_dim):
                raise ValueError(
                    f"conflicting ARCQ out_dim values: {configured_out_dim} and {out_dim}"
                )
        if "pretrained" in kwargs:
            raise ValueError("ARCQ-Road has no public pretrained checkpoint")
        # Keep the classifier/head contract shared across capacity variants.
        # A variant changes only this class-level backbone factory; the loss,
        # normalization, dropout and 27-way readout remain identical.
        self.backbone = self.backbone_class(out_dim=int(out_dim), **kwargs)
        self.norm = nn.LayerNorm(int(out_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(out_dim), self.spec.num_classes)
        self.use_aux_heads = bool(use_aux_heads)
        self.classifier_fp32_logits = bool(classifier_fp32_logits)
        self.factor_head_mode = str(factor_head_mode).strip().lower()
        if self.factor_head_mode not in {"legacy_coral", "compact_categorical"}:
            raise ValueError(
                "factor_head_mode must be 'legacy_coral' or "
                "'compact_categorical'"
            )
        if self.factor_head_mode == "compact_categorical" and not self.use_aux_heads:
            raise ValueError(
                "factor_head_mode='compact_categorical' requires "
                "use_aux_heads=true"
            )
        product_scale = float(factor_evidence_product_scale)
        if not math.isfinite(product_scale) or product_scale < 0.0:
            raise ValueError(
                "factor_evidence_product_scale must be finite and non-negative"
            )
        if product_scale > 0.0 and not self.use_aux_heads:
            raise ValueError(
                "positive factor_evidence_product_scale requires use_aux_heads=true"
            )
        if product_scale > 0.0 and self.factor_head_mode != "compact_categorical":
            raise ValueError(
                "factor evidence fusion requires factor_head_mode="
                "'compact_categorical'"
            )
        nullspace_graph_readout_mode = str(
            nullspace_graph_readout_mode
        ).strip().lower()
        valid_nullspace_modes = {
            "disabled",
            ENDPOINT_ENERGY_CONTROL,
            GRID_SIGNED_CONSENSUS,
        }
        if nullspace_graph_readout_mode not in valid_nullspace_modes:
            raise ValueError(
                "nullspace_graph_readout_mode must be one of "
                f"{sorted(valid_nullspace_modes)}, got "
                f"{nullspace_graph_readout_mode!r}"
            )
        if nullspace_graph_readout_mode != "disabled":
            if not self.classifier_fp32_logits:
                raise ValueError(
                    "active nullspace_graph_readout_mode requires "
                    "classifier_fp32_logits=true so A/B/C use identical FP32 "
                    "linear logits and CE inputs"
                )
            if self.use_aux_heads:
                raise ValueError(
                    "the first NGSC causal gate requires use_aux_heads=false"
                )
            if product_scale != 0.0:
                raise ValueError(
                    "the first NGSC causal gate requires "
                    "factor_evidence_product_scale=0"
                )
            if str(self.backbone.q3_spatial_organizer_mode) != "disabled":
                raise ValueError(
                    "the first NGSC causal gate requires the Q3 spatial "
                    "organizer to be disabled"
                )
            if str(self.backbone.qodc_lift_mode) != "disabled":
                raise ValueError(
                    "the first NGSC causal gate requires qodc_lift_mode='disabled'"
                )
            if (
                str(self.backbone.ri_caqc_mode) != "disabled"
                or self.backbone.ri_caqc is not None
            ):
                raise ValueError(
                    "the first NGSC causal gate requires ri_caqc_mode="
                    "'disabled': otherwise the declared dual-LayerNorm "
                    "deployment tangent would omit the RI-CAQC residual"
                )
        if str(self.backbone.qodc_lift_mode) != "disabled":
            if self.use_aux_heads:
                raise ValueError(
                    "the first QODC causal gate requires use_aux_heads=false"
                )
            if product_scale != 0.0:
                raise ValueError(
                    "the first QODC causal gate requires "
                    "factor_evidence_product_scale=0"
                )
            if nullspace_graph_readout_mode != "disabled":
                raise ValueError(
                    "the first QODC causal gate requires the NGSC readout to be disabled"
                )
        if str(self.backbone.ecqd_downsample_mode) != _ECQD_DISABLED:
            # ECQD A/B/C is a single-backbone ordering experiment.  Auxiliary
            # factor losses, post-head products and graph readouts would turn a
            # result into a compound intervention rather than an evidence about
            # phase-complete quotient decimation.
            if self.use_aux_heads:
                raise ValueError("the first ECQD gate requires use_aux_heads=false")
            if product_scale != 0.0:
                raise ValueError(
                    "the first ECQD gate requires factor_evidence_product_scale=0"
                )
            if nullspace_graph_readout_mode != "disabled":
                raise ValueError(
                    "the first ECQD gate requires the NGSC readout to be disabled"
                )
        if str(self.backbone.pcqt_transition_mode) != PCQT_DISABLED:
            # PCQT is a protected C4 transition A/B/C, not a factor-head or
            # post-logit experiment.  Keep the classifier exactly CE-linear so
            # a score difference cannot be attributed to another late module.
            if self.use_aux_heads:
                raise ValueError("the first PCQT gate requires use_aux_heads=false")
            if product_scale != 0.0:
                raise ValueError(
                    "the first PCQT gate requires factor_evidence_product_scale=0"
                )
            if nullspace_graph_readout_mode != "disabled":
                raise ValueError(
                    "the first PCQT gate requires the NGSC readout to be disabled"
                )
        if (
            str(self.backbone.composition_coordinate_mode)
            != _ILR_ARCQ_CENTERED_SIMPLEX
        ):
            # The coordinate study remains a backbone-only intervention all
            # the way through the classifier.  These optional heads/readouts
            # are separately registered mechanisms, not part of the fixed
            # A/B/C coordinate comparison.
            if self.use_aux_heads:
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require use_aux_heads=false"
                )
            if product_scale != 0.0:
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require "
                    "factor_evidence_product_scale=0"
                )
            if nullspace_graph_readout_mode != "disabled":
                raise ValueError(
                    "active ILR-ARCQ coordinate modes require the NGSC "
                    "readout to be disabled"
                )
        self.nullspace_graph_readout_mode = nullspace_graph_readout_mode
        # Constructing NGSC consumes no RNG.  Omitting the module entirely in
        # the disabled/default path preserves every legacy parameter, buffer,
        # state-dict key and random-number transition.
        self.nullspace_graph_readout = (
            NullspaceGraphSignedConsensus(mode=nullspace_graph_readout_mode)
            if nullspace_graph_readout_mode != "disabled"
            else None
        )
        self.register_buffer(
            "factor_evidence_product_scale",
            torch.tensor(product_scale, dtype=torch.float32),
            # The scale is a frozen config hyperparameter, not learned state.
            # Keeping it out of state_dict preserves strict loading of every
            # pre-existing ARCQ checkpoint; the checkpoint/config contract
            # still binds candidate evaluation to the same fixed value.
            persistent=False,
        )
        final_evidence_dim = int(self.backbone.evidence_channels[-1])
        self.factor_evidence_product = (
            FactorEvidenceProductResidual(self.spec.class_to_factor)
            if self.factor_head_mode == "compact_categorical"
            else None
        )
        factor_heads: nn.ModuleDict | None = None
        if self.use_aux_heads:
            factor_heads = nn.ModuleDict(
                {
                    "friction": nn.Linear(
                        final_evidence_dim,
                        len(FACTOR_LABELS["friction"]),
                    ),
                    "material": nn.Linear(
                        final_evidence_dim,
                        len(FACTOR_LABELS["material"]) - 1,
                    ),
                }
            )
            if self.factor_head_mode == "compact_categorical":
                factor_heads["roughness"] = nn.Linear(
                    int(out_dim),
                    len(FACTOR_LABELS["roughness"]) - 1,
                )
                if self.factor_evidence_product is None:
                    raise RuntimeError("factor evidence mapping was not initialized")
                self.factor_evidence_product.initialize_factor_heads(factor_heads)
        self.factor_heads = factor_heads
        self.roughness_coral_head = (
            CoralOrdinalHead(int(out_dim), num_classes=3)
            if self.use_aux_heads and self.factor_head_mode == "legacy_coral"
            else None
        )
        self.head_type = (
            "factor_evidence_product_residual"
            if product_scale > 0.0
            else "linear"
        )

    def forward(
        self,
        image: torch.Tensor,
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        feature = self.norm(self.backbone(image))
        dropped = self.dropout(feature)
        if self.classifier_fp32_logits:
            # A/B/C share this explicit FP32 projection.  This makes the
            # treatment variable solely the NGSC graph statistic instead of
            # accidentally mixing it with an autocast/dtype change.
            with torch.amp.autocast(
                device_type=image.device.type,
                enabled=False,
            ):
                linear_logits = F.linear(
                    dropped.float(),
                    self.head.weight.float(),
                    None if self.head.bias is None else self.head.bias.float(),
                )
        else:
            # Default ARCQ behavior remains byte-for-byte compatible.
            linear_logits = self.head(dropped)
        backbone_aux = self.backbone.last_aux
        protected_stage_embeddings = self.backbone.last_protected_stages
        appearance_embedding = backbone_aux["appearance_embedding"]
        protected_embedding = backbone_aux["protected_embedding"]
        factor_logits: dict[str, torch.Tensor] = {}
        roughness_coral_logits: torch.Tensor | None = None
        if self.factor_heads is not None:
            if self.factor_head_mode == "compact_categorical":
                # Keep prior-identity exact under CUDA BF16/FP16.  Autocast
                # would round the log-prior bias before the FP32 evidence
                # subtraction, creating a small non-zero product residual at
                # initialization even though every factor head has zero
                # weights.  The compact heads are tiny, so computing only
                # these three projections in FP32 has negligible cost.
                with torch.amp.autocast(
                    device_type=image.device.type,
                    enabled=False,
                ):
                    factor_logits = {
                        "friction": self.factor_heads["friction"](
                            self.dropout(appearance_embedding.float())
                        ),
                        "material": self.factor_heads["material"](
                            self.dropout(protected_embedding.float())
                        ),
                        "roughness": self.factor_heads["roughness"](
                            dropped.float()
                        ),
                    }
            else:
                factor_logits = {
                    "friction": self.factor_heads["friction"](
                        self.dropout(appearance_embedding)
                    ),
                    "material": self.factor_heads["material"](
                        self.dropout(protected_embedding)
                    ),
                }
                if self.roughness_coral_head is None:
                    raise RuntimeError(
                        "ARCQ auxiliary heads are incompletely initialized"
                    )
                roughness_coral_logits = self.roughness_coral_head(dropped)

        factor_evidence_logits: torch.Tensor | None = None
        if self.factor_evidence_product is not None:
            factor_evidence_logits = self.factor_evidence_product(factor_logits)
        ngsc_correction: torch.Tensor | None = None
        ngsc_aux: dict[str, torch.Tensor] | None = None
        if self.nullspace_graph_readout is not None:
            feature_map = self.backbone.last_feature_map
            if not torch.is_tensor(feature_map):
                raise RuntimeError(
                    "ARCQ backbone did not expose last_feature_map for NGSC"
                )
            logits = self.nullspace_graph_readout(
                feature_map=feature_map,
                base_logits=linear_logits,
                first_norm=self.backbone.output_norm,
                second_norm=self.norm,
                head=self.head,
            )
            ngsc_correction = logits - linear_logits.float()
            with torch.no_grad():
                ngsc_aux = {
                    key: value.detach()
                    for key, value in self.nullspace_graph_readout.last_aux.items()
                }
                ngsc_aux.update(
                    {
                        "correction_rms": ngsc_correction.detach()
                        .square()
                        .mean(dim=1)
                        .sqrt(),
                        "correction_abs_max": ngsc_correction.detach()
                        .abs()
                        .amax(dim=1),
                        "correction_class_sum_abs": ngsc_correction.detach()
                        .sum(dim=1)
                        .abs(),
                        "correction_nonzero_class_fraction": (
                            ngsc_correction.detach().abs() > 1.0e-12
                        )
                        .float()
                        .mean(dim=1),
                    }
                )
        elif factor_evidence_logits is None:
            # Preserve the exact legacy linear-head dtype and behavior when
            # compact factor evidence is disabled.
            logits = linear_logits
        else:
            # A 0.10 evidence correction is intentionally weak.  Casting it to
            # BF16/FP16 before adding it to an O(1) logit can round the entire
            # treatment to zero (and FP16 can overflow for a very unlikely
            # state).  Both compact B/C arms therefore fuse in FP32, including
            # the scale-zero control, while the legacy classifier remains
            # untouched above.
            with torch.amp.autocast(
                device_type=image.device.type,
                enabled=False,
            ):
                scale = self.factor_evidence_product_scale.to(
                    device=linear_logits.device,
                    dtype=torch.float32,
                )
                logits = (
                    linear_logits.float()
                    + scale * factor_evidence_logits.float()
                )
        if not return_aux:
            return logits

        output: dict[str, Any] = {
            "logits": logits,
            "linear_logits": linear_logits,
            "feature": feature,
            "features": feature,
            "factor_logits": factor_logits,
            "roughness_coral_logits": roughness_coral_logits,
            "boundary_logits": {},
            "arcq_appearance_embedding": appearance_embedding,
            "arcq_protected_embedding": protected_embedding,
            "arcq_protected_stage_embeddings": protected_stage_embeddings,
            "arcq_aux": backbone_aux,
        }
        if factor_evidence_logits is not None:
            output["factor_evidence_logits"] = factor_evidence_logits
            output["factor_evidence_residual"] = (
                logits - linear_logits
            )
        if ngsc_correction is not None:
            output["nullspace_graph_readout_correction"] = ngsc_correction
            output["nullspace_graph_readout_alpha"] = (
                self.nullspace_graph_readout.alpha
            )
            output["nullspace_graph_readout_aux"] = ngsc_aux
        return output
