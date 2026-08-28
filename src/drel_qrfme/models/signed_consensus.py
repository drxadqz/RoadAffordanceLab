"""Nullspace Graph Signed-Consensus (NGSC) readout for ARCQ-Road.

NGSC is a classifier-side experimental readout.  It does not modify the ARCQ
carrier or its protected composition state.  Instead, it asks whether spatial
evidence that is *exactly invisible to global average pooling* becomes useful
after the existing classifier has assigned class semantics to feature
directions.

For every class, the module constructs the class-vs-rest soft-margin tangent
through the exact two affine LayerNorms used by ARCQ's deterministic deployment
path.  The tangent is recomputed from the current model on every forward pass,
but is deliberately detached.  Consequently:

* no ``autograd.grad`` or higher-order graph is built;
* the local projection remains first-order differentiable with respect to the
  live feature map;
* the classifier head and LayerNorms continue to train through the ordinary
  base-logit path; and
* training Dropout is not sampled or reused by NGSC.

The two causal arms have identical trainable state (one zero-initialized scalar)
and differ only in their fixed grid statistic:

``endpoint_energy_control`` (B)
    Measures endpoint self-energy on the same valid four-neighbour edges.

``grid_signed_consensus`` (C)
    Measures same-sign products across those edges.  For any scalar grid x,
    endpoint_energy(x) - grid_consensus(x) is exactly the non-negative graph
    Dirichlet energy, up to finite-precision roundoff.

All NGSC arithmetic is FP32.  The public disabled ARCQ path is intentionally not
implemented here: integration should simply omit this module so legacy logits,
dtype, RNG behaviour, and checkpoints remain byte-for-byte unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F


__all__ = [
    "ENDPOINT_ENERGY_CONTROL",
    "GRID_SIGNED_CONSENSUS",
    "NullspaceGraphSignedConsensus",
    "class_vs_rest_competitor_weights",
    "detached_deployment_margin_tangent",
    "endpoint_energy",
    "grid_signed_consensus",
    "layer_norm_vjp",
    "nullspace_local_evidence",
]


ENDPOINT_ENERGY_CONTROL: Final[str] = "endpoint_energy_control"
GRID_SIGNED_CONSENSUS: Final[str] = "grid_signed_consensus"
_LAMBDA_MAX: Final[float] = 0.10


def _require_floating(tensor: Tensor, *, name: str) -> None:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


def _validate_layer_norm(norm: nn.LayerNorm, width: int, *, name: str) -> None:
    if not isinstance(norm, nn.LayerNorm):
        raise TypeError(f"{name} must be torch.nn.LayerNorm")
    normalized_shape = tuple(int(value) for value in norm.normalized_shape)
    if normalized_shape != (width,):
        raise ValueError(
            f"{name}.normalized_shape must be ({width},), got "
            f"{normalized_shape}"
        )
    if not math.isfinite(float(norm.eps)) or float(norm.eps) <= 0.0:
        raise ValueError(f"{name}.eps must be finite and positive")


def _validate_projection_inputs(
    feature_map: Tensor,
    first_norm: nn.LayerNorm,
    second_norm: nn.LayerNorm,
    head: nn.Linear,
) -> tuple[int, int, int, int]:
    _require_floating(feature_map, name="feature_map")
    if feature_map.ndim != 4:
        raise ValueError(
            "feature_map must have shape BxDxHxW, got "
            f"{tuple(feature_map.shape)}"
        )
    batch, width, height, spatial_width = (
        int(value) for value in feature_map.shape
    )
    if batch <= 0 or width <= 0:
        raise ValueError("feature_map batch and channel dimensions must be positive")
    if height < 2 or spatial_width < 2:
        raise ValueError(
            "NGSC requires H>=2 and W>=2 so the valid four-neighbour graph "
            f"is non-empty, got H={height}, W={spatial_width}"
        )
    _validate_layer_norm(first_norm, width, name="first_norm")
    _validate_layer_norm(second_norm, width, name="second_norm")
    if not isinstance(head, nn.Linear):
        raise TypeError("head must be torch.nn.Linear")
    if int(head.in_features) != width:
        raise ValueError(
            f"head.in_features must be {width}, got {int(head.in_features)}"
        )
    classes = int(head.out_features)
    if classes < 2:
        raise ValueError("NGSC class-vs-rest margins require at least two classes")

    tensors: list[tuple[str, Tensor | None]] = [
        ("first_norm.weight", first_norm.weight),
        ("first_norm.bias", first_norm.bias),
        ("second_norm.weight", second_norm.weight),
        ("second_norm.bias", second_norm.bias),
        ("head.weight", head.weight),
        ("head.bias", head.bias),
    ]
    for tensor_name, tensor in tensors:
        if tensor is not None and tensor.device != feature_map.device:
            raise ValueError(
                f"{tensor_name} is on {tensor.device}, but feature_map is on "
                f"{feature_map.device}"
            )
    return batch, width, height, spatial_width


def _layer_norm_forward_fp32(
    value: Tensor,
    norm: nn.LayerNorm,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate affine LayerNorm and retain its analytic VJP state.

    The biased population variance matches ``torch.nn.LayerNorm``.  Returning
    ``x_hat`` and reciprocal standard deviation avoids recovering normalized
    coordinates from affine outputs, which would be invalid when a learned
    scale is zero.
    """

    value = value.float()
    mean = value.mean(dim=-1, keepdim=True)
    centered = value - mean
    inverse_std = torch.rsqrt(
        centered.square().mean(dim=-1, keepdim=True) + float(norm.eps)
    )
    normalized = centered * inverse_std
    if norm.weight is None:
        scale = torch.ones(
            int(value.shape[-1]), device=value.device, dtype=torch.float32
        )
    else:
        scale = norm.weight.float()
    output = normalized * scale
    if norm.bias is not None:
        output = output + norm.bias.float()
    return output, normalized, inverse_std, scale


def layer_norm_vjp(
    upstream: Tensor,
    normalized: Tensor,
    inverse_std: Tensor,
    scale: Tensor,
) -> Tensor:
    r"""Apply the exact analytic VJP of one affine LayerNorm in FP32.

    For ``y = gamma * x_hat + beta`` and upstream vector ``v``, define
    ``a = gamma * v``.  The vector-Jacobian product is

    ``rstd * (a - mean(a) - x_hat * mean(a * x_hat))``.

    ``upstream`` may contain any number of axes between the batch and feature
    axes.  NGSC uses shape ``B x K x D`` to evaluate all K classes at once.
    """

    for name, tensor in (
        ("upstream", upstream),
        ("normalized", normalized),
        ("inverse_std", inverse_std),
        ("scale", scale),
    ):
        _require_floating(tensor, name=name)
    if upstream.ndim < 2:
        raise ValueError("upstream must have at least batch and feature axes")
    if normalized.ndim != 2:
        raise ValueError("normalized must have shape BxD")
    if inverse_std.shape != (normalized.shape[0], 1):
        raise ValueError("inverse_std must have shape Bx1")
    if scale.shape != (normalized.shape[1],):
        raise ValueError("scale must have shape D")
    if upstream.shape[0] != normalized.shape[0]:
        raise ValueError("upstream and normalized batch dimensions must match")
    if upstream.shape[-1] != normalized.shape[-1]:
        raise ValueError("upstream and normalized feature dimensions must match")

    extra_axes = upstream.ndim - 2
    normalized_view = normalized.float().reshape(
        normalized.shape[0], *((1,) * extra_axes), normalized.shape[1]
    )
    inverse_std_view = inverse_std.float().reshape(
        inverse_std.shape[0], *((1,) * extra_axes), 1
    )
    scale_view = scale.float().reshape(
        *((1,) * (upstream.ndim - 1)), scale.shape[0]
    )
    affine_upstream = upstream.float() * scale_view
    return inverse_std_view * (
        affine_upstream
        - affine_upstream.mean(dim=-1, keepdim=True)
        - normalized_view
        * (affine_upstream * normalized_view).mean(dim=-1, keepdim=True)
    )


def class_vs_rest_competitor_weights(logits: Tensor) -> Tensor:
    r"""Return vectorized soft competitors for every class.

    Row ``y`` contains the softmax over all classes except ``y``.  It is the
    derivative of ``logsumexp(logits[j], j != y)`` and therefore defines the
    deterministic class-vs-rest soft margin

    ``logits[y] - logsumexp(logits[j], j != y)``.

    The construction is invariant to a common logit shift and never divides by
    ``1 - p_y``, avoiding the catastrophic cancellation of probability-ratio
    implementations.
    """

    _require_floating(logits, name="logits")
    if logits.ndim != 2:
        raise ValueError("logits must have shape BxK")
    classes = int(logits.shape[1])
    if classes < 2:
        raise ValueError("class-vs-rest weights require at least two classes")
    logits = logits.float()
    diagonal = torch.eye(classes, device=logits.device, dtype=torch.bool)
    competitors = logits[:, None, :].expand(-1, classes, -1).masked_fill(
        diagonal[None, :, :],
        -torch.inf,
    )
    return torch.softmax(competitors, dim=-1)


def _dual_layer_norm_margin_tangent(
    pooled_feature: Tensor,
    first_norm: nn.LayerNorm,
    second_norm: nn.LayerNorm,
    head: nn.Linear,
) -> tuple[Tensor, Tensor]:
    first_output, first_normalized, first_inverse_std, first_scale = (
        _layer_norm_forward_fp32(pooled_feature, first_norm)
    )
    second_output, second_normalized, second_inverse_std, second_scale = (
        _layer_norm_forward_fp32(first_output, second_norm)
    )
    deployment_logits = F.linear(
        second_output,
        head.weight.float(),
        None if head.bias is None else head.bias.float(),
    )
    competitors = class_vs_rest_competitor_weights(deployment_logits)
    # (I - pi_y) W, written without allocating another BxKxK matrix.
    second_upstream = head.weight.float().unsqueeze(0) - torch.matmul(
        competitors,
        head.weight.float(),
    )
    first_upstream = layer_norm_vjp(
        second_upstream,
        second_normalized,
        second_inverse_std,
        second_scale,
    )
    tangent = layer_norm_vjp(
        first_upstream,
        first_normalized,
        first_inverse_std,
        first_scale,
    )
    return tangent, deployment_logits


def detached_deployment_margin_tangent(
    feature_map: Tensor,
    first_norm: nn.LayerNorm,
    second_norm: nn.LayerNorm,
    head: nn.Linear,
) -> Tensor:
    """Return the stop-gradient, no-Dropout deployment-margin tangent.

    This is the exact analytic derivative of the deterministic mathematical
    evaluator implemented above.  It is intentionally *not* the gradient of a
    stochastic train-time Dropout realization.  The returned tensor never has
    an autograd history, even if all inputs require gradients.
    """

    _validate_projection_inputs(feature_map, first_norm, second_norm, head)
    with torch.no_grad(), torch.autocast(
        device_type=feature_map.device.type,
        enabled=False,
    ):
        pooled_feature = feature_map.detach().float().mean(dim=(-2, -1))
        tangent, _ = _dual_layer_norm_margin_tangent(
            pooled_feature,
            first_norm,
            second_norm,
            head,
        )
    return tangent.detach()


def nullspace_local_evidence(feature_map: Tensor, tangent: Tensor) -> Tensor:
    r"""Project class tangents onto the exact spatial nullspace of GAP.

    ``r_y(u) = g_y^T (F(u) - GAP(F))`` has zero spatial mean analytically.  A
    second explicit centering removes only finite-precision accumulation error.
    The tangent is required to be detached so callers cannot accidentally turn
    NGSC into a higher-order gradient method.
    """

    _require_floating(feature_map, name="feature_map")
    _require_floating(tangent, name="tangent")
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape BxDxHxW")
    if tangent.ndim != 3:
        raise ValueError("tangent must have shape BxKxD")
    if tangent.requires_grad or tangent.grad_fn is not None:
        raise ValueError("tangent must be detached before local projection")
    if feature_map.shape[0] != tangent.shape[0]:
        raise ValueError("feature_map and tangent batch dimensions must match")
    if feature_map.shape[1] != tangent.shape[2]:
        raise ValueError("feature_map and tangent feature dimensions must match")
    feature_map_fp32 = feature_map.float()
    pooled = feature_map_fp32.mean(dim=(-2, -1), keepdim=True)
    evidence = torch.einsum(
        "bkd,bdhw->bkhw",
        tangent.float(),
        feature_map_fp32 - pooled,
    )
    return evidence - evidence.mean(dim=(-2, -1), keepdim=True)


def _validate_grid(value: Tensor, *, name: str) -> tuple[int, int, int]:
    _require_floating(value, name=name)
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape BxKxHxW")
    height, width = int(value.shape[-2]), int(value.shape[-1])
    if height < 2 or width < 2:
        raise ValueError(
            f"{name} requires H>=2 and W>=2, got H={height}, W={width}"
        )
    edges = height * (width - 1) + (height - 1) * width
    return height, width, edges


def endpoint_energy(value: Tensor) -> Tensor:
    r"""B-arm endpoint self-energy on valid horizontal and vertical edges."""

    _, _, edges = _validate_grid(value, name="value")
    horizontal = (
        value[..., :, :-1].square() + value[..., :, 1:].square()
    ).sum(dim=(-2, -1))
    vertical = (
        value[..., :-1, :].square() + value[..., 1:, :].square()
    ).sum(dim=(-2, -1))
    return (horizontal + vertical) / float(2 * edges)


def grid_signed_consensus(value: Tensor) -> Tensor:
    r"""C-arm same-sign consensus on valid horizontal and vertical edges."""

    _, _, edges = _validate_grid(value, name="value")
    horizontal = (value[..., :, :-1] * value[..., :, 1:]).sum(dim=(-2, -1))
    vertical = (value[..., :-1, :] * value[..., 1:, :]).sum(dim=(-2, -1))
    return (horizontal + vertical) / float(edges)


def _signed_graph_statistics(
    local_evidence: Tensor,
    *,
    mode: str,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_grid(local_evidence, name="local_evidence")
    bounded = local_evidence.float() / (1.0 + local_evidence.float().abs())
    positive = F.relu(bounded)
    negative = F.relu(-bounded)
    if mode == ENDPOINT_ENERGY_CONTROL:
        positive_statistic = endpoint_energy(positive)
        negative_statistic = endpoint_energy(negative)
    elif mode == GRID_SIGNED_CONSENSUS:
        positive_statistic = grid_signed_consensus(positive)
        negative_statistic = grid_signed_consensus(negative)
    else:  # pragma: no cover - constructor and mode-code audit fail earlier.
        raise RuntimeError(f"unsupported NGSC mode {mode!r}")
    return (
        positive_statistic - negative_statistic,
        positive_statistic,
        negative_statistic,
    )


def _centre_and_bound_signed_statistic(signed_statistic: Tensor) -> Tensor:

    # Centre once before bounding so a common class offset cannot drive tanh;
    # centre a second time after tanh so the final residual has exact zero
    # class sum (up to floating-point accumulation order).
    signed_statistic = signed_statistic - signed_statistic.mean(
        dim=1, keepdim=True
    )
    bounded_statistic = torch.tanh(signed_statistic)
    return 0.5 * (
        bounded_statistic - bounded_statistic.mean(dim=1, keepdim=True)
    )


def _signed_graph_delta(local_evidence: Tensor, *, mode: str) -> Tensor:
    signed_statistic, _, _ = _signed_graph_statistics(
        local_evidence,
        mode=mode,
    )
    return _centre_and_bound_signed_statistic(signed_statistic)


class NullspaceGraphSignedConsensus(nn.Module):
    """One-scalar NGSC residual with checkpoint-separated B/C causal arms."""

    VALID_MODES: Final[frozenset[str]] = frozenset(
        {ENDPOINT_ENERGY_CONTROL, GRID_SIGNED_CONSENSUS}
    )
    _MODE_CODES: Final[dict[str, int]] = {
        ENDPOINT_ENERGY_CONTROL: 0,
        GRID_SIGNED_CONSENSUS: 1,
    }
    lambda_max: Final[float] = _LAMBDA_MAX

    def __init__(self, *, mode: str) -> None:
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(self.VALID_MODES)}, got {mode!r}"
            )
        self.mode = mode
        self.alpha = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "_mode_code",
            torch.tensor(self._MODE_CODES[mode], dtype=torch.int64),
        )
        # Detached per-sample scalar summaries only.  No feature map is kept,
        # and this non-state diagnostic dictionary never enters checkpoints.
        self.last_aux: dict[str, Tensor] = {}

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.alpha)

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
        state_name = prefix + "_mode_code"
        supplied = state_dict.get(state_name)
        invalid = supplied is not None and not (
            torch.is_tensor(supplied)
            and supplied.dtype == torch.int64
            and supplied.shape == torch.Size([])
            and torch.equal(supplied.detach().cpu(), self._mode_code.cpu())
        )
        if invalid:
            error_msgs.append(
                f'Invalid NGSC mode buffer "{state_name}": a checkpoint from '
                "the other causal arm cannot be loaded."
            )
            # ``nn.Module`` would otherwise copy the incompatible buffer before
            # raising.  Substitute the expected value only for the duration of
            # the parent loader, then restore the caller's state dictionary.
            state_dict[state_name] = self._mode_code.detach().clone()
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
            if invalid:
                assert supplied is not None
                state_dict[state_name] = supplied

    def delta(
        self,
        feature_map: Tensor,
        *,
        first_norm: nn.LayerNorm,
        second_norm: nn.LayerNorm,
        head: nn.Linear,
    ) -> Tensor:
        """Return the bounded, twice class-centred B/C statistic in FP32."""

        expected_code = self._MODE_CODES[self.mode]
        if (
            self._mode_code.dtype != torch.int64
            or self._mode_code.shape != torch.Size([])
            or int(self._mode_code.item()) != expected_code
        ):
            raise RuntimeError("NGSC mode buffer does not match the constructed arm")
        _validate_projection_inputs(feature_map, first_norm, second_norm, head)
        tangent = detached_deployment_margin_tangent(
            feature_map,
            first_norm,
            second_norm,
            head,
        )
        with torch.autocast(device_type=feature_map.device.type, enabled=False):
            local_evidence = nullspace_local_evidence(feature_map, tangent)
            signed_statistic, positive_statistic, negative_statistic = (
                _signed_graph_statistics(local_evidence, mode=self.mode)
            )
            delta = _centre_and_bound_signed_statistic(signed_statistic)
            with torch.no_grad():
                self.last_aux = {
                    "local_evidence_rms": local_evidence.detach()
                    .square()
                    .mean(dim=(1, 2, 3))
                    .sqrt(),
                    "positive_statistic_rms": positive_statistic.detach()
                    .square()
                    .mean(dim=1)
                    .sqrt(),
                    "negative_statistic_rms": negative_statistic.detach()
                    .square()
                    .mean(dim=1)
                    .sqrt(),
                    "delta_rms": delta.detach()
                    .square()
                    .mean(dim=1)
                    .sqrt(),
                }
            return delta

    def correction(
        self,
        feature_map: Tensor,
        *,
        first_norm: nn.LayerNorm,
        second_norm: nn.LayerNorm,
        head: nn.Linear,
    ) -> Tensor:
        """Return the zero-sum correction bounded by 0.10 per class."""

        delta = self.delta(
            feature_map,
            first_norm=first_norm,
            second_norm=second_norm,
            head=head,
        )
        return _LAMBDA_MAX * torch.tanh(self.alpha.float()) * delta

    def forward(
        self,
        feature_map: Tensor,
        base_logits: Tensor,
        *,
        first_norm: nn.LayerNorm,
        second_norm: nn.LayerNorm,
        head: nn.Linear,
    ) -> Tensor:
        """Add NGSC to live base logits while preserving FP32 CE semantics."""

        _require_floating(base_logits, name="base_logits")
        batch, _, _, _ = _validate_projection_inputs(
            feature_map,
            first_norm,
            second_norm,
            head,
        )
        expected_shape = (batch, int(head.out_features))
        if tuple(base_logits.shape) != expected_shape:
            raise ValueError(
                f"base_logits must have shape {expected_shape}, got "
                f"{tuple(base_logits.shape)}"
            )
        if base_logits.device != feature_map.device:
            raise ValueError("base_logits and feature_map must be on the same device")
        correction = self.correction(
            feature_map,
            first_norm=first_norm,
            second_norm=second_norm,
            head=head,
        )
        return base_logits.float() + correction
