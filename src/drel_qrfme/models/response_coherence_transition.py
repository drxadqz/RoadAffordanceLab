from __future__ import annotations

"""PCQT's deliberately small, protected C4 residual writer.

This module does *not* define a second feature backbone.  Its input coordinate
is constructed in :mod:`arcq_road` from two calls to the same constrained S4
matched-response bank.  Keeping the writer separate makes the causal contract
auditable: PCQT can only write a bounded residual into the protected C4 state;
it cannot observe RGB, the appearance state, logits, or labels.

The immutable buffers are intentionally part of the state dictionary.  A
checkpoint trained with the marginal-energy control must never be loaded as the
coherence-quotient candidate merely because their learned tensor shapes happen
to agree.
"""

import math
from typing import Any

import torch
from torch import nn


__all__ = [
    "PCQT_COHERENCE_QUOTIENT",
    "PCQT_DISABLED",
    "PCQT_MARGINAL_ENERGY_CONTROL",
    "PairedResponseCoherenceTransition",
]


PCQT_DISABLED = "disabled"
PCQT_MARGINAL_ENERGY_CONTROL = "marginal_energy_control"
PCQT_COHERENCE_QUOTIENT = "coherence_quotient"

_MODE_CODES = {
    PCQT_MARGINAL_ENERGY_CONTROL: 1,
    PCQT_COHERENCE_QUOTIENT: 2,
}
_SCHEMA_VERSION = 1


class PairedResponseCoherenceTransition(nn.Module):
    r"""A zero-initialized bounded write from PCQT coordinates into C4.

    Let ``z`` be either the matched marginal-energy coordinate or the matched
    signed coherence coordinate.  The map implemented here is

    .. math::

       q=Wz,\quad \hat q=q/(\operatorname{RMS}(q)^2+\delta^2)^{1/2},\quad
       \Delta C=\rho_{\max}\tanh(a)\,\operatorname{RMS}_{\rm sg}(C)\,\hat q.

    ``a`` begins at zero, so an active PCQT arm starts with exactly the base
    ARCQ function.  ``W`` is deliberately non-zero and bias-free: ``a`` gets a
    first-order gradient on a non-degenerate sample, and once it moves the
    writer can learn without manufacturing an unconditional offset when the
    PCQT coordinate is zero.
    """

    def __init__(
        self,
        *,
        mode: str,
        source_h_channels: int,
        response_channels: int,
        groups: int,
        orientations: int,
        phases: int,
        coordinate_channels: int,
        target_channels: int,
        source_stage: int = 2,
        target_stage: int = 3,
        dilation_pair: tuple[int, int] = (1, 2),
        pool_spec: tuple[int, int, int] = (3, 2, 1),
        support_floor: float = 1.0e-8,
        normalization_delta: float = 1.0e-4,
        residual_max: float = 0.10,
    ) -> None:
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in _MODE_CODES:
            raise ValueError(
                f"mode must be one of {sorted(_MODE_CODES)}, got {mode!r}"
            )
        values = {
            "source_h_channels": int(source_h_channels),
            "response_channels": int(response_channels),
            "groups": int(groups),
            "orientations": int(orientations),
            "phases": int(phases),
            "coordinate_channels": int(coordinate_channels),
            "target_channels": int(target_channels),
        }
        if any(value <= 0 for value in values.values()):
            raise ValueError("all PCQT channel/group geometry values must be positive")
        if values["response_channels"] != (
            values["groups"] * values["orientations"] * values["phases"]
        ):
            raise ValueError("PCQT response geometry must equal groups*orientations*phases")
        if values["coordinate_channels"] != (
            values["groups"] * values["orientations"]
        ):
            raise ValueError("PCQT coordinates must contain one value per group/orientation")
        if tuple(int(value) for value in dilation_pair) != (1, 2):
            raise ValueError("PCQT v1 requires the fixed dilation pair (1, 2)")
        if tuple(int(value) for value in pool_spec) != (3, 2, 1):
            raise ValueError("PCQT v1 requires the fixed pool specification (3, 2, 1)")
        if int(source_stage) != 2 or int(target_stage) != 3:
            raise ValueError("PCQT v1 is defined only for the S3-to-S4 transition")
        if not math.isfinite(float(support_floor)) or support_floor <= 0.0:
            raise ValueError("support_floor must be finite and positive")
        if not math.isfinite(float(normalization_delta)) or normalization_delta <= 0.0:
            raise ValueError("normalization_delta must be finite and positive")
        if not math.isfinite(float(residual_max)) or not 0.0 < residual_max <= 1.0:
            raise ValueError("residual_max must be finite and in (0, 1]")

        self.mode = mode
        self.source_h_channels = values["source_h_channels"]
        self.response_channels = values["response_channels"]
        self.groups = values["groups"]
        self.orientations = values["orientations"]
        self.phases = values["phases"]
        self.coordinate_channels = values["coordinate_channels"]
        self.target_channels = values["target_channels"]
        self.support_floor = float(
            torch.tensor(float(support_floor), dtype=torch.float64).item()
        )
        self.normalization_delta = float(
            torch.tensor(float(normalization_delta), dtype=torch.float32).item()
        )
        self.residual_max = float(
            torch.tensor(float(residual_max), dtype=torch.float32).item()
        )

        self.writer = nn.Conv2d(
            self.coordinate_channels,
            self.target_channels,
            kernel_size=1,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.zeros(()))
        self.last_aux: dict[str, torch.Tensor] = {}

        # These buffers are a fail-closed description of the causal arm, not
        # optimizer settings.  They intentionally remain present with
        # ``strict=False`` loading.
        self.register_buffer(
            "_pcqt_schema_version",
            torch.tensor(_SCHEMA_VERSION, dtype=torch.int64),
        )
        self.register_buffer(
            "_pcqt_mode_code",
            torch.tensor(_MODE_CODES[self.mode], dtype=torch.int64),
        )
        self.register_buffer(
            "_pcqt_source_target_stages",
            torch.tensor((int(source_stage), int(target_stage)), dtype=torch.int64),
        )
        self.register_buffer(
            "_pcqt_dilation_pair",
            torch.tensor(tuple(int(value) for value in dilation_pair), dtype=torch.int64),
        )
        self.register_buffer(
            "_pcqt_pool_spec",
            torch.tensor(tuple(int(value) for value in pool_spec), dtype=torch.int64),
        )
        self.register_buffer(
            "_pcqt_geometry_spec",
            torch.tensor(
                (
                    self.source_h_channels,
                    self.response_channels,
                    self.groups,
                    self.orientations,
                    self.phases,
                    self.coordinate_channels,
                    self.target_channels,
                ),
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "_pcqt_support_floor",
            torch.tensor(self.support_floor, dtype=torch.float64),
        )
        self.register_buffer(
            "_pcqt_normalization_delta",
            torch.tensor(self.normalization_delta, dtype=torch.float32),
        )
        self.register_buffer(
            "_pcqt_residual_max",
            torch.tensor(self.residual_max, dtype=torch.float32),
        )

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        super()._apply(fn, recurse=recurse)
        # The support floor is a serialized numerical definition.  Keep its
        # FP64 value even when a caller moves the rest of the network to BF16.
        self._pcqt_support_floor = self._pcqt_support_floor.double()
        return self

    def initialize_local_parameters(self) -> None:
        """Initialize after base ARCQ in a caller-owned local RNG fork."""

        nn.init.trunc_normal_(self.writer.weight, std=0.02)
        with torch.no_grad():
            self.alpha.zero_()

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
            "schema": self._pcqt_schema_version,
            "mode": self._pcqt_mode_code,
            "source/target stages": self._pcqt_source_target_stages,
            "dilation pair": self._pcqt_dilation_pair,
            "pool specification": self._pcqt_pool_spec,
            "geometry": self._pcqt_geometry_spec,
            "support floor": self._pcqt_support_floor,
            "normalization delta": self._pcqt_normalization_delta,
            "residual maximum": self._pcqt_residual_max,
        }
        names = {
            "schema": "_pcqt_schema_version",
            "mode": "_pcqt_mode_code",
            "source/target stages": "_pcqt_source_target_stages",
            "dilation pair": "_pcqt_dilation_pair",
            "pool specification": "_pcqt_pool_spec",
            "geometry": "_pcqt_geometry_spec",
            "support floor": "_pcqt_support_floor",
            "normalization delta": "_pcqt_normalization_delta",
            "residual maximum": "_pcqt_residual_max",
        }
        missing = [description for description, name in names.items() if prefix + name not in state_dict]
        if missing:
            error_msgs.append(
                "PCQT checkpoint must contain immutable buffers: " + ", ".join(missing)
            )
        for description, expected_value in expected.items():
            supplied = state_dict.get(prefix + names[description])
            if supplied is None:
                continue
            if not isinstance(supplied, torch.Tensor):
                error_msgs.append(f"PCQT checkpoint {description} buffer is not a tensor")
                continue
            if supplied.dtype != expected_value.dtype or supplied.shape != expected_value.shape:
                error_msgs.append(
                    f"PCQT checkpoint {description} buffer dtype/shape does not match "
                    "the instantiated transition"
                )
                continue
            if not torch.equal(supplied.detach().cpu(), expected_value.detach().cpu()):
                error_msgs.append(
                    f"PCQT checkpoint {description} does not match the instantiated "
                    f"mode {self.mode!r}"
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

    def forward(
        self,
        coordinate: torch.Tensor,
        base_state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if coordinate.ndim != 4 or coordinate.shape[1] != self.coordinate_channels:
            raise ValueError(
                f"PCQT coordinate expected Bx{self.coordinate_channels}xHxW, got "
                f"{tuple(coordinate.shape)}"
            )
        if base_state.ndim != 4 or base_state.shape[1] != self.target_channels:
            raise ValueError(
                f"PCQT base state expected Bx{self.target_channels}xHxW, got "
                f"{tuple(base_state.shape)}"
            )
        if coordinate.shape[0] != base_state.shape[0] or coordinate.shape[-2:] != base_state.shape[-2:]:
            raise ValueError("PCQT coordinate and C4 base state must share batch/spatial geometry")

        # The coordinate derives from constrained FP32 response energies.  Keep
        # this writer/normalization in FP32 too, then cast only the final
        # residual back to C4's deployment dtype.
        with torch.autocast(device_type=coordinate.device.type, enabled=False):
            written = self.writer(coordinate.float())
            normalization_delta = self._pcqt_normalization_delta.to(written)
            written_rms = written.square().mean(dim=(1, 2, 3), keepdim=True).add(
                normalization_delta.square()
            ).sqrt()
            normalized = written / written_rms
            base_rms = base_state.detach().float().square().mean(
                dim=(1, 2, 3), keepdim=True
            ).sqrt()
            scale = self._pcqt_residual_max.to(written) * torch.tanh(self.alpha.float())
            residual = scale * base_rms * normalized
            updated = base_state + residual.to(dtype=base_state.dtype)

        with torch.no_grad():
            coordinate_float = coordinate.detach().float()
            residual_rms = residual.detach().square().mean(dim=(1, 2, 3)).sqrt()
            base_rms_scalar = base_rms.detach().reshape(-1)
            aux = {
                "coordinate_mean": coordinate_float.mean(dim=(1, 2, 3)),
                "coordinate_abs_mean": coordinate_float.abs().mean(dim=(1, 2, 3)),
                "coordinate_abs_max": coordinate_float.abs().amax(dim=(1, 2, 3)),
                "writer_rms": written.detach().square().mean(dim=(1, 2, 3)).sqrt(),
                "residual_scale": scale.detach().expand(coordinate.shape[0]),
                "residual_rms": residual_rms,
                "base_rms": base_rms_scalar,
                "residual_to_base_rms_ratio": residual_rms
                / base_rms_scalar.clamp_min(1.0e-12),
            }
        self.last_aux = aux
        return updated, aux
