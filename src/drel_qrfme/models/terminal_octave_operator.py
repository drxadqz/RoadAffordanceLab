"""ARCQ Terminal-Octave Consolidation (TOC).

The frozen ARCQ/RSPNet stage-wise audit localized the reproducible spatial
organization gap to ARCQ's last appearance transition.  A follow-up read-only
audit (``docs/CLAUDE_READONLY_AUDIT_20260729.md``) measured *why* that stage
looked different: ARCQ and RSPNet do not share a spatial schedule.

On the frozen ``360 x 240`` D350 support the two backbones run at::

    ARCQ    A1/C1 180x120   A2/C2 90x60   A3/C3 45x30   A4/C4 23x15  (/16)
    RSPNet  R1     90x60    R2    45x30   R3    23x15   R4    12x8   (/32)

so ``A_k`` is scale-matched to ``R_{k-1}``, and ARCQ simply has no stage at
RSPNet's terminal octave.  Re-pairing the same frozen contribution maps on a
common grid shrinks the registered ``A4`` gap from ``+0.12 / -0.19 / +0.23`` to
``+0.08 / -0.08 / +0.09``, which no longer satisfies that gate's own two-of-three
major-gap predicate.

Because global average pooling is linear, area-pooling ``A4`` from 23x15 to
12x8 before the mean is *exactly* the same function as pooling directly.  The
missing ingredient therefore cannot be pooling: it must be a learned,
non-linear consolidation evaluated at the coarser octave.  ``LBET`` already
falsified the competing "the terminal kernel oscillates" story by improving all
three A4 topology statistics while losing Top-1, Macro-F1 and the composite
score.

TOC tests exactly one structural variable: **at which spatial scale ARCQ
performs its last evidence consolidation**.  Both active arms append the same
consolidation unit to the protected ``C`` stream and to the appearance ``A``
stream; they differ only in the terminal transition stride.

===================  ========  ==================================
mode                 stride    terminal grid on 360x240 input
===================  ========  ==================================
same_scale_control   1         23 x 15  (unchanged ARCQ octave)
terminal_octave      2         12 x  8  (RSPNet's terminal octave)
===================  ========  ==================================

The two arms own identical parameter tensors: neither ``nn.Conv2d`` nor the
reused ARCQ blocks consume stride-dependent randomness, so a shared forked RNG
seed makes them bit-identical at initialization.  Consequently a difference in
validation cannot be attributed to capacity, depth, operator family or
initialization -- only to the octave at which consolidation happens.

Note that the control arm is the *more* expensive one: it evaluates the same
blocks on a 23x15 grid rather than 12x8, roughly 3.6x the spatial positions.
A win for the candidate therefore cannot be explained by extra compute either.

This module deliberately imports nothing from ``arcq_road``.  The ARCQ blocks
and normalization are injected as factories so that ``arcq_road`` can own the
only import edge and no circular import is created.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F


__all__ = [
    "TOC_DISABLED",
    "TOC_MODE_CODES",
    "TOC_MODE_STRIDES",
    "TOC_SAME_SCALE_CONTROL",
    "TOC_SCHEMA_VERSION",
    "TOC_TERMINAL_OCTAVE",
    "TOC_VALID_MODES",
    "TerminalOctaveConsolidation",
    "TerminalOctavePair",
]


TOC_DISABLED = "disabled"
TOC_SAME_SCALE_CONTROL = "same_scale_control"
TOC_TERMINAL_OCTAVE = "terminal_octave"

TOC_MODE_CODES = {
    TOC_SAME_SCALE_CONTROL: 1,
    TOC_TERMINAL_OCTAVE: 2,
}
TOC_MODE_STRIDES = {
    TOC_SAME_SCALE_CONTROL: 1,
    TOC_TERMINAL_OCTAVE: 2,
}
TOC_VALID_MODES = frozenset({TOC_DISABLED, *TOC_MODE_CODES})
TOC_SCHEMA_VERSION = 1


def _reflect_pad(x: torch.Tensor, padding: int) -> torch.Tensor:
    """Reflect padding with ARCQ's exact degenerate-size fallback.

    This intentionally mirrors ``arcq_road._reflect_pad`` rather than importing
    it, so that ``arcq_road`` owns the single import edge between the two
    modules.  The release smoke test asserts the two implementations agree
    bit-for-bit, so the duplication cannot silently drift.
    """

    padding = int(padding)
    if padding <= 0:
        return x
    if x.shape[-2] <= padding or x.shape[-1] <= padding:
        return F.pad(x, (padding, padding, padding, padding), mode="replicate")
    return F.pad(x, (padding, padding, padding, padding), mode="reflect")


class TerminalOctaveConsolidation(nn.Module):
    """One consolidation unit appended to a single ARCQ evidence stream.

    The unit is deliberately the ordinary ARCQ stage pattern -- a normalized
    strided depthwise transition, a pointwise projection, then ``depth`` copies
    of that stream's own residual block -- so the experiment compares *where*
    ARCQ consolidates rather than introducing a new operator family.

    ``in_channels`` equals ``out_channels``, hence every downstream ARCQ module
    (readout interaction, fusion projection, output norm, classifier head)
    keeps an identical shape and parameter count across all three arms.
    """

    def __init__(
        self,
        channels: int,
        *,
        mode: str,
        depth: int,
        norm_factory: Callable[[int], nn.Module],
        block_factory: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError("channels must be positive")
        mode = str(mode).strip().lower()
        if mode not in TOC_MODE_CODES:
            raise ValueError(
                "active TOC mode must be one of "
                f"{sorted(TOC_MODE_CODES)}, got {mode!r}"
            )
        depth = int(depth)
        if depth <= 0:
            raise ValueError("TOC depth must be a positive integer")

        self.channels = channels
        self.mode = mode
        self.depth = depth
        self.stride = TOC_MODE_STRIDES[mode]

        self.norm = norm_factory(channels)
        # Both arms store exactly this parameter topology.  Conv2d
        # initialization does not read ``stride``, so the two arms are
        # bit-identical under a shared RNG seed.
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=self.stride,
            groups=channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.blocks = nn.Sequential(
            *[block_factory(channels) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"TerminalOctaveConsolidation expected Bx{self.channels}xHxW, "
                f"got {tuple(x.shape)}"
            )
        z = self.depthwise(_reflect_pad(self.norm(x), 1))
        return self.blocks(self.pointwise(z))


class TerminalOctavePair(nn.Module):
    """Independent consolidation units for the protected C and appearance A streams.

    The two units never exchange tensors.  In particular the composition unit
    reads only ``composition_state``, so ARCQ's structural absence of an
    ``A -> C`` edge is preserved exactly: no appearance activation and no
    appearance parameter gradient can reach the protected stream through TOC.
    """

    def __init__(
        self,
        channels: int,
        *,
        mode: str,
        depth: int,
        norm_factory: Callable[[int], nn.Module],
        composition_block_factory: Callable[[int], nn.Module],
        appearance_block_factory: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        # Construction order is fixed so that both active arms consume the
        # forked RNG stream identically.
        self.composition = TerminalOctaveConsolidation(
            channels,
            mode=mode,
            depth=depth,
            norm_factory=norm_factory,
            block_factory=composition_block_factory,
        )
        self.appearance = TerminalOctaveConsolidation(
            channels,
            mode=mode,
            depth=depth,
            norm_factory=norm_factory,
            block_factory=appearance_block_factory,
        )
        self.channels = int(channels)
        self.mode = self.composition.mode
        self.depth = self.composition.depth
        self.stride = self.composition.stride

    def forward(
        self,
        composition_state: torch.Tensor,
        appearance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.composition(composition_state),
            self.appearance(appearance),
        )

    @torch.no_grad()
    def summaries(
        self,
        composition_state: torch.Tensor,
        appearance: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return frozen per-sample diagnostics for the gate feature audit."""

        def stats(tensor: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
            work = tensor.detach().float()
            return {
                f"{prefix}_rms": work.square().mean(dim=(1, 2, 3)).sqrt(),
                f"{prefix}_abs_max": work.abs().amax(dim=(1, 2, 3)),
                f"{prefix}_channel_std": work.std(dim=1).mean(dim=(1, 2)),
                f"{prefix}_height": work.new_full(
                    (work.shape[0],),
                    float(work.shape[2]),
                ),
                f"{prefix}_width": work.new_full(
                    (work.shape[0],),
                    float(work.shape[3]),
                ),
            }

        summary = stats(composition_state, "toc_composition")
        summary.update(stats(appearance, "toc_appearance"))
        return summary
