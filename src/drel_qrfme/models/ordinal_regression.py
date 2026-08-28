from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class CoralOrdinalHead(nn.Module):
    """Rank-consistent CORAL head with shared weights and ordered thresholds.

    For ``num_classes`` ordered categories, the head returns
    ``num_classes - 1`` cumulative logits.  A single learned score is compared
    with monotonically increasing thresholds, so the cumulative probabilities
    cannot cross by construction.
    """

    def __init__(self, in_features: int, num_classes: int = 3) -> None:
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("CORAL requires at least two ordered classes")
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.score = nn.Linear(self.in_features, 1, bias=False)
        self.base_threshold = nn.Parameter(torch.tensor(-0.5))
        gap_count = self.num_classes - 2
        if gap_count > 0:
            initial_gap = 1.0
            inverse_softplus = math.log(math.expm1(initial_gap))
            self.raw_threshold_gaps = nn.Parameter(
                torch.full((gap_count,), float(inverse_softplus))
            )
        else:
            self.register_parameter("raw_threshold_gaps", None)

    def thresholds(self) -> torch.Tensor:
        base = self.base_threshold.view(1)
        if self.raw_threshold_gaps is None:
            return base
        gaps = F.softplus(self.raw_threshold_gaps) + 1e-4
        return torch.cat((base, base + torch.cumsum(gaps, dim=0)), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.score(x)
        return score - self.thresholds().view(1, -1)

