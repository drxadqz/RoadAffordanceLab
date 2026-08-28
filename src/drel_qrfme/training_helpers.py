from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

"""学习率、EMA、序数损失与验证综合分数等训练辅助函数。"""

from drel_qrfme.rscd_label_factors import RSCDFactorSpec, class_factor_targets


class ModelEMA:
    """Exponential moving average with a complete, strictly loadable model."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        target = self.module.state_dict()
        if source.keys() != target.keys():
            raise RuntimeError("EMA source and target state dictionaries do not match")
        for name, ema_value in target.items():
            source_value = source[name]
            if isinstance(ema_value, torch.Tensor) and isinstance(
                source_value, torch.Tensor
            ):
                value = source_value.detach().to(device=ema_value.device)
                if ema_value.is_floating_point():
                    ema_value.mul_(self.decay).add_(
                        value, alpha=1.0 - self.decay
                    )
                else:
                    ema_value.copy_(value)
                continue
            if isinstance(ema_value, torch.Tensor) != isinstance(
                source_value, torch.Tensor
            ):
                raise RuntimeError(
                    f"EMA state kind mismatch for {name}: "
                    f"{type(ema_value).__name__} vs "
                    f"{type(source_value).__name__}"
                )
            # nn.Module extra state may be a dictionary rather than a tensor.
            # It encodes immutable scientific semantics and must match exactly;
            # averaging or silently replacing it would corrupt the checkpoint.
            if ema_value != source_value:
                raise RuntimeError(f"EMA non-tensor state mismatch for {name}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "module": self.module.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.module.load_state_dict(state["module"], strict=True)


def set_warmup_cosine_lr(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """Set the epoch learning rate using linear warmup and cosine decay."""

    epoch = int(epoch)
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(min(int(warmup_epochs), total_epochs), 0)
    base_lr = float(base_lr)
    min_lr = float(min_lr)
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        lr = base_lr * float(epoch) / float(warmup_epochs)
    elif total_epochs <= warmup_epochs:
        lr = base_lr
    else:
        progress = (float(epoch) - float(warmup_epochs)) / float(
            max(total_epochs - warmup_epochs, 1)
        )
        progress = min(max(progress, 0.0), 1.0)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        scale = float(group.get("lr_scale", 1.0))
        group["lr"] = lr * scale
    return float(lr)


def coral_roughness_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    spec: RSCDFactorSpec,
) -> tuple[torch.Tensor, dict[str, float]]:
    """CORAL loss for the defined RSCD roughness states.

    The factor vocabulary is ``none/smooth/slight/severe``.  ``none`` denotes
    an undefined attribute and is therefore masked.  The remaining three
    ordered categories use two cumulative targets.
    """

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            "RSCD smooth/slight/severe CORAL logits must have shape [B, 2], "
            f"got {tuple(logits.shape)}"
        )
    factors = class_factor_targets(labels, spec, labels.device)
    roughness = factors["roughness"]
    valid = roughness.gt(0)
    if not bool(valid.any()):
        zero = logits.new_zeros(())
        return zero, {
            "loss_roughness_coral": 0.0,
            "acc_roughness_coral": 0.0,
            "roughness_coral_valid": 0.0,
        }
    logits_valid = logits[valid].float()
    ordered_target = roughness[valid] - 1
    levels = torch.arange(2, device=labels.device).view(1, 2)
    cumulative_target = ordered_target.view(-1, 1).gt(levels).to(dtype=logits_valid.dtype)
    loss = F.binary_cross_entropy_with_logits(logits_valid, cumulative_target)
    prediction = logits_valid.gt(0).sum(dim=1)
    accuracy = prediction.eq(ordered_target).float().mean()
    logs = {
        "loss_roughness_coral": float(loss.detach().cpu()),
        "acc_roughness_coral": float(accuracy.detach().cpu()),
        "roughness_coral_valid": float(valid.sum().detach().cpu()),
    }

    friction = factors["friction"]
    material = factors["material"]
    family_indices = {
        "dry_asphalt": (0, 1),
        "dry_concrete": (0, 2),
        "wet_asphalt": (1, 1),
        "wet_concrete": (1, 2),
        "water_asphalt": (2, 1),
        "water_concrete": (2, 2),
    }
    for name, (friction_index, material_index) in family_indices.items():
        family_mask = (
            valid
            & friction.eq(int(friction_index))
            & material.eq(int(material_index))
        )
        count = int(family_mask.sum().detach().cpu())
        logs[f"roughness_coral_valid_{name}"] = float(count)
        if count == 0:
            logs[f"loss_roughness_coral_{name}"] = 0.0
            logs[f"acc_roughness_coral_{name}"] = 0.0
            continue
        family_logits = logits[family_mask].float()
        family_target = roughness[family_mask] - 1
        family_cumulative_target = family_target.view(-1, 1).gt(levels).to(
            dtype=family_logits.dtype
        )
        family_loss = F.binary_cross_entropy_with_logits(
            family_logits,
            family_cumulative_target,
        )
        family_prediction = family_logits.gt(0).sum(dim=1)
        family_accuracy = family_prediction.eq(family_target).float().mean()
        logs[f"loss_roughness_coral_{name}"] = float(
            family_loss.detach().cpu()
        )
        logs[f"acc_roughness_coral_{name}"] = float(
            family_accuracy.detach().cpu()
        )

    # Boundary diagnostics use the corresponding cumulative logit directly.
    # They answer whether the head orders the adjacent pair, without allowing
    # the third roughness state to dominate the statistic.
    boundary_specs = {
        "smooth_slight": (1, 2, 0),
        "slight_severe": (2, 3, 1),
    }
    for name, (left, right, logit_index) in boundary_specs.items():
        boundary_mask = roughness.eq(left) | roughness.eq(right)
        count = int(boundary_mask.sum().detach().cpu())
        logs[f"roughness_coral_valid_{name}_boundary"] = float(count)
        if count == 0:
            logs[f"acc_roughness_coral_{name}_boundary"] = 0.0
            continue
        boundary_target = roughness[boundary_mask].eq(right)
        boundary_prediction = logits[boundary_mask, logit_index].gt(0)
        boundary_accuracy = boundary_prediction.eq(boundary_target).float().mean()
        logs[f"acc_roughness_coral_{name}_boundary"] = float(
            boundary_accuracy.detach().cpu()
        )

    return loss.to(dtype=logits.dtype), logs


def weighted_validation_score(
    summary: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compute the configurable ARCQ validation-only checkpoint score."""

    weights = weights or {
        "top1": 0.4,
        "macro_f1": 0.4,
        "bottom5_mean_f1": 0.2,
    }
    total_weight = sum(max(float(value), 0.0) for value in weights.values())
    if total_weight <= 0.0:
        raise ValueError("validation score weights must contain a positive value")
    score = 0.0
    for key, weight in weights.items():
        score += max(float(weight), 0.0) * float(summary.get(str(key), 0.0))
    return float(score / total_weight)
