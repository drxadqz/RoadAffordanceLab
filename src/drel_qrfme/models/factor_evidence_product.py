"""把 RSCD 三个结构化因素的证据映射回合法的 27 类输出。

这是一个有意保持很小的性能组件。它不假设三个 RSCD 因素统计独立；每个因素头
只提供相对于封闭 RSCD 类别体系先验的“对数证据比”。这些证据比在概率空间相乘，
再作为一个较弱的残差叠加到不受约束的 27 类 logits 上；主分类头因此仍能保留
真实类别先验和因素之间的条件依赖关系。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FactorEvidenceProductResidual(nn.Module):
    """把紧凑因素 logits 映射成合法的 27 类证据残差。

    material 和 roughness 的状态 ``0`` 表示该因素在语义上不适用。这些轴应贡献
    乘法单位元（对数空间中的 0），不能把“不适用”误当成虚构类别或硬路由。
    """

    def __init__(self, class_to_factor: Tensor) -> None:
        super().__init__()
        mapping = torch.as_tensor(class_to_factor, dtype=torch.long)
        if mapping.ndim != 2 or mapping.shape[1] != 3:
            raise ValueError("class_to_factor must have shape Kx3")
        if bool((mapping[:, 0] < 0).any()):
            raise ValueError("friction must be defined for every class")
        if bool((mapping[:, 1:] < 0).any()):
            raise ValueError("undefined factors must use the explicit state 0")
        self.register_buffer("class_to_factor", mapping.clone())

        # 以下查表张量只依赖固定的 27 类定义，因此在 CPU 上一次性建立并检查，
        # 避免每次 forward 都把 CUDA 张量缩减成 Python 数值。它们属于可推导缓存，
        # 设置 persistent=False 后不会写入 checkpoint，从而兼容旧版严格加载。
        self.register_buffer(
            "friction_class_index",
            mapping[:, 0].clone(),
            persistent=False,
        )
        material_state = mapping[:, 1]
        roughness_state = mapping[:, 2]
        self.register_buffer(
            "material_class_index",
            (material_state - 1).clamp_min(0),
            persistent=False,
        )
        self.register_buffer(
            "roughness_class_index",
            (roughness_state - 1).clamp_min(0),
            persistent=False,
        )
        self.register_buffer(
            "material_defined_mask",
            material_state > 0,
            persistent=False,
        )
        self.register_buffer(
            "roughness_defined_mask",
            roughness_state > 0,
            persistent=False,
        )

        friction_prior = torch.bincount(
            mapping[:, 0],
            minlength=int(mapping[:, 0].max()) + 1,
        ).float()
        material_defined = mapping[:, 1] > 0
        roughness_defined = mapping[:, 2] > 0
        if not bool(material_defined.any()) or not bool(roughness_defined.any()):
            raise ValueError("material and roughness need observable legal classes")
        material_prior = torch.bincount(
            mapping[material_defined, 1] - 1,
            minlength=int(mapping[:, 1].max()),
        ).float()
        roughness_prior = torch.bincount(
            mapping[roughness_defined, 2] - 1,
            minlength=int(mapping[:, 2].max()),
        ).float()
        for name, counts in {
            "friction": friction_prior,
            "material": material_prior,
            "roughness": roughness_prior,
        }.items():
            if bool((counts <= 0).any()):
                raise ValueError(f"{name}: every compact state must be occupied")
            probability = counts / counts.sum()
            self.register_buffer(f"prior_logits__{name}", probability.log())

    @staticmethod
    def _relative_log_evidence(logits: Tensor, prior_logits: Tensor) -> Tensor:
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("factor logits must have shape BxK with K>=2")
        if prior_logits.ndim != 1 or prior_logits.shape[0] != logits.shape[1]:
            raise ValueError("factor prior width must match factor logits")
        # AMP 下把唯一较脆弱的运算强制放到 FP32。两项使用相同 log-softmax，
        # 使“只输出先验”的因素头得到浮点意义上的精确 0，而不是近似 0。
        return F.log_softmax(logits.float(), dim=1) - F.log_softmax(
            prior_logits.float(),
            dim=0,
        )

    def initialize_factor_heads(self, heads: nn.ModuleDict) -> None:
        if set(heads) != {"friction", "material", "roughness"}:
            raise ValueError("compact factor-head set is incomplete")
        for name, head in heads.items():
            if not isinstance(head, nn.Linear):
                raise TypeError("compact factor heads must be nn.Linear modules")
            prior_logits = getattr(self, f"prior_logits__{name}")
            if head.out_features != prior_logits.numel():
                raise ValueError(f"{name}: factor-head width disagrees with ontology")
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.copy_(prior_logits)

    def forward(self, factor_logits: dict[str, Tensor]) -> Tensor:
        required = {"friction", "material", "roughness"}
        if set(factor_logits) != required:
            raise ValueError(
                "factor_logits must contain exactly friction, material, roughness"
            )
        friction = self._relative_log_evidence(
            factor_logits["friction"], self.prior_logits__friction
        )
        material = self._relative_log_evidence(
            factor_logits["material"], self.prior_logits__material
        )
        roughness = self._relative_log_evidence(
            factor_logits["roughness"], self.prior_logits__roughness
        )
        batch = friction.shape[0]
        if material.shape[0] != batch or roughness.shape[0] != batch:
            raise ValueError("factor-logit batch sizes must agree")

        joint = friction.index_select(1, self.friction_class_index)
        material_value = material.index_select(1, self.material_class_index)
        joint = joint + material_value * self.material_defined_mask
        roughness_value = roughness.index_select(1, self.roughness_class_index)
        joint = joint + roughness_value * self.roughness_defined_mask

        # 减去每个样本的均值用于固定规范（gauge fixing）：既便于跨样本比较诊断量，
        # 也避免给主 27 类线性 logits 加上对所有类别相同、毫无作用的偏移。
        return joint - joint.mean(dim=1, keepdim=True)
