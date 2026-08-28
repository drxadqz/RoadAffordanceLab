"""DREL-QRFME 最终网络定义（建议从本文件开始阅读算法）。

读代码顺序
----------
1. :class:`RoadContextBlock`：语义主干的基本残差块；
2. :class:`NormBudgetWriter`：把物理响应写回语义流，并限制写入幅度；
3. :class:`FactorMeasureStage`：用可靠性与语义 query 形成空间概率测度；
4. :class:`DRELRTSurfaceClassifier.__init__`：组装四阶段网络；
5. :class:`DRELRTSurfaceClassifier.forward`：完整的前向数据流。

主要张量记号
------------
``image`` 为 ``[B,3,H,W]``；``semantic`` 是语义特征；``ledger`` 是方向
响应账本；``composition`` 是方向组成坐标；``radial`` 是径向/可靠性坐标；
最终 ``logits`` 为 ``[B,27]``。RSCD 的每一类还被分解成 friction、material、
roughness 三个因素，RFME 先估计因素残差，再按类别的因素组合还原为 27 类残差。

续训兼容警告
------------
不要改模块名、参数名、通道数、层数或 forward 的数值顺序；这些内容共同决定
checkpoint 的 state_dict 键和优化轨迹。研究新结构时应复制配置并从 Epoch 1 开始。

本文件有意独立于历史 C3/S7 计算图：只包含一个自定义语义主干、一个受保护的
方向/径向证据账本，以及可选的论文对照读出。这里不接受预训练权重、教师模型、
特定类别修正规则或辅助损失。

最终 ``rfme`` 分支把 RFCR 的有效部分直接合并进 RFME，而不是堆叠两个注意力
模块：语义因素 query 提供“与当前任务是否相关”的分数，再与可靠性一起形成
归一化空间测度，最后在该测度下估计各 codeword 的条件残差矩。
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

"""DREL-QRFME 最终模型。

模块关系：语义主干提取全局路面语义；方向响应算子提取局部纹理证据；
可靠性估计把不可信位置降权；Query-conditioned RFME 对每个路面因子形成
条件残差统计；NormBudgetWriter 只允许有限幅度写回语义流，避免物理证据
破坏稳定语义表示。

续训时不要修改通道数、深度、方向数、RFME 维度或 write ratio。任何结构改动
都会使 Epoch 50 checkpoint 不再兼容。若要做消融，请复制新配置并从头训练。
"""

from .directional_response_operators import DropPath, MatchedResponseConv2d, RadialCompositionQuotient
from drel_qrfme.rscd_label_factors import (
    FACTOR_AXES,
    FACTOR_LABELS,
    build_rscd_factor_spec,
)


_VARIANTS = {
    "zero_write": 0,
    "drel_rt": 1,
    "uniform_rfme": 2,
    "rfme": 3,
    "readonly_rfme": 4,
}


def _four(values: tuple[int, ...] | list[int], name: str) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 4 or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain four positive integers")
    return result  # type: ignore[return-value]


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _mean_pool_to(x: torch.Tensor, target: tuple[int, int]) -> torch.Tensor:
    """用确定性的精确平均池化，把张量对齐到固定金字塔分辨率。"""

    source = tuple(int(value) for value in x.shape[-2:])
    if source == target:
        return x
    if source[0] % target[0] or source[1] % target[1]:
        raise ValueError(
            "DREL-RT pyramid requires integer mean-pooling ratios: "
            f"source={source} target={target}"
        )
    kernel = (source[0] // target[0], source[1] // target[1])
    return F.avg_pool2d(x, kernel_size=kernel, stride=kernel)


class RoadContextBlock(nn.Module):
    """自定义局部/中尺度残差块；它是特征载体，不作为论文核心创新点。"""

    def __init__(self, channels: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        channels = int(channels)
        hidden = 2 * channels
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.expand = nn.Conv2d(channels, 4 * channels, 1)
        self.local = nn.Conv2d(
            hidden, hidden, 3, padding=1, groups=hidden, bias=False
        )
        self.context = nn.Conv2d(
            hidden, hidden, 7, padding=3, groups=hidden, bias=False
        )
        self.project = nn.Conv2d(4 * channels, channels, 1)
        self.layer_scale = nn.Parameter(torch.full((channels,), 1.0e-6))
        self.drop_path = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.expand(self.norm(x)).chunk(2, dim=1)
        update = torch.cat((self.local(left), self.context(right)), dim=1)
        update = self.project(F.silu(update))
        update = update * self.layer_scale[None, :, None, None]
        return x + self.drop_path(update)


class EvidenceRefiner(nn.Module):
    """用轻量 depthwise/pointwise 残差块清理每一级方向响应。"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.scale = nn.Parameter(torch.full((channels,), 1.0e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.pointwise(F.silu(self.depthwise(self.norm(x))))
        return x + update * self.scale[None, :, None, None]


class NormBudgetWriter(nn.Module):
    """将方向证据写入语义特征，并对每个样本施加显式范数上限。

    ``candidate`` 是待写入证据，``budget`` 是允许写入量占语义范数的比例。
    返回值 ``actual`` 是实际写入范数比例，供训练日志审计。卷积权重从零开始，
    因而 Epoch 0 不会突然破坏语义主干。
    """

    def __init__(
        self,
        composition_channels: int,
        radial_channels: int,
        semantic_channels: int,
    ) -> None:
        super().__init__()
        self.direction = nn.Conv2d(composition_channels, semantic_channels, 1, bias=False)
        self.radial = nn.Conv2d(radial_channels, semantic_channels, 1, bias=False)
        nn.init.zeros_(self.direction.weight)
        nn.init.zeros_(self.radial.weight)

    def forward(
        self,
        semantic: torch.Tensor,
        composition: torch.Tensor,
        radial: torch.Tensor,
        reliability: torch.Tensor,
        budget: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) 三种流必须落在同一空间分辨率后才能逐位置融合。
        target = tuple(int(value) for value in semantic.shape[-2:])
        composition = _mean_pool_to(composition, target)
        radial = _mean_pool_to(radial, target)
        reliability = _mean_pool_to(reliability, target).mean(1, keepdim=True)
        # 2) 方向组成和径向证据投影到语义通道；可靠性低的位置自动被压小。
        candidate = torch.tanh(
            self.direction(composition) + self.radial(radial)
        ) * reliability
        # 两个投影卷积故意从零初始化，使所有消融分支在 Epoch 0 的 logits 完全相同。
        # ``sqrt(sum(x**2))`` 在精确零点的导数没有定义，因此必须在开平方之前加
        # epsilon，保证第一次 BF16 反向传播仍为有限数。
        semantic_norm = semantic.float().square().sum((1, 2, 3), keepdim=True).add(1.0e-12).sqrt()
        candidate_norm = candidate.float().square().sum((1, 2, 3), keepdim=True).add(1.0e-12).sqrt()
        ratio = budget.reshape(1, 1, 1, 1).to(candidate)
        # 3) scale<=1，确保 ||update|| / ||semantic|| 不超过 budget。
        scale = torch.minimum(
            torch.ones_like(candidate_norm),
            ratio * semantic_norm / candidate_norm.clamp_min(1.0e-6),
        ).to(candidate)
        update = scale * candidate
        actual = update.float().square().sum((1, 2, 3)).sqrt() / semantic_norm.flatten().clamp_min(1.0e-6)
        return semantic + update, actual


class FactorMeasureStage(nn.Module):
    """单级 Query-conditioned RFME（可靠性因素测度编码器）。

    直观理解：每个空间位置有三种信息——语义、方向证据和可靠性。全局语义
    生成 query，局部融合特征生成 key；query-key 相关性与可靠性共同决定该
    位置在空间测度中的权重。随后以可学习 codeword 做软分配，得到条件残差矩。
    """

    def __init__(
        self,
        semantic_channels: int,
        evidence_channels: int,
        reliability_groups: int,
        codewords: int,
        *,
        dim: int = 64,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.codewords_count = int(codewords)
        self.semantic = nn.Conv2d(semantic_channels, dim, 1, bias=False)
        self.evidence = nn.Conv2d(evidence_channels, dim, 1, bias=False)
        self.local_key = nn.Conv2d(dim, dim, 1, bias=False)
        self.global_query = nn.Linear(semantic_channels, dim, bias=False)
        self.reliability_mix = nn.Parameter(torch.zeros(reliability_groups))
        self.codewords = nn.Parameter(torch.empty(codewords, dim))
        self.log_scale = nn.Parameter(torch.zeros(codewords))
        nn.init.trunc_normal_(self.codewords, std=0.02)

    def forward(
        self,
        semantic: torch.Tensor,
        evidence: torch.Tensor,
        reliability: torch.Tensor,
        *,
        use_reliable_measure: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = tuple(int(value) for value in semantic.shape[-2:])
        evidence = _mean_pool_to(evidence, target)
        reliability = _mean_pool_to(reliability, target)
        # 把语义流和证据流投影到共同的 rfme_dim 维空间。
        z = self.semantic(semantic) + self.evidence(evidence)
        batch, dim, height, width = z.shape
        descriptor = z.flatten(2).transpose(1, 2)
        if use_reliable_measure:
            # reliability_mix 学习不同方向组的重要性；softmax 保证权重和为 1。
            mix = self.reliability_mix.softmax(dim=0)
            local_reliability = torch.einsum("bghw,g->bhw", reliability, mix)
            local_reliability = local_reliability.flatten(1).clamp(1.0e-6, 1.0)
            query = self.global_query(semantic.mean((2, 3)))
            key = self.local_key(z).flatten(2).transpose(1, 2)
            relevance = torch.einsum("bd,bnd->bn", query, key) / math.sqrt(float(dim))
            # 空间测度同时偏好“可观测/可靠”和“与当前语义 query 相关”的位置。
            measure = F.softmax(local_reliability.log() + relevance, dim=1)
        else:
            measure = descriptor.new_full(
                (batch, height * width), 1.0 / float(height * width)
            )

        # 可靠性必须改变“空间测度”；不能给 codeword softmax 的所有项加同一个
        # 偏移量，因为相同偏移会在 softmax 中被数学抵消，实际不起任何作用。
        # 对每个局部 descriptor 与每个因素 codeword 计算残差及软归属概率。
        diff = descriptor[:, :, None, :] - self.codewords[None, None, :, :]
        distance = diff.float().square().sum(dim=-1)
        scale = F.softplus(self.log_scale.float()) + 1.0e-4
        assignment = F.softmax(-distance * scale[None, None, :], dim=2).to(diff)
        joint = measure[:, :, None].to(assignment) * assignment
        denominator = joint.sum(dim=1).clamp_min(1.0e-6)
        residual = torch.einsum("bnk,bnkd->bkd", joint, diff) / denominator[:, :, None]
        diagnostics = {
            "measure_sum": measure.detach().float().sum(dim=1),
            "measure_max": measure.detach().float().amax(dim=1),
            "assignment_sum": assignment.detach().float().sum(dim=2).mean(dim=1),
            "conditional_den_min": denominator.detach().float().amin(dim=1),
        }
        return residual.flatten(1), diagnostics


class DRELRTSurfaceClassifier(nn.Module):
    """最终四阶段 DREL 主干 + Query-conditioned RFME 的 27 类分类器。

    正式实验 ``variant='rfme'``。其他 variant 仅用于消融：``zero_write``
    关闭证据写入，``uniform_rfme`` 使用均匀空间测度，``readonly_rfme`` 允许
    RFME 读取证据但不写入语义流。不要在继续正式训练时切换 variant。
    """

    head_type = "linear27_plus_zero_start_factor_measure"
    architecture_version = "drel_rt_rfme_v1"

    def __init__(
        self,
        class_to_idx: dict[str, int],
        *,
        variant: str = "rfme",
        pretrained: bool = False,
        semantic_channels: tuple[int, ...] | list[int] = (96, 192, 384, 640),
        semantic_depths: tuple[int, ...] | list[int] = (2, 3, 9, 3),
        ledger_channels: tuple[int, ...] | list[int] = (24, 32, 48, 64),
        response_kernel_sizes: tuple[int, ...] | list[int] = (7, 5, 3, 3),
        orientations: int = 4,
        drop_path_rate: float = 0.10,
        max_write_ratio: float = 0.05,
        initial_write_ratio: float = 0.012,
        rfme_dim: int = 64,
        rfme_rank: int = 128,
        head_init_seed: int = 970027,
    ) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("DREL-RT is scratch-only and rejects pretrained weights")
        variant = str(variant).strip().lower()
        if variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {sorted(_VARIANTS)}")
        self.variant = variant
        # ``readonly_rfme`` 是 v2 唯一的结构干预：DREL 证据仍可参与查询条件概率
        # 测度，但不能写回语义主干。这样可在不增加参数、损失、类别规则或可调
        # 系数的条件下，区分“读取证据”和“用证据改变语义”两种作用。
        self.architecture_version = (
            "drel_rt_readonly_rfme_v2"
            if variant == "readonly_rfme"
            else "drel_rt_rfme_v1"
        )
        self.spec = build_rscd_factor_spec(class_to_idx)
        channels = _four(semantic_channels, "semantic_channels")
        depths = _four(semantic_depths, "semantic_depths")
        ledger_widths = _four(ledger_channels, "ledger_channels")
        kernels = _four(response_kernel_sizes, "response_kernel_sizes")
        orientations = int(orientations)
        if any(width % (2 * orientations) for width in ledger_widths):
            raise ValueError("ledger channels must be divisible by 2*orientations")
        if not 0.0 < initial_write_ratio < max_write_ratio <= 0.10:
            raise ValueError("write ratios must satisfy 0 < initial < maximum <= 0.10")

        self.register_buffer("_variant_code", torch.tensor(_VARIANTS[variant], dtype=torch.int64))
        self.register_buffer("_schema", torch.tensor(1, dtype=torch.int64))
        self.register_buffer("class_to_factor", self.spec.class_to_factor.long().clone())
        self.max_write_ratio = float(max_write_ratio)

        # -------- A. 语义主干：4x 下采样 stem + 四阶段道路上下文块 --------
        self.semantic_stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 4, stride=4, bias=False),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
        )
        self.semantic_transitions = nn.ModuleList(
            nn.Sequential(
                nn.GroupNorm(_group_count(channels[i - 1]), channels[i - 1]),
                nn.Conv2d(channels[i - 1], channels[i], 3, stride=2, padding=1, bias=False),
            )
            for i in range(1, 4)
        )
        rates = torch.linspace(0.0, float(drop_path_rate), sum(depths)).tolist()
        cursor = 0
        stages: list[nn.Module] = []
        for width, depth in zip(channels, depths, strict=True):
            blocks = []
            for _ in range(depth):
                blocks.append(RoadContextBlock(width, drop_path=float(rates[cursor])))
                cursor += 1
            stages.append(nn.Sequential(*blocks))
        self.semantic_stages = nn.ModuleList(stages)
        self.final_norm = nn.GroupNorm(_group_count(channels[-1]), channels[-1])

        # -------- B. DREL 证据账本：方向滤波→清理→组成/径向分解→受限写回 --------
        response_banks: list[nn.Module] = []
        refiners: list[nn.Module] = []
        quotients: list[nn.Module] = []
        writers: list[nn.Module] = []
        previous = 3
        for ledger_width, semantic_width, kernel in zip(
            ledger_widths, channels, kernels, strict=True
        ):
            response_banks.append(
                MatchedResponseConv2d(
                    previous, ledger_width, kernel, stride=2, orientations=orientations
                )
            )
            refiners.append(EvidenceRefiner(ledger_width))
            quotient = RadialCompositionQuotient(
                ledger_width,
                group_width=2 * orientations,
                orientations=orientations,
                smoothing=0.05,
            )
            quotients.append(quotient)
            writers.append(
                NormBudgetWriter(
                    quotient.composition_channels,
                    2 * quotient.num_groups,
                    semantic_width,
                )
            )
            previous = ledger_width
        self.response_banks = nn.ModuleList(response_banks)
        self.evidence_refiners = nn.ModuleList(refiners)
        self.quotients = nn.ModuleList(quotients)
        self.writers = nn.ModuleList(writers)

        initial_fraction = initial_write_ratio / max_write_ratio
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        self.write_budget_logits = nn.Parameter(torch.full((4,), initial_logit))

        # -------- C. RFME：只读取第 2、3 阶段，兼顾空间细节与高层语义 --------
        # 只读第 2、3 阶段：既保留足够空间细节，又避开第 1 阶段的高成本/噪声，
        # 以及最后 7x7 特征图过于粗糙的问题。
        selected = (1, 2)
        self.rfme_stages = selected
        encoders: dict[str, nn.ModuleList] = {}
        reducers: dict[str, nn.Module] = {}
        semantic_readers: dict[str, nn.Module] = {}
        factor_heads: dict[str, nn.Module] = {}
        for axis in FACTOR_AXES:
            count = len(FACTOR_LABELS[axis])
            per_axis = []
            for stage_index in selected:
                quotient = self.quotients[stage_index]
                evidence_channels = quotient.composition_channels + 2 * quotient.num_groups
                per_axis.append(
                    FactorMeasureStage(
                        channels[stage_index],
                        evidence_channels,
                        quotient.num_groups,
                        count,
                        dim=rfme_dim,
                    )
                )
            encoders[axis] = nn.ModuleList(per_axis)
            reducers[axis] = nn.Sequential(
                nn.LayerNorm(2 * count * rfme_dim),
                nn.Linear(2 * count * rfme_dim, rfme_rank),
                nn.GELU(),
            )
            semantic_readers[axis] = nn.Sequential(
                nn.LayerNorm(channels[-1]), nn.Linear(channels[-1], rfme_rank), nn.GELU()
            )
            factor_heads[axis] = nn.Linear(rfme_rank, count)
        self.rfme_encoders = nn.ModuleDict(encoders)
        self.rfme_reducers = nn.ModuleDict(reducers)
        self.rfme_semantic = nn.ModuleDict(semantic_readers)
        self.rfme_heads = nn.ModuleDict(factor_heads)
        self.rfme_gate_logit = nn.Parameter(torch.zeros(()))

        # -------- D. 基础 27 类线性头；RFME 输出作为零起点残差叠加 --------
        self.head = nn.Linear(channels[-1], self.spec.num_classes)
        generator = torch.Generator(device=self.head.weight.device)
        generator.manual_seed(int(head_init_seed))
        nn.init.kaiming_uniform_(self.head.weight, a=math.sqrt(5), generator=generator)
        bound = 1.0 / math.sqrt(channels[-1])
        nn.init.uniform_(self.head.bias, -bound, bound, generator=generator)
        self.last_factor_logits: dict[str, torch.Tensor] = {}
        self.last_aux: dict[str, Any] = {}

    @property
    def write_budgets(self) -> torch.Tensor:
        return self.max_write_ratio * torch.sigmoid(self.write_budget_logits)

    def _factor_residual(
        self,
        final_semantic: torch.Tensor,
        stage_semantics: list[torch.Tensor],
        stage_evidence: list[torch.Tensor],
        stage_reliability: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """把多尺度 RFME 因素输出转换成每个样本的 27 类 logit 残差。"""
        pooled = final_semantic.mean((2, 3))
        reliable = self.variant in {"rfme", "readonly_rfme"}
        factor_logits: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, torch.Tensor] = {}
        for axis in FACTOR_AXES:
            # 同一因素分别编码第 2/3 阶段，再融合纹理矩与最终全局语义。
            moments = []
            for local_index, stage_index in enumerate(self.rfme_stages):
                moment, diag = self.rfme_encoders[axis][local_index](
                    stage_semantics[stage_index],
                    stage_evidence[stage_index],
                    stage_reliability[stage_index],
                    use_reliable_measure=reliable,
                )
                moments.append(moment)
                for name, value in diag.items():
                    diagnostics[f"rfme_{axis}_s{stage_index + 1}_{name}"] = value
            texture = self.rfme_reducers[axis](torch.cat(moments, dim=1))
            semantic = self.rfme_semantic[axis](pooled)
            factor_logits[axis] = self.rfme_heads[axis](texture * semantic)

        class_factor = self.class_to_factor.to(device=pooled.device)
        delta = pooled.new_zeros((pooled.shape[0], self.spec.num_classes))
        # class_to_factor[c,axis] 指出第 c 类在该因素轴上的取值索引。
        for axis_index, axis in enumerate(FACTOR_AXES):
            index = class_factor[:, axis_index]
            delta = delta + factor_logits[axis].index_select(1, index)
        delta = F.layer_norm(delta, (self.spec.num_classes,))
        return delta, factor_logits, diagnostics

    def forward(
        self,
        image: torch.Tensor,
        *,
        return_aux: bool = False,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        """完整前向传播；普通推理返回 logits，训练审计可要求辅助字典。

        数据流为：RGB → 四级语义/DREL 双流 → 受限证据写入 → 全局语义分类头
        → RFME 因素残差 → 27 类 logits。
        """
        if valid_mask is not None:
            raise ValueError("DREL-RT uses direct 224x224 support and does not accept valid_mask")
        if image.ndim != 4 or int(image.shape[1]) != 3:
            raise ValueError(f"expected Bx3xHxW input, got {tuple(image.shape)}")
        semantic = self.semantic_stem(image)
        ledger = image
        stage_semantics: list[torch.Tensor] = []
        stage_evidence: list[torch.Tensor] = []
        stage_reliability: list[torch.Tensor] = []
        write_ratios: list[torch.Tensor] = []

        # 四个 stage 的分辨率逐级降低；语义流和证据账本始终并行推进。
        for stage_index in range(4):
            if stage_index:
                semantic = self.semantic_transitions[stage_index - 1](semantic)
            # (1) 成对方向滤波得到局部响应，EvidenceRefiner 清理噪声。
            response = self.response_banks[stage_index](ledger)
            ledger = self.evidence_refiners[stage_index](response)
            # (2) 将响应分成方向组成 composition 与能量/可观测性信息。
            decomposition = self.quotients[stage_index](ledger)
            composition = decomposition["composition"]
            log_energy = decomposition["log_energy"]
            # 按样本、按方向组标准化，消除任意响应尺度；sigmoid 把结果变成
            # 0 到 1 之间的局部可观测性/可靠性权重。
            mean = log_energy.float().mean((2, 3), keepdim=True)
            std = log_energy.float().var((2, 3), unbiased=False, keepdim=True).add(1.0e-6).sqrt()
            normalized_energy = (log_energy.float() - mean) / std
            # (3) 逐样本逐组标准化后压到 [0,1]，作为局部可靠性，不跨样本泄漏。
            reliability = torch.sigmoid(normalized_energy).to(log_energy)
            radial = torch.cat((torch.tanh(normalized_energy).to(log_energy), 2.0 * reliability - 1.0), dim=1)

            if self.variant in {"zero_write", "readonly_rfme"}:
                budget = self.write_budgets[stage_index] * 0.0
            else:
                budget = self.write_budgets[stage_index]
            # (4) 证据以受限范数写入语义流；readonly/zero 消融的 budget 为 0。
            semantic, actual = self.writers[stage_index](
                semantic, composition, radial, reliability, budget
            )
            semantic = self.semantic_stages[stage_index](semantic)
            stage_semantics.append(semantic)
            stage_evidence.append(torch.cat((composition, radial), dim=1))
            stage_reliability.append(reliability)
            write_ratios.append(actual)

        final_map = self.final_norm(semantic)
        pooled = final_map.mean((2, 3))
        # 基础预测只来自最终语义特征，因此 RFME 残差始终是可审计的增量。
        semantic_logits = self.head(pooled)
        factor_logits: dict[str, torch.Tensor] = {}
        rfme_diagnostics: dict[str, torch.Tensor] = {}
        if self.variant in {"uniform_rfme", "rfme", "readonly_rfme"}:
            delta, factor_logits, rfme_diagnostics = self._factor_residual(
                final_map, stage_semantics, stage_evidence, stage_reliability
            )
            # tanh 将 RFME 总影响限制在 [-1,1]；gate 从 0 开始学习。
            alpha = torch.tanh(self.rfme_gate_logit)
            logits = semantic_logits + alpha * delta
        else:
            alpha = self.rfme_gate_logit * 0.0
            logits = semantic_logits
        self.last_factor_logits = factor_logits
        self.last_aux = {
            "drel_rt_variant": self.variant,
            "drel_rt_write_budgets": self.write_budgets.detach().float(),
            "drel_rt_write_ratio": torch.stack(write_ratios, dim=1),
            "drel_rt_rfme_alpha": alpha.detach().float(),
            **rfme_diagnostics,
        }
        if not return_aux:
            return logits
        return {
            "logits": logits,
            "linear_logits": semantic_logits,
            "feature": pooled,
            "features": pooled,
            "backbone_embedding": pooled,
            "factor_logits": factor_logits,
            "boundary_logits": {},
            "baseline_backbone": self.architecture_version,
            **self.last_aux,
        }


__all__ = ["DRELRTSurfaceClassifier", "RoadContextBlock"]
