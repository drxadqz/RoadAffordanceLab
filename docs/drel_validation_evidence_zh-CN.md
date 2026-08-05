# DREL D350 验证证据、当前进度与结论边界

> 对应模型：DREL-E B `component_full`。本页只报告冻结 D350 **validation**；没有读取、解析或推断 development proxy-test，也没有使用历史 49,500 张 formal test 选择 DREL。机器可读数据见 [`results/drel_d350_validation/metrics_summary.json`](../results/drel_d350_validation/metrics_summary.json)。[English](drel_validation_evidence.md)。

## 1. 当前最可靠结论

当前应进入后续完整 RSCD 训练的干净候选是 **DREL-E B `component_full`**，即：

- bounded-energy Evidence Ledger；
- 方向与径向/可靠性双证据；
- `0.25 × tanh` 有界写入；
- MomentPreservingTransition；
- 完整 RegionalContrastBlock 均值/偏离分解；
- 全约束 MatchedResponseConv2d；
- DropPath 0.10；
- 320 维 embedding 和 27 类线性头。

它是**当前验证证据最完整的生产候选**，不是已经刷新仓库 S7 正式测试结果的模型。首页 S7 正式测试结论保持不变，DREL 与其证据域严格分开。

## 2. 公平 Gate8 三种子结果

### 2.1 每颗种子

所有结果均来自独立 FP32 validation。训练协议为 D350 train 9,450 / val 1,350、原生 `360×240` tensor、从零训练、CE-only、无增强、无 teacher、无预训练、batch 16、梯度累积 4、相同 30-epoch cosine horizon，并在 epoch 8 比较。

| Seed | Top-1 | Macro-F1 | Bottom-5 | Min-class | WCS F1 | Roughness | 困难对 1 | 困难对 2 | Score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 97 | 0.334074 | 0.320235 | 0.172509 | 0.155844 | 0.212766 | 0.568148 | 0.220000 | 0.230000 | 0.296226 |
| 197 | 0.343704 | 0.328804 | 0.186541 | 0.151899 | 0.342342 | 0.611111 | 0.320000 | 0.390000 | 0.306311 |
| 297 | 0.348148 | 0.330983 | 0.173478 | 0.166667 | 0.177215 | 0.580000 | 0.160000 | 0.220000 | 0.306348 |
| **均值** | **0.341975** | **0.326674** | **0.177509** | **0.158137** | **0.244108** | **0.586420** | **0.233333** | **0.280000** | **0.302962** |
| **样本标准差** | 0.007194 | 0.005681 | 0.007837 | 0.007646 | 0.086911 | 0.022189 | 0.080829 | 0.095394 | 0.005834 |

其中：

- `Bottom-5` 是每次运行中最弱五类 F1 的平均值；
- `Min-class` 是最弱单类 F1；
- `WCS` 是 `water_concrete_slight` 类 F1；
- 困难对 1 是 `water_concrete_slight` 与 `water_concrete_severe` 的二类判别；
- 困难对 2 是 `water_concrete_slight` 与 `wet_concrete_slight` 的二类判别；
- `Score = 0.4×Top1 + 0.4×Macro-F1 + 0.2×Bottom-5`。

### 2.2 与冻结 RSPNet envelope 的公平比较

RSPNet envelope 是同 D350、同原生输入、同 Gate8 预算下，按列取 RSPNet-M 与 RSPNet-L 较高值的冻结参考。它来自 seed97 单种子，不是三种子分布。

| 指标 | DREL Gate8 三种子均值 ± 标准差 | RSPNet Gate8 envelope | 均值差（百分点） | 结论 |
|---|---:|---:|---:|---|
| Top-1 | **0.341975 ± 0.007194** | 0.322963 | **+1.901** | DREL 均值更高 |
| Macro-F1 | **0.326674 ± 0.005681** | 0.309301 | **+1.737** | DREL 均值更高 |
| Bottom-5 | **0.177509 ± 0.007837** | 0.172349 | **+0.516** | DREL 均值更高 |
| Min-class | **0.158137 ± 0.007646** | 0.126582 | **+3.156** | DREL 均值更高 |
| WCS F1 | **0.244108 ± 0.086911** | 0.227273 | **+1.684** | 均值更高，但种子波动大 |
| Roughness | 0.586420 ± 0.022189 | **0.608889** | **−2.247** | 唯一低于 envelope 的指标 |
| 困难对 1 | **0.233333 ± 0.080829** | 0.190000 | **+4.333** | 均值更高，种子波动大 |
| 困难对 2 | **0.280000 ± 0.095394** | 0.260000 | **+2.000** | 均值更高，种子波动大 |
| Score | **0.302962 ± 0.005834** | 0.280691 | **+2.227** | DREL 均值更高 |

严格表述应是：

> 在相同 D350 Gate8 validation 预算下，DREL `component_full` 的三种子均值在九个冻结指标中的八个高于单种子 RSPNet-M/L envelope，roughness 低 2.25 个百分点。

不能表述为：

- “DREL 已经在正式测试集超过 RSPNet”；
- “DREL 每颗种子、每个指标都超过 RSPNet”；
- “三种子显著超过 RSPNet”，因为 RSPNet 参考只有一颗种子，无法做匹配多种子显著性检验。

## 3. Gate30 收敛与生产确认

### 3.1 `component_full` 三种子 Gate30

| Seed | Top-1 | Macro-F1 | Bottom-5 | Min-class | WCS F1 | Roughness | Score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 97 | 0.385185 | 0.378540 | 0.204044 | 0.144928 | 0.265306 | 0.615556 | 0.346299 |
| 197 | 0.385185 | 0.383431 | 0.252150 | 0.146341 | 0.296875 | 0.627407 | 0.357877 |
| 297 | 0.392593 | 0.386616 | 0.215188 | 0.164384 | 0.339623 | 0.625926 | 0.354721 |
| **均值** | **0.387654** | **0.382863** | **0.223794** | **0.151884** | **0.300601** | **0.622963** | **0.352966** |
| **样本标准差** | 0.004274 | 0.004057 | 0.025005 | 0.010711 | 0.037233 | 0.006535 | 0.005961 |

roughness 从 Gate8 的 0.586 提升到 Gate30 的 0.623，说明架构有能力学到该属性，早期预算的确存在收敛不足。但是 RSPNet 没有同协议 Gate30 参考，因此 **0.623 不能用于公平宣称超过 RSPNet Gate8 的 0.609**。

### 3.2 为什么没有删除 RegionalContrast 分解

`component_no_regional` 在 Gate8 三种子筛选中一度是冠军，因此执行了预先冻结的 Gate30 删除确认。门槛和结果：

| 冻结门槛 | 结果 | 判定 |
|---|---:|---|
| 每颗种子 Score Δ ≥ +0.002 | seed197 +0.0005；seed297 −0.0070 | 失败 |
| Mean Macro-F1 Δ ≥ −0.001 | −0.002687 | 失败 |
| Mean Bottom-5 Δ ≥ −0.005 | +0.010960 | 通过 |

必须通过全部门槛才能删除，因此最终结论是 **REGIONAL_RETAINED**。这也是公开代码选择完整 `component_full` 而不是 Gate8 单点冠军的原因。

## 4. 已完成的负结果及其价值

以下路线均按预先冻结门槛停止，不应为了追求漂亮结果事后放宽或重开：

| 路线 | 试图解决的问题 | 冻结结果 | 生产处理 |
|---|---|---|---|
| RPCC + TERM | 增强乘积一致性和局部矩信息 | Gate3 失败 | 不加入 |
| all-stage amplitude | 让幅值写入所有阶段 | Gate3 失败，三种子综合分下降 | 不加入 |
| late-stage amplitude | 只在后期写幅值 | Gate3 失败 | 不加入 |
| BLADE | 分支解耦、选择性 Ledger、自适应幅值 | Gate3 失败 | 不加入 |
| CORAL roughness auxiliary | 直接提供粗糙度有序监督 | Gate3 最好仅约 +0.22pp roughness，未过 +1.0pp 门槛 | 不加入 |
| regional deletion | 简化区域分解 | Gate30 确认失败 | 保留 regional |

负结果不是“白跑”：它们防止把会竞争、会损害尾部类别或只在早期偶然占优的模块堆进最终模型。

## 5. 当前仍未解决的瓶颈

1. **公平 Gate8 roughness 缺口：** `−2.247pp`，是唯一低于 RSPNet envelope 的指标。
2. **Min-class 绝对值偏低：** Gate8 0.158，Gate30 0.152；平均指标上升不代表最弱类同步改善。
3. **WCS 与困难对跨种子波动：** Gate8 WCS F1 从 0.177 到 0.342，困难对标准差约 0.08–0.10。
4. **对照种子不匹配：** DREL 有三种子，RSPNet envelope 只有 seed97；当前只能报告均值相对 envelope，不能报告配对显著性。
5. **没有 DREL 正式 test 结果：** 这是有意的数据防火墙，不是遗漏。只有架构、训练与选择完全冻结后，才能一次性做正式测试。

## 6. 当前在研路线如何处理

针对 early roughness 的后续 RFCE（roughness-fiber consistency/evidence）筛选在通过自己的冻结 Gate3/Gate8 之前，严格视为实验分支：

- 不写入公开生产模型；
- 不改变本页 `component_full` 结论；
- 不使用 test 或 proxy-test 决策；
- 若失败则停止，若通过则需要完整三种子独立 FP32 结果和等预算比较后才能提议升级。

因此这次发布的是“现在已经有证据支持的最好干净算法”，而不是把未完成实验提前包装成改进。

## 7. 推荐的下一步

按性能价值排序：

1. 完成当前预注册的粗糙度单变量 Gate3；未过门槛立即停止；
2. 只有通过 Gate3 的冻结候选才与全部对照一起续到 Gate8；
3. 若 Gate8 同时改善 roughness、综合分和尾部指标，再考虑升级生产候选；
4. 架构冻结后，在完整 RSCD 上做匹配三种子训练；
5. 最后只做一次正式 test，并将 checkpoint、配置、数据哈希和逐类指标一起发布。

## 8. 可审计文件

- 算法实现：[`src/friction_affordance/models/drel.py`](../src/friction_affordance/models/drel.py)
- 冻结配置：[`configs/drel/component_full_d350.yaml`](../configs/drel/component_full_d350.yaml)
- 单元测试：[`tests/test_drel.py`](../tests/test_drel.py)
- 最小训练/推理示例：[`examples/drel_quickstart.py`](../examples/drel_quickstart.py)
- 机器可读指标：[`results/drel_d350_validation/metrics_summary.json`](../results/drel_d350_validation/metrics_summary.json)
- 完整算法解释：[DREL 算法说明](drel_algorithm_zh-CN.md)
