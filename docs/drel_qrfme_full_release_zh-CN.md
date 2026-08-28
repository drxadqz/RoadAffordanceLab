# DREL-QRFME：完整 RSCD 发布说明

[English](drel_qrfme_full_release.md) | [仓库首页](../README_zh-CN.md) | [机器可读证据](../results/drel_qrfme_epoch097/metrics_summary.json)

## 1. 问题与设计假设

RSCD 的 27 个类别是耦合路面状态，例如 `water_concrete_slight` 同时包含摩擦/状态、材质和粗糙度。材质判断依赖整体外观，但 wet–water、smooth–slight 等边界常依赖微弱的方向性、多尺度纹理响应。普通语义主干可能在下采样时抹掉这些响应；无限制融合又可能让局部噪声压过稳定语义。

DREL-QRFME 因此把**稳定语义上下文**和可审计的**方向响应证据账本**分开，用显式范数上界连接两者，再用因子语义查询读取证据。它是一条统一的因果设计链，而不是多个注意力模块的堆叠。

## 2. 从输入到输出的严格计算

输入为归一化图像 \(x\in\mathbb{R}^{3\times224\times224}\)。网络同步产生四级语义图 \(s_l\) 和四级证据账本 \(e_l\)。

### 2.1 语义路面上下文流

Stem 将图像下采样 4 倍。每个 `RoadContextBlock` 先归一化、扩展通道，一半通过局部 3×3 depthwise 卷积，另一半通过 7×7 context 卷积，拼接投影后以小 LayerScale 残差写回：

\[
s' = s + \operatorname{DropPath}\left(\gamma\odot W_p\,\sigma\bigl([D_{3\times3}(u),D_{7\times7}(v)]\bigr)\right).
\]

四级通道数为 `[96, 192, 384, 640]`，深度为 `[2, 3, 9, 3]`。

### 2.2 方向响应证据账本

每一级先用 `MatchedResponseConv2d` 计算成对方向响应，由 `EvidenceRefiner` 清理局部噪声，再用 `RadialCompositionQuotient` 分解为带符号的方向组成、对数响应能量和有界径向坐标。局部可靠性只用同一样本、同一方向组内部的统计量：

\[
r_{l,g,p}=\sigma\!\left(\frac{a_{l,g,p}-\mu_{l,g}}{\sqrt{\operatorname{Var}(a_{l,g})+10^{-6}}}\right).
\]

不同样本之间不共享统计量，因此不存在跨样本信息泄漏。

### 2.3 有界证据写入

方向组成与径向证据投影到语义通道后由可靠性调制。`NormBudgetWriter` 对每个样本裁剪写入项的 L2 范数：

\[
\tilde u_l=\tanh(W_c c_l+W_r q_l)\odot r_l,\quad
u_l=\min\left(1,\frac{b_l\lVert s_l\rVert_2}{\lVert\tilde u_l\rVert_2}\right)\tilde u_l,
\]

\[
s_l\leftarrow s_l+u_l,\qquad \frac{\lVert u_l\rVert_2}{\lVert s_l\rVert_2}\le b_l\le0.05.
\]

两个投影卷积从零初始化，所以证据支路不会在训练初期突然破坏语义函数。5% 上界由前向计算强制保证并写入日志，而不是只靠正则项软约束。

### 2.4 查询条件 RFME

RFME 读取第 2、3 级特征：它们保留了足够空间细节，同时避开第 1 级高成本噪声和最终 7×7 特征过粗的问题。对因素轴 \(a\in\{\text{friction},\text{material},\text{roughness}\}\)：

\[
z^a_p=W^a_s s_p+W^a_e e_p.
\]

全局语义查询和局部 key 计算相关性，再与可靠性共同构造归一化空间测度：

\[
m^a_p=\operatorname{softmax}_p\left(\log \bar r_p + \frac{\langle q^a,k^a_p\rangle}{\sqrt d}\right),
\qquad \sum_p m^a_p=1.
\]

局部描述子被软分配给因素 codeword，条件残差矩为：

\[
\rho^a_k=\frac{\sum_p m^a_p\,\pi^a_{p,k}(z^a_p-c^a_k)}{\max(\sum_p m^a_p\,\pi^a_{p,k},10^{-6})}.
\]

两个尺度的残差矩与最终语义特征相乘融合，按固定 27 类因素表还原成类别残差 \(\Delta\)。最终输出：

\[
y = y_{\mathrm{semantic}} + \tanh(\alpha)\,\Delta.
\]

门控从 0 开始，先建立稳定语义基线，再由训练决定因素残差是否有用。

## 3. 理论与程序的对应关系

| 数学角色 | 程序实现 |
|---|---|
| 语义载体 | `src/drel_qrfme/models/drel_qrfme_model.py::RoadContextBlock` |
| 方向证据账本 | `MatchedResponseConv2d`、`EvidenceRefiner`、`RadialCompositionQuotient` |
| 写入硬约束 | `NormBudgetWriter` |
| 查询条件测度与 codeword 残差 | `FactorMeasureStage` |
| 固定因素—类别映射 | `src/drel_qrfme/rscd_label_factors.py` |
| 最终模型 | `DRELRTSurfaceClassifier` |
| 只用于训练的类别先验修正 | `training_engine.py` 中的 Balanced Softmax |

公开 checkpoint 对应 `drel_rt_rfme_v1`。修改通道、深度、账本结构、类别顺序或前向数值次序都会破坏严格兼容性。

## 4. 训练协议

- 训练/验证/官方测试：958,941 / 19,860 / 49,500 张；
- seed 97，完整训练 100 轮，自然采样；
- AdamW，初始学习率 5e-4，5 轮 warm-up，余弦衰减到 1e-6；
- batch size 64，梯度累积 2；
- BF16 训练、FP32 验证与测试；
- Balanced Softmax 只作用于训练损失；
- 从零训练，不使用 ImageNet 预训练、教师模型、类别 logit 补丁、TTA 或集成。

checkpoint 综合分数为 `0.60 Top-1 + 0.30 Macro-F1 + 0.10 Bottom-5 mean F1`。Epoch 97 由验证集选中；Epoch 100 指标更低，不能替代最佳模型。

公开 YAML 保留冻结 manifest 哈希。原始完整 manifest 含服务器数据路径，因此不重新分发；使用者需在不改变类别顺序和 split 成员的前提下生成等价本地 manifest，并配置发布包路径映射。

## 5. 实验结果

### 5.1 验证集选择

| Epoch | 验证 Top-1 | Macro-F1 | Bottom-5 F1 | 综合分数 |
|---:|---:|---:|---:|---:|
| **97** | **91.309%** | **89.623%** | **75.995%** | **89.272%** |

### 5.2 官方测试集：direct resize 224

| Top-1 | Macro-F1 | Weighted-F1 | Bottom-5 F1 | 最差类别 F1 | NLL | ECE-15 |
|---:|---:|---:|---:|---:|---:|---:|
| **92.265%** | **90.061%** | **92.283%** | **79.359%** | **75.508%** | 0.22025 | 0.02168 |

三个因素准确率分别为 friction 97.453%、material 98.152%、roughness 95.523%。

### 5.3 与同协议 RSPNet-L 比较

| 指标 | DREL-QRFME | RSPNet-L 本机复测 | 差值 |
|---|---:|---:|---:|
| Top-1 | 92.265% | 92.034% | +0.230 pp |
| Macro-F1 | 90.061% | 89.474% | +0.587 pp |
| Bottom-5 F1 | 79.359% | 77.670% | +1.689 pp |
| 最差类别 F1 | 75.508% | 72.814% | +2.693 pp |

### 5.4 多协议复核

同一冻结 checkpoint 在“短边缩放 1.14 后中心裁剪 224”协议下取得 92.057% Top-1 和 89.819% Macro-F1。仓库同时发布两种协议的完整结果，让读者能够区分模型能力和图像预处理的影响。

## 6. 结论边界与后续实验

该成绩是真实完整官方测试结果，不是 100k 代理结果或 validation-only 指标。它支持“DREL-QRFME 在 direct-resize 同协议下超过本机公开 RSPNet-L 复测”的结论；但目前仅完成 seed 97，尚不能证明统计显著的全局 SOTA。论文级结论还需要多种子、DREL/Writer/RFME 训练消融，以及所有外部方法在明确公共协议下的直接复现。

目前最弱类别为 `water_concrete_slight`。下一步应围绕这一因素边界做受控分析，而不是加入手工类别修正规则。
