# DREL-E B `component_full`：从零开始的完整算法说明

> 版本：2026-08-05；代码：[`src/friction_affordance/models/drel.py`](../src/friction_affordance/models/drel.py)；冻结配置：[`configs/drel/component_full_d350.yaml`](../configs/drel/component_full_d350.yaml)。

## 1. 先用一句人话说明它在做什么

DREL 的目标是识别一张路面图像属于 27 种路面状态中的哪一种。普通分类网络会把所有视觉信息混在一条特征流中；DREL 则把计算分成两条职责不同的路径：

- **语义载波（semantic carrier）**负责回答“这整体上是什么路面”；
- **证据账本（evidence ledger）**专门记录方向纹理强度、总能量和可靠性，再以严格有界的方式把这些细粒度证据写给语义载波。

可以把它想成一个审稿流程：语义载波负责写最终结论，证据账本只能提交有格式、有上限的证据；结论不能反过来篡改原始证据。这个单向约束是 DREL 的核心，而不是简单地再加一条卷积分支。

## 2. 零基础预备知识

### 2.1 张量、通道和空间尺寸

图像进入 PyTorch 后通常写成 `B × C × H × W`：

- `B` 是一次送入模型的图片数量；
- `C=3` 是 RGB 三个颜色通道；
- `H` 和 `W` 是高和宽。

冻结 D350 实验使用 `B × 3 × 360 × 240`。网络中的“48 通道”可以理解为每个像素位置有 48 个不同的特征测量值。

### 2.2 卷积、下采样和池化

- **卷积**用一个小窗口扫描图像，提取边缘、纹理或更高层模式；
- **stride=2** 表示每次移动两格，使空间尺寸约减半；
- **depthwise 卷积**为每个通道分别做卷积，计算量较低；
- **1×1 卷积**不看邻居，只在同一位置混合通道；
- **全局平均池化**把每个通道的整张特征图压成一个数。

### 2.3 激活、归一化和残差

- `BatchNorm2d` 稳定每个通道的尺度；
- `ReLU6(x)=min(max(x,0),6)` 提供非线性并限制正值上界；
- `sigmoid` 把数压到 0 到 1；
- `tanh` 把数压到 -1 到 1；
- 残差连接 `output = input + update` 让模块只需要学习“应该改多少”。

## 3. 总体结构

```text
输入 RGB：B×3×360×240
        │
        ├──────────────── Evidence Ledger ────────────────┐
        │       /2、/4、/8、/16，始终更精细一倍          │
        │       方向响应 → 能量/可靠性 → 有界证据         │
        │                         │ 只能单向写入           │
        ▼                         ▼                       │
Semantic Stem → Stage 0 → Stage 1 → Stage 2 → Stage 3    │
                 /4        /8        /16       /32        │
                 48ch      96ch      192ch     320ch      │
        │
        ▼
BatchNorm → 全局平均池化 → 320 维向量 → Linear(320,27)
        │
        ▼
27 个类别 logits
```

冻结通道与深度如下：

| 阶段 | 语义通道 | 区域块数量 | Ledger 通道 | 响应核 | 语义尺寸 | Ledger 响应尺寸 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 48 | 1 | 16 | 7×7 | 90×60 | 180×120 |
| 1 | 96 | 2 | 24 | 5×5 | 45×30 | 90×60 |
| 2 | 192 | 5 | 32 | 3×3 | 23×15 | 45×30 |
| 3 | 320 | 2 | 48 | 3×3 | 12×8 | 23×15 |

Ledger 恰好比对应语义网格精细一个八度。“八度”在这里就是宽和高约为两倍。奇数尺寸使用 `ceil(H/2)`，代码会显式检查池化后的尺寸；关系漂移时直接报错，不静默插值。

## 4. 语义载波

### 4.1 Semantic Stem

输入先经过两次 stride-2：

```text
Conv3×3, 3→48, stride=2
BatchNorm + ReLU6
Depthwise Conv3×3, 48→48, stride=2
Conv1×1, 48→48
BatchNorm + ReLU6
```

结果从 `3×360×240` 变为 `48×90×60`。Stem 的任务是用较低成本建立基础语义特征。

### 4.2 MomentPreservingTransition

普通 stride-2 卷积可能把正负相抵的细纹理平均掉。DREL 在阶段之间先计算局部一阶和二阶矩：

\[
\mu = \operatorname{AvgPool}_{3\times3}(x),
\]

\[
m_2 = \operatorname{AvgPool}_{3\times3}(x^2),
\]

\[
\sigma = \sqrt{\max(m_2-\mu^2,0)+10^{-6}}.
\]

然后拼接 `[μ, σ]`，使用 `1×1 Conv + BN + ReLU6` 投影到下一阶段通道数。直觉上：

- `μ` 告诉模型局部平均是什么；
- `σ` 告诉模型局部起伏有多大；
- 即使粗糙纹理的正负响应平均接近零，`σ` 仍不会变成零。

这不是声称完整保留了所有像素，而是保证下采样不会只因为有符号均值抵消就丢掉局部色散。

### 4.3 RegionalContrastBlock

每个语义阶段由若干区域对比块组成。对输入 `x`：

1. 先做 BatchNorm，得到 `z`；
2. 反射边界下做 3×3 局部平均 `m`；
3. 计算局部偏离 `d=z-m`；
4. 用 5×5 depthwise 卷积处理低频区域上下文 `m`；
5. 用 dilation=2 的 3×3 depthwise 卷积处理高频偏离 `d`；
6. 两支相加、归一化和激活；
7. 用 1×1 卷积扩展为内容和门控两部分；
8. 计算 `ReLU6(content) × sigmoid(gate)`；
9. 压回原通道数，经 DropPath 后残差相加。

公式化表示：

\[
m=\operatorname{Avg}_{3\times3}(z),\qquad d=z-m,
\]

\[
o=\operatorname{ReLU6}\left(\operatorname{BN}
(K_{5\times5}^{ctx}(m)+K_{3\times3,d=2}^{dev}(d))\right),
\]

\[
(c,g)=\operatorname{split}(\operatorname{BN}(W_{expand}o)),
\]

\[
y=x+\operatorname{DropPath}\left(
\operatorname{BN}(W_{compress}(\operatorname{ReLU6}(c)\odot\sigma(g)))
\right).
\]

为什么保留它？删除区域均值/偏离分解的版本在 Gate8 看起来更好，但在预先冻结的三种子 Gate30 删除确认中，seed297 的综合分数反向下降 0.70 个百分点，Macro-F1 均值也未过门槛。因此证据不支持生产删除，`component_full` 保留该模块。

## 5. 证据账本

### 5.1 为什么需要单独的 Ledger

一条普通语义网络会不断降采样和混合通道。这样有利于识别整体类别，但方向细纹理、弱边缘和局部粗糙度可能过早被覆盖。Ledger 保持更细的空间网格，并具有三条规则：

1. 只从 RGB 或上一级 Ledger 读取；
2. 可向语义载波写证据；
3. 不允许从语义载波回读，因此语义偏好不能污染证据定义。

### 5.2 MatchedResponseConv2d

每个 group 有四个方向，每个方向有 even/odd 两个相位，所以 `group_width=4×2=8`。输出通道顺序是：

```text
group → orientation → (even, odd)
```

它不是普通自由卷积。每个 group 只学习一个规范 even/odd 残差，再旋转为四个方向；四个方向因此共享同一个径向基础。每次前向都会执行：

- 每个输入通道切片减去空间均值，实现零 DC；
- 残差范数严格限制在 0.25 以下；
- even 滤波器归一化为单位范数；
- odd 对 even 做 Gram–Schmidt 正交化，再归一化。

测试公开了三个数值审计接口：`zero_dc_error()`、`unit_norm_error()` 和 `pair_orthogonality_error()`。

设响应为 `r[g,o,p,h,w]`，其中 `g` 是组、`o` 是方向、`p∈{even,odd}` 是相位。方向能量为：

\[
E_{g,o}=r_{g,o,even}^2+r_{g,o,odd}^2.
\]

相位平方和使能量不依赖局部相位正负，更适合表示纹理强度。

### 5.3 RadialCompositionQuotient

对四个方向能量求和：

\[
T_g=\sum_o E_{g,o}.
\]

模块还计算：

\[
L_g=\frac{1}{2}\log(\max(T_g,10^{-12})),
\]

\[
R_g=\frac{T_g}{T_g+\nu_g},
\qquad
\nu_g=10^{-6}+(1-10^{-6})\operatorname{sigmoid}(a_g),
\]

其中 `a_g` 是每组可学习的可靠性阈值 logit。低能量区域可靠性接近 0，高能量区域接近 1。

原始 quotient 的可观测性平滑仍会被计算并保存在诊断输出里，但生产 `component_full` 的方向写入使用经过消融验证的 **bounded-energy coordinate**，不是 protected quotient composition。

### 5.4 方向与径向证据坐标

方向坐标先把非负能量压到 `[0,1)`：

\[
B_{g,o}=\frac{E_{g,o}}{1+E_{g,o}},
\]

再减去组内四方向均值：

\[
D_{g,o}=B_{g,o}-\frac{1}{4}\sum_{o'=1}^{4}B_{g,o'}.
\]

这样 `D` 表示“某方向相对同组平均方向更强还是更弱”，而不是总亮度或总对比度。

径向坐标拼接两部分：

\[
A_g=\left[\tanh(L_g/4),\;2R_g-1\right].
\]

- `tanh(L/4)` 是有界的总响应尺度；
- `2R-1` 把可靠性从 `[0,1]` 映射到 `[-1,1]`。

所以方向分支回答“纹理朝哪里”，径向分支回答“纹理响应有多强、是否可信”。

### 5.5 单向有界写入

方向和径向坐标先用固定的反射 3×3、stride-2 平均池化到语义网格，再各自通过 1×1 投影：

\[
u_s=P_s^D(\operatorname{pool}(D_s))
    +P_s^A(\operatorname{pool}(A_s)).
\]

最终写入严格有界：

\[
w_s=0.25\tanh(u_s).
\]

因此每个写入元素必定位于 `(-0.25,0.25)`。语义阶段的输入为：

\[
S_s'=S_s+w_s.
\]

两个 1×1 writer 在初始化时全零，所以 epoch 0 的 DREL 与零写入对照具有完全相同的输出；训练后只有数据支持的证据才逐渐进入语义载波。这一设计让消融比较不会被不同随机起点混淆。

### 5.6 HomogeneousCarrierBlock

前三个 Ledger 阶段用轻量残差块细化响应：

```text
normalized depthwise 3×3
→ PReLU
→ normalized pointwise 1×1
→ 0.001 × update
→ residual add
```

卷积权重在前向时归一化，参数和计算保持 FP32。这个路径没有 bias，因此对正比例缩放保持齐次性，不会凭空制造一个与输入无关的响应偏置。最后一级不再细化，直接使用响应。

## 6. 分类头与输出

Stage 3 输出经 BatchNorm 后，在空间维度求平均得到 320 维 embedding。冻结模型的 `out_dim=320`，所以 output projection 是 Identity。最后使用：

```text
Linear(320, 27)
```

得到 27 个 logits。logit 不是概率；用 `softmax(logits, dim=1)` 才得到和为 1 的类别概率。训练使用交叉熵：

\[
\mathcal{L}_{CE}=-\log
\frac{\exp(z_y)}{\sum_{k=1}^{27}\exp(z_k)}.
\]

冻结 D350 证据全部是 CE-only，没有 teacher、预训练、数据增强或额外粗糙度损失。

## 7. 完整前向伪代码

```python
semantic = semantic_stem(image)   # /4
ledger = image                    # /1

for stage in range(4):
    if stage > 0:
        semantic = moment_transition(semantic)

    response = matched_response.at(stage)(ledger)  # Ledger /2 each stage
    q = quotient.at(stage)(response)
    ledger = homogeneous_refiner.at(stage)(response)

    energy = directional_energy(response)
    direction = energy / (1 + energy)
    direction = direction - mean(direction, over="orientation")
    radial = concat(tanh(q.log_energy / 4), 2*q.reliability - 1)

    raw = direction_writer(pool_one_octave(direction)) \
        + radial_writer(pool_one_octave(radial))
    write = 0.25 * tanh(raw)

    semantic = regional_stage.at(stage)(semantic + write)

embedding = global_average(batch_norm(semantic))
logits = linear_27(embedding)
```

## 8. 参数量从哪里来

- DREL backbone：**2,751,694** 个可学习参数；
- 320→27 线性头：`320×27+27=8,667`；
- 完整分类器：**2,760,361** 个可学习参数。

固定 anchor、旋转网格和语义 schema buffer 不属于可学习参数。测试会锁定这些参数量，意外增加或删除模块时 CI 会失败。

## 9. 最短使用方式

```python
import torch
from friction_affordance.models.drel import build_drel_component_full

model = build_drel_component_full(num_classes=27, head_init_seed=970027)
image = torch.randn(2, 3, 360, 240)
logits = model(image)              # [2, 27]
probability = logits.softmax(1)    # [2, 27]
prediction = probability.argmax(1)
```

查看内部证据诊断：

```python
model.eval()
with torch.no_grad():
    result = model(image[:1], return_aux=True)

print(result["drel"]["drel_write_rms"])          # 四阶段写入强度
print(result["drel"]["drel_mean_reliability"])  # 四阶段平均可靠性
```

运行一个包含前向、交叉熵、反向和优化器 step 的完整最小示例：

```bash
python examples/drel_quickstart.py
```

## 10. checkpoint 兼容性

公开类保留了研究模型 `DRELComponentStudyBackbone(study_mode="full")` 的参数名和四个语义 buffer：

- `_drel_ledger_mode_code = 1`；
- `_drel_schema_version = 1`；
- `_drel_component_study_code = 0`；
- `_drel_component_study_schema = 1`。

因此从训练 checkpoint 取出模型 state dict 后，可以严格加载：

```python
payload = torch.load("checkpoint.pth", map_location="cpu")
state = payload["model"]  # 按实际 checkpoint 容器键取值
model.load_state_dict(state, strict=True)
```

发布前的本地等价审计使用相同种子分别创建研究 `component_full` 和公开 `DRELBackbone`：2,751,694 个参数、state-dict 键、每个 tensor 的值和前向输出均完全相同。

## 11. 哪些东西被明确排除

“干净”不等于随意删模块，而是只保留通过冻结决策的生产路径。以下路线没有加入：

- `component_no_regional`：Gate30 删除确认失败；
- RPCC product consensus 与 TERM local moment：Gate3 失败；
- 全阶段或后阶段 additive amplitude：Gate3 失败；
- BLADE 分支解耦、选择性 Ledger、自适应幅值：Gate3 失败；
- CORAL roughness auxiliary：Gate3 失败；
- 未完成冻结判定的后续粗糙度实验：在通过门槛前不进入生产代码。

这可以防止“做过的模块全都堆进去”造成参数膨胀、机制竞争和事后选择偏差。

## 12. 如何正确理解当前结论

DREL 是当前 D350 上证据最完整的**生产候选架构**，不是已经在历史 49,500 张正式测试集上刷新 S7 的模型。公平 Gate8 validation 下，它在九个跟踪指标中的八个超过冻结 RSPNet envelope，但 roughness 仍低 2.25 个百分点。Gate30 roughness 提升到 0.623，说明架构能够学习该属性，但没有同预算 RSPNet Gate30 对照，因此不能用它声称公平超越。

完整数字、限制和禁止使用的表述见[验证证据与结果边界](drel_validation_evidence_zh-CN.md)。
