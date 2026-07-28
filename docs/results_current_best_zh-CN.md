# 当前已验证的全量测试结果

[English](results_current_best.md) | **简体中文**

当前公开的最佳自包含 S7 checkpoint 为：

```text
c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709
```

该 checkpoint 已完成 49,500 张 RSCD 图像的历史全量测试协议。另有一个“母模型 checkpoint + source-reliable router”的推理记录在同一测试集上达到 90.640% Top-1，但它在评估时加载母模型并附加 router，因此与自包含 S7 checkpoint 分开记录。

## 指标摘要

| 指标 | 数值 |
|---|---:|
| Top-1 | 90.632% |
| Macro-F1 | 88.920% |
| 平均精确率 | 88.729% |
| 平均召回率 | 89.226% |
| Weighted F1 | 90.654% |
| 总参数量 | 32.49M |
| S7 prefix-tuning 可训练参数 | 1.09M |
| 测试图像 | 49,500 |
| 类别数 | 27 |
| 最弱类别 | `water_concrete_slight` |
| 最弱类别 F1 | 75.693% |

## 指标含义

Top-1 是普通分类准确率：

```text
Top-1 = 第一预测正确的图像数 / 全部测试图像数
```

Macro-F1 先对每个类别单独计算 F1，再对 27 个类别等权平均：

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)
Macro-F1    = 27 个类别 F1_c 的平均值
```

Macro-F1 能防止样本较多或较容易的类别掩盖困难类别，因此它比只看 Top-1 更适合评估 RSCD。

## 当前主要短板

最弱类别是 `water_concrete_slight`。其困难来自三种视觉因素同时耦合：

- wet 与 water 的水膜边界可能非常接近；
- 水膜、反光和倒影会遮挡混凝土颗粒；
- slight 位于 smooth 与 severe 之间，类别边界狭窄。

因此后续创新重点应放在能改变中间表征的主干和证据组织方式，而不是只在最终 logits 上继续增加类别补丁。

机器可读证据位于：

```text
results/current_best_s7/metrics_summary.json
results/current_best_s7/per_class_metrics.csv
results/current_best_s7/confusion_matrix.csv
results/current_best_s7/hard_pair_metrics.csv
```

历史母模型、90.045% 初始验证记录和 90.640% router 记录见 [S7 checkpoint 训练谱系](s7_training_lineage.md)。

