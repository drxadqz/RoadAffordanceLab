# Current Verified Full-Test Result

![Verified C3-FaRNet S7 result overview](../assets/results-overview.svg)

The current best self-contained completed S7 checkpoint is:

```text
c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709
```

It completed the full RSCD test split with 49,500 images.

There is also a parent-checkpoint plus source-reliable-router inference record with `90.640%` Top-1 on the same 49,500-image test split. That record is documented separately because it loads the parent checkpoint and applies the router at evaluation time, rather than using the standalone formal S7 checkpoint.

## Summary

| metric | value |
|---|---:|
| Top-1 | 90.632% |
| Macro-F1 | 88.920% |
| Mean precision | 88.729% |
| Mean recall | 89.226% |
| Weighted F1 | 90.654% |
| Total parameters | 32.49M |
| Trainable parameters in S7 prefix-tuning | 1.09M |
| Test images | 49,500 |
| Classes | 27 |
| Weakest class | water_concrete_slight |
| Weakest-class F1 | 75.693% |

## What the Metrics Mean

Top-1 is the ordinary classification accuracy:

```text
Top-1 = number of correctly classified test images / number of all test images
```

Macro-F1 computes F1 for every class and then averages all classes equally:

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
Macro-F1    = average(F1_c over all 27 classes)
```

Macro-F1 is important here because RSCD has hard classes and easier classes. A high Top-1 can still hide weak minority or boundary classes.

## Main Weakness

The weakest class is:

```text
water_concrete_slight
```

Its F1 is 75.693%. The main reason is visual coupling:

- water and wet films can look similar
- concrete texture can be partially hidden by water film or glare
- slight roughness is between smooth and severe, so its visual boundary is naturally narrow

This is why the next research direction should focus on early feature conditioning and coupled factor modeling rather than only adding late classifier heads.

See:

```text
results/current_best_s7/per_class_metrics.csv
results/current_best_s7/confusion_matrix.csv
results/current_best_s7/hard_pair_metrics.csv
```

For the parent checkpoint, the `90.045%` initial-validation number, and the source-router `90.640%` record, see [s7_training_lineage.md](s7_training_lineage.md).
