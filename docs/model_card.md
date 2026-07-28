# Model Card: C3-FaRNet S7

## Model summary

| Field | Value |
|---|---|
| Model | C3-FaRNet S7 |
| Task | 27-class visual road-surface condition recognition |
| Dataset | RSCD |
| Framework | PyTorch |
| Input | 192×192 RGB image, aspect-ratio-preserving letterbox |
| Output | 27-class logits plus internal factor/physics evidence |
| Parameters | 32.49M total; approximately 1.09M trainable in the S7 recipe |
| License | MIT for repository code; dataset and third-party assets retain original licenses |

C3-FaRNet stands for **Coupled Conditioned Friction-Affordance Road Network**. It combines a ConvNeXt visual carrier, explicit physics/texture evidence and a factor-coupled classification head.

## Intended use

Appropriate uses include:

- research on road-surface image classification;
- visual road-state recognition benchmarks;
- factorized and fine-grained classification research;
- model evaluation, calibration and failure analysis;
- transfer-learning and domain-generalization experiments.

## Out-of-scope use

The model must not be used as:

- a direct tire-road friction coefficient sensor;
- the sole input to an autonomous-driving or braking decision;
- a substitute for calibrated physical road-friction measurement;
- a guaranteed predictor under unseen cameras, countries, weather or road materials;
- evidence of causal physical recovery from a single image.

## Label semantics

Most RSCD labels can be decomposed into condition/friction state, material and roughness. For example:

```text
water_concrete_slight = water + concrete + slight
```

Snow and ice classes have different factor validity from asphalt/concrete classes. The implementation maintains canonical factor mappings so that invalid factor combinations are not treated as ordinary labels.

## Training data and protocol

The released result records:

| Split | Images |
|---|---:|
| Training | 958,941 |
| Validation | 19,860 |
| Test | 49,500 |

The historical S7 run uses a parent checkpoint, a dry-concrete teacher checkpoint and selective fine-tuning. The exact lineage is documented in [`s7_training_lineage.md`](s7_training_lineage.md). The dataset images are not redistributed.

## Evaluation result

The standalone released S7 checkpoint has the following frozen full-test record:

| Metric | Value |
|---|---:|
| Top-1 | 90.6323% |
| Macro-F1 | 88.9197% |
| Weighted F1 | 90.6539% |
| Condition/friction accuracy | 96.5960% |
| Material accuracy | 97.2101% |
| Roughness accuracy | 95.1758% |
| Weakest-class F1 | 75.6931% (`water_concrete_slight`) |
| Test samples | 49,500 |

This is a single-model result. It is not reported as a new public SOTA claim. The result is specific to the historical RSCD protocol and should only be compared with methods evaluated under an equivalent split and inference setup.

## Primary limitations

1. **Visual proxy rather than physical friction.** The target labels describe visible road conditions, not synchronized tire-force measurement.
2. **Protocol dependence.** The reported model uses 192×192 letterbox preprocessing and a historical warm-start chain.
3. **Weakest classes remain difficult.** Water/wet concrete and slight/severe boundaries dominate the remaining errors.
4. **Potential temporal correlation.** RSCD can contain visually adjacent frames; group-disjoint evaluation is needed to estimate cross-route generalization.
5. **Domain shift is not solved.** Camera response, geographic region, illumination and surface composition can change the evidence distribution.
6. **Physics cues are approximations.** Brightness, specularity and gradients are useful image evidence but not a complete generative model of a wet road.

## Bias and reliability considerations

- Aggregate accuracy can hide low-F1 classes; Macro-F1 and per-class F1 must be reported.
- Confidence scores have not been certified for safety-critical probability calibration.
- Padding and input geometry may become shortcuts if preprocessing changes.
- Models can exploit capture-specific patterns. Cross-date, cross-route and cross-dataset protocols should be added before deployment claims.

## Reproducibility levels

| Level | Publicly supported | Requirements |
|---|---|---|
| Evidence audit | Yes | Python only; no dataset or GPU |
| Checkpoint integrity | Yes | Git LFS artifacts |
| Full evaluation | Yes, given data | Local RSCD copy and CUDA-capable environment |
| Historical training continuation | Yes, given data | Parent/teacher LFS checkpoints and RSCD manifests |
| Exact raw-data redistribution | No | RSCD must be obtained from its original source |

## Contact and reporting

Use GitHub Issues for reproducibility problems. Security-sensitive reports should follow [`SECURITY.md`](../SECURITY.md).
