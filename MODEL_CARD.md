# Model Card — C3-FaRNet-S7

This is the top-level model card for the checkpoint released with
RoadAffordanceLab. The extended engineering notes remain available in
[`docs/model_card.md`](docs/model_card.md).

## Model summary

| Field | Value |
|---|---|
| Model | C3-FaRNet-S7 |
| Task | Single-image, 27-class visual road-surface condition recognition |
| Label structure | Condition/friction × material × roughness |
| Framework | PyTorch with CUDA mixed precision |
| Historical input protocol | 192×192 letterbox |
| Output | One 27-class logit vector plus structured auxiliary evidence |
| Evaluation mode | Single model, single crop, no ensemble |

The model combines a ConvNeXt visual carrier, explicit wetness/texture evidence,
factor-coupled interactions and calibrated residual handling for difficult class
boundaries. It is a visual recognition model, not a direct physical-friction
sensor.

## Verified result

The released checkpoint is linked to a frozen 49,500-image RSCD evaluation
record.

| Metric | Value |
|---|---:|
| Top-1 accuracy | **90.632%** |
| Macro-F1 | **88.920%** |
| Weighted-F1 | **90.654%** |
| Condition/friction accuracy | **96.596%** |
| Material accuracy | **97.210%** |
| Roughness accuracy | **95.176%** |
| Weakest-class F1 (`water_concrete_slight`) | **75.693%** |

These values are release evidence, not a claim of a new public SOTA. The
machine-readable source is [`results/current_best_s7`](results/current_best_s7),
and the checkpoint lineage is recorded in
[`results/s7_lineage/checkpoint_manifest.json`](results/s7_lineage/checkpoint_manifest.json).

## Intended use

- research on visual road-surface condition classification;
- factor-aware and fine-grained recognition experiments;
- reproducible training, evaluation and failure-slicing demonstrations;
- transfer-learning research after a separately frozen target-domain protocol.

## Out-of-scope use

- direct estimation of a physical friction coefficient;
- sole-source control decisions for a vehicle or other safety-critical system;
- deployment on unseen cameras, cities or weather regimes without validation;
- interpreting softmax confidence as calibrated physical certainty.

## Known limitations

- RSCD labels are visual proxies rather than synchronized force measurements.
- The verified result uses the historical 192×192 letterbox protocol.
- Water, reflection, shadows and weak roughness can remain visually ambiguous.
- Cross-dataset and native-resolution results must be reported as separate
  protocols and must not be mixed with the historical result.
- Reproducing the full evaluation requires a separately obtained RSCD copy and
  the Git-LFS checkpoint artifacts.

## Reproducibility entry points

1. [`configs/c3_farnet/current_best_s7_public.yaml`](configs/c3_farnet/current_best_s7_public.yaml)
2. [`scripts/verify_release.py`](scripts/verify_release.py)
3. [`docs/data_and_reproduction.md`](docs/data_and_reproduction.md)
4. [`docs/s7_training_lineage.md`](docs/s7_training_lineage.md)
5. [`docs/engineering.md`](docs/engineering.md)

## License and data

Repository code is released under the [MIT License](LICENSE). RSCD images and
third-party artifacts retain their original licenses and are not redistributed
by this repository.
