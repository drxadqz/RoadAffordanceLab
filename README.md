<div align="center">

![RoadAffordanceLab banner](assets/road-affordance-lab-banner.svg)

[**English**](README.md) | [**简体中文**](README_zh-CN.md) | [DREL-QRFME method](docs/drel_qrfme_full_release.md) | [Reproducibility](docs/data_and_reproduction.md) | [Evidence](results/drel_qrfme_epoch097/metrics_summary.json)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](environment-faf-paper.yml)
[![Repository quality](https://github.com/drxadqz/RoadAffordanceLab/actions/workflows/repository-quality.yml/badge.svg)](https://github.com/drxadqz/RoadAffordanceLab/actions/workflows/repository-quality.yml)
[![License](https://img.shields.io/github/license/drxadqz/RoadAffordanceLab?color=155EEF)](LICENSE)

**DREL-QRFME: bounded directional-response evidence and query-conditioned factor measures for 27-class road-surface recognition.**

</div>

## Full-RSCD result

DREL-QRFME was trained from scratch on all **958,941** training images. The released Epoch-97 checkpoint was selected only on the **19,860-image validation split** and then evaluated once on the disjoint **49,500-image official test split**. There is no basename overlap between test and train/validation.

| Model | Test protocol | Top-1 | Macro-F1 | Bottom-5 F1 | Weakest-class F1 | Parameters |
|---|---|---:|---:|---:|---:|---:|
| **DREL-QRFME (ours)** | released RSPNet direct resize 224 | **92.265%** | **90.061%** | **79.359%** | **75.508%** | 26.15M |
| RSPNet-L (released implementation, local rerun) | released RSPNet direct resize 224 | 92.034% | 89.474% | 77.670% | 72.814% | 3.69M |
| **Difference (ours − RSPNet-L)** | identical machine and preprocessing | **+0.230 pp** | **+0.587 pp** | **+1.689 pp** | **+2.693 pp** | — |

The DREL-QRFME row uses one model, one crop, raw logits, no ensemble, and no test-time augmentation. Its weakest class is `water_concrete_slight`. A second frozen preprocessing audit—resize the short side by 1.14 and center crop 224—produced **92.057% Top-1**, **89.819% Macro-F1**, **79.055% Bottom-5 F1**, and **75.587% weakest-class F1**.

> Claim boundary: the direct-resize comparison is a post-hoc same-protocol comparison of one frozen checkpoint, not transform selection. The result exceeds our same-machine RSPNet-L rerun, but it is still a single-seed experiment; multi-seed confirmation is required before claiming statistically established SOTA.

Machine-readable metrics, per-class scores, confusion matrices, protocol metadata, training history, and hashes are in [`results/drel_qrfme_epoch097`](results/drel_qrfme_epoch097). The released checkpoint is selected at **Epoch 97**, contains **26,145,426** parameters, and has SHA-256 `72c2c2cd34545dcc1e22d5c89d43d1c79b3278a3edd7bc5857e0ab0a493dff55`.

## Architecture

![DREL-QRFME architecture](assets/drel-qrfme-overview.svg)

DREL-QRFME is organized around one scientific question: how can fragile road-texture evidence modify a stable semantic representation without overwhelming it?

1. **Semantic road-context stream.** Four stages combine local 3×3 and wider 7×7 depthwise context to represent global material and appearance.
2. **Directional Response Evidence Ledger (DREL).** Signed directional responses preserve weak texture and boundary cues; protected quotient decomposition separates composition from response energy/reliability.
3. **Bounded evidence writing.** `NormBudgetWriter` injects evidence into the semantic stream under an explicit per-stage norm budget capped at 5%.
4. **Query-conditioned RFME.** Semantic factor queries and local reliability define normalized spatial measures. Conditional residual moments are decoded along condition, material, and roughness axes and added to the 27-class semantic logits through a learned, bounded residual gate.

Balanced Softmax corrects the long-tailed class prior only inside the training loss. Validation and inference always use the model's raw logits. The model rejects pretrained weights and does not use a teacher, hand-written class correction, TTA, or ensemble.

The equations, tensor shapes, invariants, and code mapping are documented in [DREL-QRFME full release](docs/drel_qrfme_full_release.md) and [中文说明](docs/drel_qrfme_full_release_zh-CN.md).

## Quick start

```bash
git clone https://github.com/drxadqz/RoadAffordanceLab.git
cd RoadAffordanceLab
git lfs install && git lfs pull
pip install -r requirements.txt
pip install -e .
```

Train with the frozen full-RSCD recipe after preparing the dataset manifests and path remapping described in the method document:

```bash
python train_drel_qrfme.py \
  --dataset-root /path/to/RSCD \
  --config configs/drel_qrfme/rscd_full_train_seed097.yaml \
  --output-dir runs/drel_qrfme_seed097
```

Evaluate the released checkpoint on the validation split:

```bash
python evaluate_drel_qrfme.py \
  --dataset-root /path/to/RSCD \
  --config configs/drel_qrfme/rscd_full_train_seed097.yaml \
  --checkpoint checkpoints/drel_qrfme_epoch097/best_checkpoint.pth \
  --output-dir runs/eval_epoch097
```

Official test evaluation is deliberately separated from the default validation command to preserve the test firewall. The exact test outputs are released as immutable evidence rather than silently reusing the test split during development.

## Repository map

```text
src/drel_qrfme/                 DREL-QRFME model, operators, losses, data, and engine
configs/drel_qrfme/             Frozen model and full-RSCD training recipe
checkpoints/drel_qrfme_epoch097 Released best checkpoint and SHA-256 record
results/drel_qrfme_epoch097/    Training, validation, test, per-class, and protocol evidence
docs/drel_qrfme_full_release*   Bilingual theory-to-code documentation
train_drel_qrfme.py             Stable training entry point
evaluate_drel_qrfme.py          Validation-safe evaluation entry point
```

Earlier C3-FaRNet-S7 full-test materials and the DREL-E B validation study remain versioned for research lineage, but they are not the headline result of this release.

## Reproducibility and integrity

- Seed: 97; 100 full epochs; best checkpoint: Epoch 97.
- Optimizer: AdamW; BF16 training and FP32 evaluation.
- Natural full-data sampling and Balanced Softmax; no pretrained weights.
- Checkpoint, metrics, evaluation predictions, and protocol payloads are hash-addressed.
- Training, validation, and test statistics are kept separate; checkpoint selection never uses test labels.

Run the repository checks with:

```bash
python scripts/check_repository_contract.py
python -m compileall -q src train_drel_qrfme.py evaluate_drel_qrfme.py
pytest -q tests/test_drel_qrfme_full.py
```

## Documentation

| Topic | English | 中文 |
|---|---|---|
| DREL-QRFME theory, architecture, and evidence | [Full release](docs/drel_qrfme_full_release.md) | [完整说明](docs/drel_qrfme_full_release_zh-CN.md) |
| Data and reproduction | [Guide](docs/data_and_reproduction.md) | [指南](docs/data_and_reproduction_zh-CN.md) |
| Earlier C3-FaRNet evidence | [Result](docs/results_current_best.md) | [结果](docs/results_current_best_zh-CN.md) |
| DREL-E B validation study | [Method](docs/drel_algorithm.md) | [方法](docs/drel_algorithm_zh-CN.md) |

## Citation and license

If this repository supports your work, cite the repository URL and the original RSCD dataset paper. A method-specific BibTeX entry will be added when the manuscript is frozen. Code is released under the [MIT License](LICENSE); RSCD images remain subject to the dataset's original terms.
