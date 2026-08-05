<div align="center">

![RoadAffordanceLab banner](assets/road-affordance-lab-banner.svg)

[**English**](README.md) | [**简体中文**](README_zh-CN.md) | [Architecture](docs/algorithm.md) | [DREL](docs/drel_algorithm.md) | [Reproducibility](docs/data_and_reproduction.md) | [Evidence](docs/results_current_best.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](environment-faf-paper.yml)
[![Repository quality](https://github.com/drxadqz/RoadAffordanceLab/actions/workflows/repository-quality.yml/badge.svg)](https://github.com/drxadqz/RoadAffordanceLab/actions/workflows/repository-quality.yml)
[![License](https://img.shields.io/github/license/drxadqz/RoadAffordanceLab?color=155EEF)](LICENSE)
[![Reproducible research](https://img.shields.io/badge/research-auditable-0891B2)](docs/data_and_reproduction.md)

**Factor-aware, physics-guided road-surface recognition with an auditable PyTorch training and evaluation stack.**

</div>

## Why this project exists

Road-surface recognition is not ordinary object classification. Material identity depends on global appearance, while wetness and roughness often depend on weak local texture, glare, dark water, and partially erased detail. The 27 RSCD labels are therefore treated as coupled physical states rather than unrelated class names:

```text
water_concrete_slight = water condition + concrete material + slight roughness
```

**C3-FaRNet** (Coupled Conditioned Friction-Affordance Road Network) learns condition, material, and roughness evidence, then models their pairwise and higher-order interactions. The repository includes the released S7 model, its checkpoint lineage, configuration-driven training and evaluation, per-class evidence, and recovery materials from the original training machine.

> Scientific scope: this is **visual road friction-affordance estimation** from images. RSCD labels are visual proxy labels, not synchronized tire-force or friction-meter measurements.

## Verified result

![Verified S7 result summary](assets/verified-results.svg)

The values below come from the released self-contained S7 checkpoint on the historical 49,500-image RSCD test split. They are not ARCQ development-subset scores and are not produced by an ensemble or test-time augmentation.

| Model | Protocol | Top-1 | Macro-F1 | Weakest-class F1 | Parameters |
|---|---|---:|---:|---:|---:|
| **C3-FaRNet-S7** | 958,941 train / 19,860 val / 49,500 test | **90.632%** | **88.920%** | **75.693%** | 32.49M total / 1.09M trainable |

- 27 coupled RSCD states
- single model, single crop
- no TTA and no model ensemble
- weakest class: `water_concrete_slight`
- full class metrics, confusion matrix, and hard-pair results are versioned under [`results/current_best_s7`](results/current_best_s7)

Read the exact evidence boundary in [Current verified result](docs/results_current_best.md) and the checkpoint ancestry in [S7 training lineage](docs/s7_training_lineage.md).

## Architecture at a glance

![C3-FaRNet architecture](assets/c3-farnet-overview.svg)

The model combines four ideas:

1. **Conditioned visual backbone** — preserves task-relevant texture and boundary evidence before the final classifier.
2. **Local physics evidence** — exposes differentiable cues for specularity, dark water, texture erasure, gradients, and local contrast.
3. **Factorized prediction** — reasons about condition, material, and roughness instead of memorizing 27 opaque labels.
4. **Coupled decision layer** — models factor interactions and difficult one-factor boundaries such as wet versus water and slight versus severe.

For the full forward path, equations, and implementation mapping, see [Architecture](docs/algorithm.md) or [中文架构说明](docs/algorithm_zh.md).

## Thirty-second tour

```text
configs/                     Reproducible experiment definitions
checkpoints/                 Released Git-LFS checkpoints
src/friction_affordance/     Models, losses, data, metrics, and training engine
results/current_best_s7/     Machine-readable full-test evidence
results/s7_lineage/          Checkpoint and experiment provenance
recovery/                    Historical source, manifests, and environment records
docs/                        Method, evidence, and reproduction documentation
train.py / validate.py / test.py
                             Stable command-line entry points
```

This layout is intentionally useful both as a research release and as an ML-engineering portfolio: the public claim is linked to a config, checkpoint, metrics payload, data protocol, and environment record rather than only to a screenshot.

## Quick start

### 1. Install

```bash
git clone https://github.com/drxadqz/RoadAffordanceLab.git
cd RoadAffordanceLab
git lfs install
git lfs pull

conda env create -f environment-faf-paper.yml
conda activate faf_paper
pip install -e .
```

Minimal pip installation is also supported:

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare manifests

Copy the local path template and point it at your RSCD download:

```bash
cp configs/data/local_paths.example.yaml configs/data/local_paths.yaml
python scripts/build_manifests.py \
  --config configs/data/local_paths.yaml \
  --out-dir data/manifests_full
```

The expected schema and split contract are documented in [Reproducibility](docs/data_and_reproduction.md). Raw RSCD images are not redistributed by this repository.

### 3. Evaluate the released S7 checkpoint

```bash
python test.py \
  --config configs/c3_farnet/current_best_s7_public.yaml \
  --checkpoint checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth \
  --output-dir outputs/reproduce_s7
```

### 4. Train from the released recipe

```bash
python train.py --config configs/c3_farnet/current_best_s7_public.yaml
```

The verified S7 run is a warm-start experiment. Fetch the Git-LFS files first and read [Parent model training](docs/parent_model_training.md) before interpreting a reproduction result.

## Engineering highlights

| Capability | What is included |
|---|---|
| Config-driven experimentation | YAML-defined data, architecture, losses, optimization, and evaluation |
| Evaluation discipline | Top-1, Macro-F1, per-class F1, hard-pair analysis, and confusion matrices |
| Provenance | Checkpoint manifests, historical environment exports, run histories, and recovery notes |
| GPU training | PyTorch mixed precision, gradient accumulation, CUDA-oriented environment capture |
| Modular modeling | Backbone registry, factor heads, physics branches, coupled heads, and extensible losses |
| Release safety | Repository contract checks, private-path scanning, documentation links, and CI |

These are the same engineering concerns that matter in larger model systems: controlled experiments, deterministic interfaces, evaluation beyond one aggregate score, artifact lineage, and reproducible GPU environments.

## Research status

- **Released:** C3-FaRNet-S7 and its verified full-test evidence.
- **Recovered:** historical manifests, source snapshot, environment records, and the surviving checkpoint chain.
- **Released as validation-only research code:** the clean DREL-E B `component_full` production candidate, its frozen D350 specification, tests, machine-readable three-seed validation evidence, and full bilingual explanation.
- **Active research:** roughness-targeted DREL follow-up studies. An unfinished route is not merged into the production model until it passes its frozen gate.

The separation is deliberate: attractive plots are useful, but only frozen evidence should become a public performance claim.

### DREL production candidate (D350 validation only)

The newly released [`DREL-E B component_full`](docs/drel_algorithm.md) implementation is a 2.76M-parameter from-scratch classifier with a one-way directional/radial evidence ledger, moment-preserving transitions, regional mean/deviation blocks, constrained matched-response filters, and a bounded `0.25 × tanh` evidence write.

At equal Gate8 D350 validation budget, its three-seed mean is above the frozen single-seed RSPNet-M/L envelope on eight of nine tracked metrics. The remaining exception is roughness: `0.586420` versus `0.608889` (−2.247 percentage points). This is a validation-screen result, not a historical 49,500-image formal-test claim, and it does not replace the S7 table above.

```python
import torch
from friction_affordance.models.drel import build_drel_component_full

model = build_drel_component_full(num_classes=27, head_init_seed=970027)
logits = model(torch.randn(2, 3, 360, 240))
```

Read the [complete DREL algorithm guide](docs/drel_algorithm.md) and the [claim-safe validation evidence](docs/drel_validation_evidence.md).

## Documentation

| Document | English | 中文 |
|---|---|---|
| Method and architecture | [Architecture](docs/algorithm.md) | [算法与架构](docs/algorithm_zh.md) |
| Data and reproduction | [Reproducibility](docs/data_and_reproduction.md) | [数据与复现](docs/data_and_reproduction_zh-CN.md) |
| Verified evidence | [Results](docs/results_current_best.md) | [验证结果](docs/results_current_best_zh-CN.md) |
| DREL production algorithm | [DREL guide](docs/drel_algorithm.md) | [DREL 完整说明](docs/drel_algorithm_zh-CN.md) |
| DREL validation evidence | [DREL evidence](docs/drel_validation_evidence.md) | [DREL 验证证据](docs/drel_validation_evidence_zh-CN.md) |
| Checkpoint ancestry | [Training lineage](docs/s7_training_lineage.md) | — |
| Release inventory | [Inventory](docs/s7_release_inventory.md) | — |
| Recovery boundary | [Recovery status](recovery/RECOVERY_STATUS.md) | — |

## Repository integrity

Run the same lightweight contract check used by CI:

```bash
python scripts/check_repository_contract.py
python -m compileall -q src scripts/check_repository_contract.py
```

The check verifies bilingual navigation, required public artifacts, local Markdown links, and accidental machine-specific absolute paths in the portfolio-facing files.

## Citation

If this repository supports your work, please cite the project URL and the RSCD paper used for the dataset protocol. A paper-specific BibTeX entry will be added after the ARCQ method and its final experimental protocol are frozen.

## License

Code is released under the [MIT License](LICENSE). Dataset images remain subject to the original RSCD terms.
