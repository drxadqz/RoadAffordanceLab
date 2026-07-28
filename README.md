<p align="center">
  <img src="docs/assets/road-affordance-lab-hero.svg" alt="RoadAffordanceLab: factor-aware, physics-guided and auditable road-surface intelligence" width="100%" />
</p>

<h1 align="center">RoadAffordanceLab</h1>

<p align="center">
  <strong>Factor-aware, physics-guided road-surface recognition with an auditable PyTorch/CUDA experiment stack.</strong>
</p>

<p align="center">
  <a href="README_zh-CN.md">中文</a> ·
  <a href="#60-second-overview">60-second overview</a> ·
  <a href="#verified-result">Verified result</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/engineering.md">Engineering notes</a> ·
  <a href="MODEL_CARD.md">Model card</a>
</p>

<p align="center">
  <a href="https://github.com/drxadqz/rp/actions/workflows/release-contract.yml"><img alt="release contract" src="https://github.com/drxadqz/rp/actions/workflows/release-contract.yml/badge.svg" /></a>
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white" />
  <img alt="Top-1 90.63" src="https://img.shields.io/badge/verified%20Top--1-90.63%25-155EEF" />
  <img alt="Macro-F1 88.92" src="https://img.shields.io/badge/verified%20Macro--F1-88.92%25-0891B2" />
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-1F2937" /></a>
</p>

---

RoadAffordanceLab is an end-to-end research-engineering project for **27-class visual road-surface condition recognition** on RSCD. It treats a class such as `water_concrete_slight` as a structured combination of condition, material and roughness instead of an unrelated class ID.

The repository is intentionally more than a model definition. It includes config-driven training, CUDA mixed precision, resumable checkpoints, hard-pair error analysis, factor-level metrics, immutable result evidence and checkpoint provenance. These are the same engineering patterns needed to build reliable foundation-model training and evaluation systems: explicit contracts, controlled comparisons, failure slicing and reproducible artifacts.

> [!IMPORTANT]
> The numbers on this page are from the released **single-model S7 checkpoint** and the frozen 49,500-image RSCD test record. They are not presented as a new public SOTA claim. Active ARCQ/TACT research is excluded until its validation protocol is frozen and completed.

### Inspect the release in three commands

The public contract check is intentionally lightweight: it validates the committed metrics, per-class table, confusion matrix, README assets and checkpoint manifest without requiring RSCD images or a GPU.

```bash
git clone https://github.com/drxadqz/rp.git && cd rp
git lfs install && git lfs pull
python scripts/verify_release.py --check-checkpoints
```

## Project at a glance

| Area | What is implemented |
|---|---|
| Representation learning | ConvNeXt visual carrier + explicit physics/texture evidence + coupled factor head |
| Structured prediction | 27 classes decomposed into condition, material and roughness factors |
| Evaluation | Top-1, Macro-F1, per-class F1, factor accuracy, hard-pair boundaries and confusion matrices |
| Training systems | AMP, gradient accumulation, selective fine-tuning, checkpoint resume and deterministic manifests |
| Research rigor | Frozen configs, SHA256 checkpoint manifests, test isolation and promotion gates |
| Failure analysis | Weakest-class diagnosis, boundary-specific metrics and provenance-linked experiment evidence |

## 60-second overview

The central problem is not simply recognizing whether a road is bright or dark. Water can darken a surface, create specular reflection and suppress the same micro-texture that distinguishes `slight` from `severe` roughness. C3-FaRNet therefore combines three kinds of evidence:

1. **Visual context** from a ConvNeXt carrier.
2. **Physics/texture evidence** for reflection, dark water, local gradients and roughness cues.
3. **Factor interactions** that reason over condition × material × roughness instead of memorizing 27 unrelated labels.

<p align="center">
  <img src="docs/assets/c3-farnet-architecture.svg" alt="C3-FaRNet architecture: image to visual and physics evidence, factor coupling, hard-pair calibration and 27 classes" width="100%" />
</p>

For a detailed explanation, see the [English method note](docs/algorithm.md) or the [Chinese method note](docs/algorithm_zh.md).

## Verified result

The released checkpoint was evaluated once under the recorded full protocol: a single model, one crop, no ensemble and 49,500 test images.

| Metric | Verified value | What it answers |
|---|---:|---|
| Top-1 accuracy | **90.632%** | How often the first prediction is correct |
| Macro-F1 | **88.920%** | Whether performance is balanced across all 27 classes |
| Weighted-F1 | **90.654%** | Class-frequency-weighted classification quality |
| Friction/condition accuracy | **96.596%** | Whether dry/wet/water/snow/ice state is correct |
| Material accuracy | **97.210%** | Whether asphalt/concrete/mud/gravel is correct |
| Roughness accuracy | **95.176%** | Whether smooth/slight/severe is correct |
| Weakest-class F1 | **75.693%** | Performance on `water_concrete_slight` |

<p align="center">
  <img src="docs/assets/verified-metrics.svg" alt="Verified C3-FaRNet S7 metrics" width="100%" />
</p>

<details>
<summary><strong>Per-class F1 — all 27 classes</strong></summary>

<p align="center">
  <img src="docs/assets/per-class-f1.svg" alt="Per-class F1 for all 27 RSCD classes" width="100%" />
</p>

</details>

All chart values are generated directly from [`results/current_best_s7`](results/current_best_s7) by [`scripts/generate_readme_assets.py`](scripts/generate_readme_assets.py). The full evidence bundle includes the confusion matrix, per-class metrics, hard-pair metrics, predictions and training lineage.

### Release artifacts

| Artifact | Purpose |
|---|---|
| [`current_best_s7_public.yaml`](configs/c3_farnet/current_best_s7_public.yaml) | Portable model, data and training contract |
| [`best_checkpoint.pth`](checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth) | Released Git-LFS checkpoint artifact |
| [`results/current_best_s7`](results/current_best_s7) | Compact metrics, confusion matrix, hard pairs and per-class evidence |
| [`checkpoint_manifest.json`](results/s7_lineage/checkpoint_manifest.json) | SHA256 checkpoint ancestry and role inventory |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Intended use, limits and result provenance |

## What makes the work technically interesting

### 1. A structured label space, not a flat classifier

The class target is modeled as a factor tuple

$$
y=(f,m,r),
$$

where $f$ is road condition/friction state, $m$ is material and $r$ is roughness. The network jointly represents factor evidence and the coupled 27-class decision. This makes errors diagnosable: a `wet_concrete_slight` → `water_concrete_slight` error is a condition-boundary error, while a `slight` → `severe` error is a roughness-boundary error.

### 2. Explicit physical evidence alongside learned features

The model computes interpretable cues for luminance, saturation, specular response, dark-water evidence, gradients and Laplacian texture. These cues do not replace learned visual features; they are fused with the learned carrier and used to condition difficult boundaries.

### 3. Coupled evidence instead of naive concatenation

Low-rank pairwise and triple interactions combine condition, material and roughness tokens. Hard-pair experts are activated only around known ambiguous boundaries, keeping the main classifier general while making targeted residual corrections auditable.

### 4. Evaluation is treated as a system

The repository records the exact resolved config, data-manifest contract, checkpoint hashes, per-class predictions, factor errors and hard-pair metrics. Candidate mechanisms are compared under equal data, initialization, training horizon and evaluation precision before promotion.

## Quick start

### 1. Clone the code and checkpoint artifacts

```bash
git clone https://github.com/drxadqz/rp.git
cd rp
git lfs install
git lfs pull
```

### 2. Create the environment

```bash
conda env create -f environment-faf-paper.yml
conda activate faf_paper
pip install -e .
```

For a lightweight source install, use:

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Verify the public release contract

This check does not need RSCD images:

```bash
python scripts/verify_release.py --check-checkpoints
python scripts/generate_readme_assets.py --check
```

### 4. Prepare RSCD manifests

Copy [`configs/data/local_paths.example.yaml`](configs/data/local_paths.example.yaml) to `configs/data/local_paths.yaml`, point it at your local RSCD copy, then run:

```bash
python scripts/build_manifests.py \
  --config configs/data/local_paths.yaml \
  --out-dir data/manifests_full
```

The expected schema is documented in [Data and reproduction](docs/data_and_reproduction.md). Raw RSCD images are not redistributed by this repository.

### 5. Evaluate the released model

```bash
python test.py \
  --config configs/c3_farnet/current_best_s7_public.yaml \
  --checkpoint checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth
```

### 6. Reproduce the selective fine-tuning recipe

```bash
python train.py --config configs/c3_farnet/current_best_s7_public.yaml
```

The full checkpoint lineage and the exact distinction between the parent model, S7 checkpoint and source-router inference record are described in [S7 training lineage](docs/s7_training_lineage.md).

## Repository map

```text
.
├── checkpoints/              # Git-LFS model artifacts with SHA256 manifests
├── configs/                  # portable training/evaluation contracts
├── docs/                     # method, model card, data and engineering notes
├── recovery/                 # historical lineage archive; not the quick-start path
├── results/
│   ├── current_best_s7/       # compact machine-readable result evidence
│   └── s7_lineage/            # full training/evaluation provenance
├── scripts/                  # data, evaluation, audit and visualization tools
├── src/friction_affordance/  # PyTorch model and experiment implementation
├── train.py
├── validate.py
└── test.py
```

## Research-engineering highlights

These parts of the project transfer directly to large-model research and evaluation work:

| Foundation-model workflow | Corresponding implementation in this project |
|---|---|
| Dataset and prompt contracts | Immutable CSV manifests and factor-label sanity checks |
| Fine-tuning | Selective parameter training with explicit trainable-prefix control |
| Scalable training | CUDA AMP, gradient accumulation, workers/prefetch and resumable checkpoints |
| Evaluation harness | Global, per-class, factor and hard-boundary metrics |
| Model behavior analysis | Confusion slicing and weakest-class diagnosis |
| Reproducible experiments | Resolved configs, SHA256 artifacts and promotion gates |
| Honest model reporting | Model card, data boundary and non-SOTA disclosure |

See [Engineering notes](docs/engineering.md) for implementation details and a short code-reading path.

## Documentation

| Document | Purpose |
|---|---|
| [Algorithm (English)](docs/algorithm.md) | C3-FaRNet architecture and module behavior |
| [算法详解（中文）](docs/algorithm_zh.md) | 中文结构、原理与公式说明 |
| [Model card](MODEL_CARD.md) | Intended use, metrics, limitations and ethical boundary |
| [Data and reproduction](docs/data_and_reproduction.md) | Manifest schema and local data preparation |
| [Engineering notes](docs/engineering.md) | Training/evaluation stack and code-reading guide |
| [Results](docs/results_current_best.md) | Verified metrics and evidence locations |
| [Training lineage](docs/s7_training_lineage.md) | Parent/teacher/S7 provenance |
| [Release inventory](docs/s7_release_inventory.md) | Exact files, checkpoints and hashes |

## Scientific and safety boundary

This model estimates **visual road-surface state and visual friction affordance**. RSCD labels are visual proxy labels, not synchronized tire-force or friction-meter measurements. It must not be used as a direct friction-coefficient sensor or as the sole component of a safety-critical driving decision.

See the [model card](MODEL_CARD.md) and [data documentation](docs/data_and_reproduction.md) before deployment or cross-dataset evaluation.

## Citation

If this repository supports your work, cite it using [`CITATION.cff`](CITATION.cff). For the RSCD dataset itself, cite the original [IEEE T-ITS paper](https://doi.org/10.1109/TITS.2023.3264588).

## License

Code in this repository is released under the [MIT License](LICENSE). Dataset images and third-party model/data assets remain subject to their original licenses.
