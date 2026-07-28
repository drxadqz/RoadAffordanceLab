<p align="center">
  <img src="assets/hero.svg" alt="C3-FaRNet — factor-aware road surface intelligence" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/drxadqz/rp/actions/workflows/public-release-check.yml"><img src="https://github.com/drxadqz/rp/actions/workflows/public-release-check.yml/badge.svg" alt="Public release check" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch and CUDA" />
  <img src="https://img.shields.io/badge/task-27--class%20fine--grained%20recognition-155EEF" alt="27-class recognition" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22C55E" alt="MIT License" /></a>
</p>

<p align="center">
  <b>English</b> · <a href="README_zh-CN.md">简体中文</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="docs/algorithm.md">Method</a> ·
  <a href="docs/results_current_best.md">Results</a> ·
  <a href="docs/engineering_case_study.md">Engineering case study</a>
</p>

## Overview

**C3-FaRNet** (Coupled Conditioned Friction-Affordance Road Network) is a research-grade PyTorch system for fine-grained visual road-state recognition. It addresses a task in which a prediction is not just a material label: each of the 27 RSCD states couples **surface condition**, **material**, and **roughness**.

The project combines:

- a ConvNeXt-Tiny visual carrier with task-specific early/mid feature conditioning;
- differentiable global, local, and semantic physics-evidence branches;
- factor-aware single, pairwise, and triple interactions;
- calibrated hard-pair corrections for visually adjacent states;
- reproducible CUDA training, checkpoint lineage, per-class diagnostics, and release audits.

This repository is deliberately presented as both a **representation-learning project** and an **ML systems case study**. The headline metrics below come only from the checked-in, self-contained S7 checkpoint and its frozen 49,500-image evaluation artifacts.

## Verified result

<p align="center">
  <img src="assets/results-overview.svg" alt="Verified C3-FaRNet S7 result overview" width="100%" />
</p>

| model record | Top-1 | Macro-F1 | weighted F1 | test images | interpretation |
|---|---:|---:|---:|---:|---|
| **C3-FaRNet S7, self-contained checkpoint** | **90.6323%** | **88.9197%** | **90.6539%** | 49,500 | Primary reportable result |
| Parent checkpoint + source-reliable router | 90.6404% | 88.9410% | — | 49,500 | Inference variant; not a separately trained checkpoint |

The full per-class table, confusion matrix, predictions, resolved configuration, logs, checkpoint SHA256 values, and lineage are included in [`results/`](results/) and documented in [`docs/s7_release_inventory.md`](docs/s7_release_inventory.md). The weakest class is `water_concrete_slight` at 75.6931% F1; the repository reports it rather than hiding it behind Top-1.

> **Claim boundary.** These are verified records under the historical RSCD 192×192 letterbox protocol. They are not presented as a cross-protocol SOTA claim. RSCD labels are visual road-state proxies, not synchronized tire-force measurements.

<details>
<summary><b>Show the full per-class F1 chart</b></summary>

<p align="center">
  <img src="assets/per-class-f1.svg" alt="Per-class F1 for the verified S7 checkpoint" width="100%" />
</p>

</details>

<details>
<summary><b>Show the normalized 27-class confusion matrix</b></summary>

<p align="center">
  <img src="assets/confusion-matrix.svg" alt="Normalized 27-class confusion matrix" width="100%" />
</p>

</details>

## Why a coupled model?

A class such as `water_concrete_slight` is represented as a structured state

$$
y=(f,m,r),
$$

where $f$ is condition/friction state, $m$ is material, and $r$ is roughness. A flat additive classifier cannot fully express that wet slight concrete looks different from the independent sum of “wet”, “concrete”, and “slight”. C3-FaRNet therefore scores compatible states with single-factor, pairwise, and triple interactions:

$$
Z(f,m,r)=A_f+B_m+C_r+D_{fm}+E_{fr}+G_{mr}+H_{fmr}.
$$

The model also retains raw appearance and differentiable evidence for glare, dark water, local gradients, texture loss, snow/ice-like brightness, and regional connectedness. Corrections are activated only near predefined hard boundaries, rather than globally rewriting every class score.

<p align="center">
  <img src="assets/architecture.svg" alt="C3-FaRNet architecture" width="100%" />
</p>

Read the complete explanation in [`docs/algorithm.md`](docs/algorithm.md) or [`docs/algorithm_zh.md`](docs/algorithm_zh.md).

## What this repository demonstrates

| area | concrete evidence in the repository |
|---|---|
| **Representation learning** | conditioned visual hierarchy, global/local physics fields, structured factor embeddings, low-rank tensor interactions |
| **Model evaluation** | Top-1, Macro-F1, factor accuracy, hard-pair accuracy, per-class F1, confusion matrices, failure attribution |
| **ML systems** | mixed precision, gradient accumulation, selective parameter training, resumable checkpoints, manifest-driven data loading |
| **Reproducibility** | resolved configs, checkpoint hashes, Git LFS weights, environment snapshot, historical manifests, release inventory |
| **Research discipline** | explicit claim boundaries, negative-result gates, frozen evaluation protocols, weakest-class reporting |
| **Interpretability** | physics-cue summaries, factor-level confusion analysis, local evidence fields, feature/failure audit scripts |

These are the same transferable capabilities needed in modern multimodal and foundation-model engineering: building a measurable representation hypothesis, implementing it in PyTorch, designing reliable evaluations, instrumenting model internals, and preserving evidence that another engineer can audit.

## Quickstart

### 1. Clone and fetch checkpoints

```bash
git clone https://github.com/drxadqz/rp.git
cd rp
git lfs pull
```

The selected S7 checkpoint, its direct parent, and its frozen teacher are stored with Git LFS. The RSCD images are **not** redistributed.

### 2. Create the environment

```bash
conda env create -f environment-faf-paper.yml
conda activate faf_paper
pip install -e .
```

Minimal pip installation:

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Prepare RSCD manifests

```bash
cp configs/data/local_paths.example.yaml configs/data/local_paths.yaml
# Edit local_paths.yaml, then:
python scripts/build_manifests.py \
  --config configs/data/local_paths.yaml \
  --out-dir data/manifests_full
```

The required schema and label canonicalization rules are documented in [`docs/data_and_reproduction.md`](docs/data_and_reproduction.md).

### 4. Evaluate the frozen S7 checkpoint

```bash
python test.py \
  --config configs/c3_farnet/current_best_s7_public.yaml \
  --checkpoint checkpoints/c3_farnet_formal_fullmanifest_source_reliable_router_s7_20260709/best_checkpoint.pth
```

### 5. Re-run the historical continuation recipe

```bash
python train.py --config configs/c3_farnet/current_best_s7_public.yaml
```

This is a one-epoch full-manifest continuation from the checked-in parent and teacher checkpoints, not a from-scratch recipe. See [`docs/s7_training_lineage.md`](docs/s7_training_lineage.md) for the exact chain and naming boundary.

### 6. Rebuild and verify the public evidence

```bash
python scripts/build_public_assets.py
python scripts/verify_public_release.py
```

The release check regenerates all README plots directly from the result payloads and verifies sample totals, matrix dimensions, local links, and headline metrics.

## Repository map

```text
assets/                         README figures generated from result files
checkpoints/                    selected model artifacts tracked by Git LFS
configs/c3_farnet/              portable S7 and parent/router configurations
docs/                           algorithm, results, lineage, and case-study docs
recovery/                       provenance archive from the historical machine
results/current_best_s7/        compact 49,500-image evaluation evidence
results/s7_lineage/             detailed parent → S7 → router evidence chain
scripts/                        data, evaluation, diagnosis, and release tooling
src/friction_affordance/        model, data, loss, engine, and metric code
train.py / validate.py / test.py
```

The large [`recovery/`](recovery/) subtree is an auditable provenance archive, not the recommended entry point for readers. Start with the files linked from this README and [`docs/s7_release_inventory.md`](docs/s7_release_inventory.md).

## Research status

- **Released / verified:** C3-FaRNet S7 and the parent-plus-router inference record.
- **Measured bottleneck:** roughness contributes to 51.50% of S7 errors; water/wet concrete slight–severe boundaries remain the hardest group.
- **Active research:** self-developed ARCQ/TACT road backbones and same-protocol public-baseline controls. They are intentionally excluded from headline claims until they pass the frozen comparison gate.
- **Not claimed:** direct physical friction-coefficient estimation, cross-protocol SOTA, or an ARCQ improvement over RSPNet.

See [`docs/research_status.md`](docs/research_status.md) for the promotion rules and current boundary between verified release and ongoing experiments.

## Documentation

- [`docs/engineering_case_study.md`](docs/engineering_case_study.md) — system design, ownership, and interview-level technical narrative
- [`docs/algorithm.md`](docs/algorithm.md) / [`docs/algorithm_zh.md`](docs/algorithm_zh.md) — complete method explanation
- [`docs/results_current_best.md`](docs/results_current_best.md) — metric definitions and verified result
- [`docs/data_and_reproduction.md`](docs/data_and_reproduction.md) — dataset and manifest contract
- [`docs/s7_training_lineage.md`](docs/s7_training_lineage.md) — checkpoint ancestry and training recipe
- [`docs/s7_release_inventory.md`](docs/s7_release_inventory.md) — source/config/checkpoint/result inventory
- [`recovery/README.md`](recovery/README.md) — provenance archive boundary

## Contributing

Bug reports, reproducibility issues, and mechanism-level ablations are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening an issue or pull request.

## License

Code is released under the [MIT License](LICENSE). Dataset files may have separate terms from their original distributor and are not included here.
